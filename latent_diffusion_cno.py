import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm
from torch.optim.adam import Adam
import torch.nn.functional as F
from typing import Dict, Optional
import math

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

########################

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="botp/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 seed: Optional[int]=42,
                 **kwargs):
        self.device = device
        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.model_key = model_key

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        
        total_timesteps = len(self.scheduler.config.num_train_timesteps if hasattr(self.scheduler.config, 'num_train_timesteps') else 1000)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def get_text_embed(self, null_prompt: list[str], prompt: list[str]):
        null_input = self.tokenizer(null_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors="pt", truncation=True).to(self.device)
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors="pt", truncation=True).to(self.device)

        null_embed = self.text_encoder(null_input.input_ids)[0]
        text_embed = self.text_encoder(text_input.input_ids)[0]
        return null_embed, text_embed

    @torch.no_grad()
    def decode(self, zt):
        zt = zt.to(self.vae.device) / 0.18215
        img = self.vae.decode(zt.to(self.vae.dtype)).sample.float()
        return img

    def predict_noise(self, zt: torch.Tensor, t: torch.Tensor, uc: torch.Tensor, c: torch.Tensor):
        t_in = t.unsqueeze(0) if len(t.shape) == 0 else t
        c_embed = torch.cat([uc, c], dim=0)
        z_in = torch.cat([zt] * 2) 
        t_in = torch.cat([t_in] * 2)
        noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
        noise_uc, noise_c = noise_pred.chunk(2)
        return noise_uc, noise_c

    @torch.enable_grad()
    def iopt_diverse(self, zt, ts, uc, c, cfg_guidance, etc_kwargs):
        """
        CNO 코어 로직: Step 0에서 InfoNCE Loss를 사용해 노이즈 배치를 최적화합니다.
        """
        b_size = len(zt)
        zt_opt = zt.clone().detach().requires_grad_(True)
        optimizer = Adam([zt_opt], lr=etc_kwargs['i_opt_lr'])

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        temperature = etc_kwargs['infoNCE_temp']
        gamma = etc_kwargs['gamma']
        w = etc_kwargs['window_size']

        # 1. Base (Reference) 예측
        with torch.no_grad():
            noise_uc, noise_c = self.predict_noise(zt_opt, ts, uc, c)
            noise_pred_base = noise_uc + cfg_guidance * (noise_c - noise_uc)
            z0t_base = (zt_opt - (1 - at).sqrt() * noise_pred_base) / at.sqrt()
            z0t_base = z0t_base.detach()

        # 2. CNO Optimization Loop
        for i in range(etc_kwargs['i_opt_iter']):
            # Tweedie 예측
            noise_uc, noise_c = self.predict_noise(zt_opt, ts, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            z0t = (zt_opt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            # Window Pooling (Patch 추출)
            z0t_base_pooled = F.adaptive_avg_pool2d(z0t_base, (w, w))
            z0t_pooled = F.adaptive_avg_pool2d(z0t, (w, w))

            B, C_dim, w_dim, _ = z0t_base_pooled.shape
            P = w_dim * w_dim

            # InfoNCE 계산을 위한 벡터 평탄화 및 정규화
            z0t_base_vecs = F.normalize(z0t_base_pooled.view(B, C_dim, P), dim=1)
            z0t_vecs = F.normalize(z0t_pooled.view(B, C_dim, P), dim=1)

            # Cosine Similarity 행렬 계산
            diag_sim_matrix = torch.einsum('bcp,bcp->bp', z0t_base_vecs, z0t_vecs) / temperature
            non_diag_sim_matrix = torch.einsum('bcp, dcp -> pbd', z0t_vecs, z0t_vecs) / temperature
            sim_matrix = non_diag_sim_matrix.clone()

            diag_indices = torch.arange(B, device=sim_matrix.device)
            patch_indices = torch.arange(P, device=sim_matrix.device)

            # Positive pair 대각 성분에 gamma 적용
            sim_matrix[patch_indices[:, None], diag_indices, diag_indices] = (diag_sim_matrix / gamma).T

            # Softmax & InfoNCE Loss 계산
            sim_exp = torch.exp(sim_matrix)
            denominator = sim_exp.sum(dim=-1)
            numerator = sim_exp[patch_indices[:, None], diag_indices, diag_indices]

            loss_each = -torch.log(numerator / (denominator + 1e-8))
            loss = loss_each.mean()

            # 역전파 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return zt_opt.detach()


@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def batch_sample(self,
                     cfg_guidance=7.5,
                     prompts=[""],
                     null_prompts=[""],
                     etc_kwargs=None,
                     **kwargs):
        
        b_size = len(prompts)
        
        # 1. 텍스트 임베딩
        uc, c = self.get_text_embed(null_prompt=null_prompts, prompt=prompts)

        # 2. 초기 노이즈 Z_T 생성
        zt = torch.randn((b_size, 4, 64, 64), device=self.device, generator=self.generator)
        
        # 3. Diffusion Sampling & CNO
        pbar = tqdm(self.scheduler.timesteps, desc="SD Generation (CNO)")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)
            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)
                
            # [핵심] Step 0에서 CNO Optimization 수행
            if step == 0 and etc_kwargs.get('use_cno', True):
                zt = self.iopt_diverse(
                    zt.detach(),
                    ts,
                    uc,
                    c,
                    cfg_guidance, 
                    etc_kwargs
                )
            
            # DDIM 노이즈 예측
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
                    
            # Tweedie x0 계산 및 다음 스텝으로 진행
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

        # 4. 이미지 디코딩
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
