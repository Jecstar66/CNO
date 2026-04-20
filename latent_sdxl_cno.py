import os
import copy
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionXLPipeline

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

class SDXL():
    def __init__(self, 
                 solver_config: dict,
                 model_key:str="stabilityai/stable-diffusion-xl-base-1.0",
                 dtype=torch.float16,
                 device='cuda',
                 seed: Optional[int]=42,
                 **kwargs):

        self.device = device
        self.dtype = dtype

        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype).to(device)

        self.tokenizer_1 = copy.deepcopy(pipe.tokenizer)
        self.tokenizer_2 = copy.deepcopy(pipe.tokenizer_2)
        self.text_enc_1 = copy.deepcopy(pipe.text_encoder)
        self.text_enc_2 = copy.deepcopy(pipe.text_encoder_2)
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod]).to(device)

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc):
        text_inputs = tokenizer(
            prompt, padding='max_length', max_length=tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        )
        prompt_embeds = text_enc(text_inputs.input_ids.to(self.device), output_hidden_states=True)
        pool_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2] # SDXL uses penultimate layer
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(self, null_prompts, prompts):
        prompt_embed_1, pool_prompt_1 = self._text_embed(prompts, self.tokenizer_1, self.text_enc_1)
        prompt_embed_2, pool_prompt_2 = self._text_embed(prompts, self.tokenizer_2, self.text_enc_2)
        
        null_embed_1, pool_null_1 = self._text_embed(null_prompts, self.tokenizer_1, self.text_enc_1)
        null_embed_2, pool_null_2 = self._text_embed(null_prompts, self.tokenizer_2, self.text_enc_2)

        prompt_embeds = torch.concat([prompt_embed_1, prompt_embed_2], dim=-1)
        null_embeds = torch.concat([null_embed_1, null_embed_2], dim=-1)

        return null_embeds, prompt_embeds, pool_null_2, pool_prompt_2

    def _get_add_time_ids(self, original_size, crops_coords, target_size, dtype, b_size):
        add_time_ids = list(original_size + crops_coords + target_size)
        add_time_ids = torch.tensor([add_time_ids] * b_size, dtype=dtype)
        return add_time_ids

    def decode(self, zt):
        return self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()

    def predict_noise(self, zt, t, uc, c, add_cond_kwargs):
        if uc is None:
            t_in = t.expand(zt.shape[0])
            noise_c = self.unet(zt, t_in, encoder_hidden_states=c, added_cond_kwargs=add_cond_kwargs)['sample']
            return None, noise_c
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = t.expand(zt.shape[0]).repeat(2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed, added_cond_kwargs=add_cond_kwargs)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
            return noise_uc, noise_c

    @torch.enable_grad()
    def iopt_diverse(self, zt, ts, uc, c, cfg_guidance, etc_kwargs, add_cond_kwargs):
        b_size = len(zt)
        zt_opt = zt.clone().detach().requires_grad_(True)
        optimizer = Adam([zt_opt], lr=etc_kwargs['i_opt_lr'])

        at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
        
        temperature = etc_kwargs['infoNCE_temp']
        gamma = etc_kwargs['gamma']
        w = etc_kwargs['window_size']

        with torch.no_grad():
            noise_uc, noise_c = self.predict_noise(zt_opt, ts, uc, c, add_cond_kwargs)
            noise_pred_base = noise_uc + cfg_guidance * (noise_c - noise_uc)
            z0t_base = (zt_opt - (1 - at).sqrt() * noise_pred_base) / at.sqrt()
            z0t_base = z0t_base.detach()

        for _ in range(etc_kwargs['i_opt_iter']):
            noise_uc, noise_c = self.predict_noise(zt_opt, ts, uc, c, add_cond_kwargs)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            z0t = (zt_opt - (1 - at).sqrt() * noise_pred) / at.sqrt()

            z0t_base_pooled = F.adaptive_avg_pool2d(z0t_base, (w, w))
            z0t_pooled = F.adaptive_avg_pool2d(z0t, (w, w))

            B, C_dim, w_dim, _ = z0t_base_pooled.shape
            P = w_dim * w_dim

            z0t_base_vecs = F.normalize(z0t_base_pooled.view(B, C_dim, P), dim=1)
            z0t_vecs = F.normalize(z0t_pooled.view(B, C_dim, P), dim=1)

            diag_sim_matrix = torch.einsum('bcp,bcp->bp', z0t_base_vecs, z0t_vecs) / temperature
            non_diag_sim_matrix = torch.einsum('bcp, dcp -> pbd', z0t_vecs, z0t_vecs) / temperature
            sim_matrix = non_diag_sim_matrix.clone()

            diag_indices = torch.arange(B, device=sim_matrix.device)
            patch_indices = torch.arange(P, device=sim_matrix.device)

            sim_matrix[patch_indices[:, None], diag_indices, diag_indices] = (diag_sim_matrix / gamma).T

            sim_exp = torch.exp(sim_matrix)
            denominator = sim_exp.sum(dim=-1)
            numerator = sim_exp[patch_indices[:, None], diag_indices, diag_indices]

            loss_each = -torch.log(numerator / (denominator + 1e-8))
            loss = loss_each.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return zt_opt.detach()


@register_solver('ddim')
class BaseDDIM(SDXL):
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def batch_sample(self,
                     cfg_guidance=7.5,
                     prompts=[""],
                     null_prompts=[""],
                     etc_kwargs=None,
                     **kwargs):
        
        b_size = len(prompts)
        
        # 1. 텍스트 임베딩 추출 (SDXL은 Pooled Embed도 필요함)
        uc, c, pool_uc, pool_c = self.get_text_embed(null_prompts, prompts)

        # 2. SDXL 전용 Add_Cond_kwargs 세팅
        height = width = self.default_sample_size * self.vae_scale_factor
        add_time_ids = self._get_add_time_ids((height, width), (0, 0), (height, width), dtype=c.dtype, b_size=b_size)
        
        # CFG를 위한 Concat
        add_text_embeds = torch.cat([pool_uc, pool_c], dim=0)
        add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            'text_embeds': add_text_embeds.to(self.device),
            'time_ids': add_time_ids_cfg.to(self.device)
        }

        # 3. 초기 노이즈 Z_T 생성
        zt = torch.randn((b_size, 4, height // self.vae_scale_factor, width // self.vae_scale_factor), 
                         device=self.device, generator=self.generator)
        
        # 4. Diffusion Sampling & CNO
        pbar = tqdm(self.scheduler.timesteps, desc="SDXL Generation (CNO)")
        for step, t in enumerate(pbar):
            ts = torch.full((b_size,), t, device=self.device, dtype=torch.long)
            at = self.scheduler.alphas_cumprod[ts].view(b_size, 1, 1, 1)
            at_prev = self.scheduler.alphas_cumprod[ts - self.skip].view(b_size, 1, 1, 1)

            # [핵심] Step 0에서 CNO Optimization 수행
            if step == 0 and etc_kwargs.get('use_cno', True):
                zt = self.iopt_diverse(
                    zt.detach(), ts, uc, c, cfg_guidance, etc_kwargs, add_cond_kwargs
                )

            # DDIM 노이즈 예측
            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, ts, uc, c, add_cond_kwargs)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # Tweedie x0 계산 및 다음 스텝
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

        # 5. 디코딩
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
