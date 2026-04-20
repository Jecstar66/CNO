import argparse
from pathlib import Path
import os

from munch import munchify
from torchvision.utils import save_image
from latent_diffusion_cno import get_solver
from itertools import islice
from tqdm import tqdm

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def main():
    parser = argparse.ArgumentParser(description="Contrastive Noise Optimization (CNO) Text-to-Image")
    # Base Generation Params
    parser.add_argument("--workdir", type=Path, default="examples/workdir/cno_output")
    parser.add_argument('--prompt_dir', type=Path, default=Path('examples/assets/coco_v1.txt'))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--null_prompt", type=str, default="")
    parser.add_argument("--cfg_guidance", type=float, default=7.5)
    parser.add_argument("--NFE", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b_size", type=int, default=4, help="Batch size (CNO requires > 1 for diversity)")
    parser.add_argument("--num_samples", type=int, default=10000)
    
    # CNO (InfoNCE) Specific Params
    parser.add_argument("--use_cno", action='store_true', default=True, help="Enable CNO")
    parser.add_argument("--i_opt_iter", type=int, default=3, help="CNO optimization steps")
    parser.add_argument("--i_opt_lr", type=float, default=1e-3, help="CNO learning rate")
    parser.add_argument("--window_size", type=int, default=1, choices=[1,2,4,8,16,32,64], help="InfoNCE patch window size")
    parser.add_argument("--gamma", type=float, default=1.0, help="InfoNCE positive pair scaling factor")
    parser.add_argument("--infoNCE_temp", type=float, default=0.1, help="InfoNCE Temperature")
    
    args = parser.parse_args()

    # Pack CNO specific params
    etc_kwargs = {
        "use_cno": args.use_cno,
        "i_opt_iter": args.i_opt_iter,
        "i_opt_lr": args.i_opt_lr,
        "window_size": args.window_size,
        "gamma": args.gamma,
        "infoNCE_temp": args.infoNCE_temp,
    }

    print(f"Starting CNO with params: {etc_kwargs}")
    os.makedirs(args.workdir / 'result', exist_ok=True)
    
    # Load Prompts
    text_list = []
    with open(args.prompt_dir, 'r') as f:
        for line in f:
            if line.strip():
                text_list.append(line.strip())
    text_list = text_list[:args.num_samples]
    
    # Repeat prompts across batch size to enforce diversity for the SAME prompt
    text_list = [text for text in text_list for _ in range(args.b_size)]
    text_list_batched = list(chunk(text_list, args.b_size))

    # Initialize Solver
    solver_config = munchify({'num_sampling': args.NFE})
    model_key = "botp/stable-diffusion-v1-5" 
    
    solver = get_solver('ddim',
                        solver_config=solver_config,
                        model_key=model_key,
                        device=args.device,
                        seed=args.seed)

    img_count = 0
    for i, prompts in enumerate(tqdm(text_list_batched, desc='Batch')):
        prompts = list(prompts)
        null_prompts = len(prompts) * [args.null_prompt]
        
        # Generation
        result = solver.batch_sample(prompts=prompts,
                                     null_prompts=null_prompts,
                                     cfg_guidance=args.cfg_guidance,
                                     etc_kwargs=etc_kwargs)
        
        # Save Images
        for img in result:
            save_path = args.workdir / f'result/{str(img_count).zfill(5)}.png'
            save_image(img, save_path, normalize=True)
            img_count += 1
            
        # Optional: Save batch as Grid
        grid_dir = args.workdir / 'grid_imgs'
        os.makedirs(grid_dir, exist_ok=True)
        save_image(result, grid_dir / f'grid_{i}.png', normalize=True, nrow=int(args.b_size/2))

if __name__ == "__main__":
    main()
