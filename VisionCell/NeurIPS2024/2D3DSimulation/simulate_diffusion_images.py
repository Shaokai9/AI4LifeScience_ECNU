import os
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from denoising_diffusion_pytorch.utils import Trainer

def simulate_images(
    image_path,
    output_dir,
    train_batch_size=32,
    train_lr=8e-5,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True
):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,  # number of steps
        sampling_timesteps=250,
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l1'  # L1 or L2
    ).cuda()

    trainer = Trainer(
        diffusion,
        image_path,
        train_batch_size=train_batch_size,
        train_lr=train_lr,
        train_num_steps=train_num_steps,
        gradient_accumulate_every=gradient_accumulate_every,
        ema_decay=ema_decay,
        amp=amp,
        output_dir=output_dir
    )

    trainer.train()

if __name__ == "__main__":
    image_path = '/Users/hassanyang/VirtualEnv/PyTorch/CellData/SimDateSet/20230314png/1-50'
    output_dir = '/Users/hassanyang/VirtualEnv/ChatGPT_PyTorch'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulate_images(image_path, output_dir)
