import torch
from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
import imageio
import numpy as np

# Define the model architecture
model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

# Define the diffusion model
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    num_frames = 10,
    timesteps = 1000,
    loss_type = 'l1'
).cuda()

# Set up the trainer
trainer = Trainer(
    diffusion,
    'cell_sim2',
    train_batch_size = 1,
    train_lr = 1e-4,
    save_and_sample_every = 500,
    train_num_steps = 50000,
    gradient_accumulate_every = 20,
    ema_decay = 0.995,
    amp = True
)

# Start training
trainer.train()

# Save the trained model
torch.save(model.state_dict(), 'cell_sim4/cell_1.pth')

# Generate some simulated videos
for i in range(10):
    simulated_video = diffusion.sample(batch_size=1)

    # convert tensor to numpy array and normalize it to 0-255
    simulated_video = simulated_video.permute(0, 2, 3, 4, 1).detach().cpu().numpy()

    # Clip values to ensure they are in the right range
    simulated_video = np.clip(simulated_video, 0, 1)
    
    # Scale to 0-255 and convert to uint8
    simulated_video = (simulated_video * 255).astype(np.uint8)

    # save as gif
    simulated_video = simulated_video.squeeze()
    simulated_video_list = [frame for frame in simulated_video]
    imageio.mimsave(f'cell_sim4/cell_{i}.gif', simulated_video_list, 'GIF', duration=0.1)
