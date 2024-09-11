import torch
import numpy as np
import imageio
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# 1. 加载模型
model_path = './path_to_save_your_model/mfullcell.pth'

model = Unet3D(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)
model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()  # 设置模型为评估模式

# 2. 定义扩散模型
diffusion = GaussianDiffusion(
    model,
    image_size=64,
    num_frames=100,
    timesteps=1000,
    loss_type='l1'
).cuda()

# 3. 生成模拟视频
for i in range(100):
    simulated_video = diffusion.sample(batch_size=1)

    # 将张量数据转换为numpy数组，进行规范化，并将其缩放到0-255范围
    simulated_video = simulated_video.permute(0, 2, 3, 4, 1).detach().cpu().numpy()
    simulated_video = np.clip(simulated_video, 0, 1)
    simulated_video = (simulated_video * 255).astype(np.uint8)

    # 保存为GIF格式
    simulated_video = simulated_video.squeeze()
    simulated_video_list = [frame for frame in simulated_video]
    imageio.mimsave(f'./path_to_save_your_simulated_videos/video_{i}.gif', simulated_video_list, 'GIF', duration=0.1)

print("模拟视频生成完成！")
