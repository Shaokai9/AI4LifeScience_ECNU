import torch
import imageio
import numpy as np
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

def preprocess_your_image(file_path):
    """
    预处理图像：读取图像，确保其尺寸，然后归一化至0-1范围。
    """
    image = imageio.imread(file_path)
    assert image.shape[:2] == (64, 64), "图像尺寸必须是64x64"
    return np.array(image, dtype=np.float32) / 255.0

# 加载预训练的模型
model_path = './path_to_save_your_model/model.pth'
model = Unet3D(dim=64, dim_mults=(1, 2, 4, 8))
model.load_state_dict(torch.load(model_path))
model = model.cuda().eval()

# 初始化扩散模型
diffusion = GaussianDiffusion(
    model,
    image_size=64,
    num_frames=100,
    timesteps=1000,
    loss_type='l1'
).cuda()

# 输入一个图像帧
file_path = input("请输入要处理的图像的路径: ")
scan_image = preprocess_your_image(file_path)
scan_image_tensor = torch.tensor(scan_image).unsqueeze(0).cuda()  # 添加批处理维度并转移到GPU

# 使用模型和扩散模拟生成视频
with torch.no_grad():
    simulated_video = diffusion.sample(batch_size=1)  # generate a video

# 将输出转换为numpy数组并标准化至0-255范围
simulated_video = simulated_video.permute(0, 2, 3, 4, 1).cpu().numpy() * 255

# 将模拟视频保存为GIF
simulated_video_list = [frame.astype(np.uint8) for frame in simulated_video[0]]
imageio.mimsave('./path_to_save_your_simulated_videos/generated_video.gif', simulated_video_list, 'GIF', duration=0.1)

print("视频已成功生成并保存!")
