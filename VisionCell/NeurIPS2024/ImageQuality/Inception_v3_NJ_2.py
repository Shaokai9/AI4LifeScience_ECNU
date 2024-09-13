import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import inception_v3

# 数据准备
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder(root='数据的路径', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = inception_v3(pretrained=False, aux_logits=False)  # 不使用辅助输出
model.fc = nn.Linear(2048, 299*299*3)  # 将模型的最后一层修改为输出重构的图像
model = model.to(device)

criterion = nn.MSELoss()  # 使用均方误差损失函数来评估重建质量
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, _) in enumerate(dataloader, 0):  # 我们不关心标签
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, 3, 299, 299)  # 调整输出形状以匹配输入
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader)}")

print('完成训练')

# 保存训练好的模型
torch.save(model.state_dict(), '自己训练的_inception_model.pth')
