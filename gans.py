import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
dataset = datasets.ImageFolder(
    root='/Users/yuetongliu/Documents/521 AI/casestudy/celeb_dataset',
    transform=transform
)
# 随机选择一部分数据
indices = torch.randperm(len(dataset))[:3000]  # 选择 3000 张图片
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SubsetRandomSampler(indices)
)

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),  # 添加 Flatten 层
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# 初始化生成器和判别器
netG = Generator()
netD = Discriminator()

# 定义优化器
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 定义损失函数
criterion = nn.BCELoss()

# 固定噪声，用于观察生成图像的变化
fixed_noise = torch.randn(64, 100, 1, 1)

# 初始化损失列表
d_losses = []
g_losses = []

# 训练函数
def train_gan(epochs):
    """训练 GAN 模型"""
    for epoch in range(epochs):
        d_epoch_losses = []
        g_epoch_losses = []
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)

            # 训练判别器
            optimizerD.zero_grad()

            # 真实图像
            real_labels = torch.ones(batch_size, 1)
            real_outputs = netD(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            d_loss_real.backward()

            # 生成图像
            noise = torch.randn(batch_size, 100, 1, 1)
            fake_images = netG(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_outputs = netD(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            d_loss_fake.backward()

            optimizerD.step()

            d_loss = d_loss_real + d_loss_fake
            d_epoch_losses.append(d_loss.item())

            # 训练生成器
            optimizerG.zero_grad()
            g_outputs = netD(fake_images)
            g_loss = criterion(g_outputs, real_labels)
            g_loss.backward()
            optimizerG.step()

            g_epoch_losses.append(g_loss.item())

            # 打印损失
            if (i + 1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}],'
                    f' d_loss: {d_loss.item():.4f},'
                    f' g_loss: {g_loss.item():.4f}'
                )

        # 每个 epoch 记录平均损失
        d_losses.append(torch.mean(torch.tensor(d_epoch_losses)))
        g_losses.append(torch.mean(torch.tensor(g_epoch_losses)))

        # 每个 epoch 保存生成的图像
        with torch.no_grad():
            fake_images = netG(fixed_noise)
            plt.figure(figsize=(8, 8))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.imshow(np.transpose(fake_images[i].numpy(), (1, 2, 0)) * 0.5 + 0.5)
                plt.axis('off')
            plt.savefig(f'generated_images_epoch_{epoch + 1}.png')
            plt.close()

    print("Training completed.")


# 训练 GAN
train_gan(epochs=20)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.close()

# 保存模型
torch.save(netG.state_dict(), 'generator.pth')
torch.save(netD.state_dict(), 'discriminator.pth')