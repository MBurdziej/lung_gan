import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Parametry
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 400
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100
SAVE_DIR = "training_results_2"
os.makedirs(SAVE_DIR, exist_ok=True)

class LungCTDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.mask_files = sorted([f.replace('slice', 'lung_mask').replace('.png', '.npy') for f in self.img_files])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        img = Image.open(img_path).convert('L')

        # Load mask from .npy file and convert to PIL Image
        mask_array = np.load(mask_path)
        mask = Image.fromarray(mask_array.astype(np.uint8))  # Ensure correct type for PIL

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return mask, img

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder (6 warstw)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), # w artykule jest s=1
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc6 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # 128 ->64
        e2 = self.enc2(e1)   # 64 ->32
        e3 = self.enc3(e2)   # 32 ->16
        e4 = self.enc4(e3)   # 16 ->8
        e5 = self.enc5(e4)   # 8 ->4
        e6 = self.enc6(e5)   # 4 ->2 (bottleneck 2x2)
        
        # Decoder
        d1 = self.dec1(e6)
        d1 = torch.cat([d1, e5], 1)
        
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e4], 1)
        
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e3], 1)
        
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e2], 1)
        
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e1], 1)
        
        out = self.final(d5)
        return out

class Discriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname in ['Conv2d', 'ConvTranspose2d']:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname == 'InstanceNorm2d' and m.affine:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if __name__ == "__main__":

    img_dir = "dataset_masks"
    mask_dir = "dataset_masks"
    full_dataset = LungCTDataset(img_dir, mask_dir, transform=transform)
    
    # Podział datasetu na 95% treningowy i 5% testowy
    train_size = int(0.95 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # DataLoader tylko dla zbioru treningowego
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G = UNetGenerator().to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    def denorm(tensor):
        return tensor * 0.5 + 0.5

    for epoch in range(EPOCHS):
        loss_G_mean = 0
        loss_D_mean = 0
        for i, (mask, img) in enumerate(train_dataloader):  # Używamy train_dataloader
            mask = mask.to(device)
            img = img.to(device)
            
            # Trening Discriminatora
            optimizer_D.zero_grad()
            
            # Real images
            real_output = D(mask, img)
            real_target = torch.ones_like(real_output)
            loss_real = criterion_GAN(real_output, real_target)
            
            # Fake images
            fake_img = G(mask)
            fake_output = D(mask, fake_img.detach())
            fake_target = torch.zeros_like(fake_output)
            loss_fake = criterion_GAN(fake_output, fake_target)
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Trening Generatora
            optimizer_G.zero_grad()
            fake_output = D(mask, fake_img)
            gen_target = torch.ones_like(fake_output)
            loss_GAN = criterion_GAN(fake_output, gen_target)
            loss_L1 = criterion_L1(fake_img, img) * LAMBDA_L1
            loss_G = loss_GAN + loss_L1
            loss_G.backward()
            optimizer_G.step()
            
            loss_G_mean += loss_G.item()
            loss_D_mean += loss_D.item()
        
        loss_G_mean /= len(train_dataloader)  # Używamy train_dataloader
        loss_D_mean /= len(train_dataloader)  # Używamy train_dataloader
        print(f'Epoch [{epoch+1}/{EPOCHS}] Loss D: {loss_D_mean:.4f} Loss G: {loss_G_mean:.4f}')
        
        if (epoch+1) % 5 == 0:
            with open(os.path.join(SAVE_DIR, 'training_log.txt'), 'a') as f:
                f.write(f'Epoch {epoch+1}: Loss D = {loss_D_mean:.4f}, Loss G = {loss_G_mean:.4f}\n')
            
            # Generowanie 5 przykładów ze zbioru testowego
            fig, axes = plt.subplots(5, 3, figsize=(15, 20))  # 5 wierszy, 3 kolumny
            
            for i in range(5):
                # Losowy przykład ze zbioru testowego
                sample_idx = np.random.randint(0, len(test_dataset))
                sample_mask, sample_img = test_dataset[sample_idx]
                sample_mask = sample_mask.unsqueeze(0).to(device)
                sample_fake_img = G(sample_mask)
                
                # Konwersja do numpy i denormalizacja
                sample_mask_np = denorm(sample_mask.squeeze().cpu()).numpy()
                sample_img_np = denorm(sample_img.squeeze().cpu()).numpy()
                sample_fake_img_np = denorm(sample_fake_img.squeeze().cpu()).detach().numpy()
                
                # Wiersz i
                axes[i, 0].imshow(sample_mask_np, cmap='gray')
                axes[i, 0].set_title(f'Test Mask {i+1}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(sample_fake_img_np, cmap='gray')
                axes[i, 1].set_title(f'Generated {i+1}')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(sample_img_np, cmap='gray')
                axes[i, 2].set_title(f'Test Real {i+1}')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f'epoch_{epoch+1}.png'))
            plt.close()

            
            if (epoch+1) % 10 == 0:
                torch.save({
                    'epoch': epoch+1,
                    'generator_state_dict': G.state_dict(),
                    'discriminator_state_dict': D.state_dict(),
                    'loss_D': loss_D_mean,
                    'loss_G': loss_G_mean,
                    'sample_mask': sample_mask,
                    'sample_fake_img': sample_fake_img,
                    'sample_img': sample_img,
                }, os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

    print('Trening zakończony!')
