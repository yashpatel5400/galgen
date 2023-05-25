'''
This code is modified from,
https://github.com/TeaPearce/Conditional_Diffusion_MNIST
'''
import random

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import cv2

class MyDataset(Dataset):
    def __init__(self, image_path, raw_image_path, psf_path, transform=None):
        self.image_path = image_path
        self.raw_image_path = raw_image_path
        self.psf_path = psf_path
        self.transform = transform
        self.images = []
        self.labels = []
        self.raw_images = []
        self.psfs = {}

        for filename in os.listdir(self.image_path):
            if filename.endswith(".png"):
                # name as "rotated_{}_{id}_{mass}_{position}.png"
                self.images.append(os.path.join(self.image_path, filename))
                label = filename.split("_")[0]
                index = filename.split("_")[1]
                self.raw_images.append(index)
                self.labels.append(label)
                if label not in self.psfs:
                    npy_file = "{}_gaussian_kernel_2d.npy".format(label)
                    npy_path = os.path.join(self.psf_path,npy_file)
                    psf = np.load(npy_path)
                    self.psfs[label] = psf


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        raw_img_path = os.path.join(self.raw_image_path, self.raw_images[idx])
        raw_img = Image.open(raw_img_path).convert('RGB')
        label = self.labels[idx]
        psf = self.psfs[label]

        if self.transform:
            img = self.transform(img)
            raw_img = self.transform(raw_img)

        return raw_img, img, torch.from_numpy(psf).float(), label

class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class EmbedConv(nn.Module):
    def __init__(self, emb_dim):
        super(EmbedConv, self).__init__()
        self.conv1 = nn.Conv2d(1,16,kernel_size=3,padding=1)
        self.fc1 = nn.Linear(16 * 5 * 5, emb_dim)
        self.fc2 = nn.Linear(emb_dim,emb_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=256, n_conditions=1):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_conditions = n_conditions

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)

        self.to_hidden = nn.Sequential(nn.Conv2d(4 * n_feat, 8 * n_feat, 3, 1, 1), nn.GELU())

        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.timeembed3 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedConv(4 * n_feat)
        self.contextembed2 = EmbedConv(2 * n_feat)
        self.contextembed3 = EmbedConv(1 * n_feat)

        self.up0 = nn.Sequential(
            nn.Conv2d(8 * n_feat, 4 * n_feat, 3, 1, 1),
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(8 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, 3, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep,
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_hidden(down3)

        # convert context to one hot embedding
        #c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        #c = c.unsqueeze(1).type(torch.float32)
        # mask out context if context_mask == 1
        #context_mask = context_mask[:, None]
        #context_mask = context_mask.repeat(1, self.n_conditions)
        context_mask = (-1 * (1 - context_mask)).type(torch.float32)  # need to flip 0 <-> 1
        c = c * context_mask
        c = c.unsqueeze(1)

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 4, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.n_feat, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down3)  # add and multiply embeddings
        up3 = self.up2(cemb2 * up2 + temb2, down2)
        up4 = self.up3(cemb3 * up3 + temb3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, x_cond, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + self.drop_prob).view(-1,1,1).to(self.device)

        # return MSE between added noise, and our predicted noise
        # TODO: 
        # print("1x_i", x.shape)
        # print("1c_i", c.shape)
        # print("1x_cond", x_cond.shape)
        # print("mask", context_mask.shape)
        return self.loss_mse(noise, self.nn_model(torch.cat([x_cond, x_t], dim=1), c, _ts / self.n_T, context_mask))

    def random_psf(self):
        variance = random.uniform(0.5,4)
        psf_1d = cv2.getGaussianKernel(10,variance)
        return np.outer(psf_1d,psf_1d), variance


    def sample(self, n_sample, size, device, x_cond, psf, guide_w=0.0):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        x_cond = x_cond.repeat(2*n_sample,1,1,1).to(device)
        c_i = psf.to(device)

        # double the batch
        c_i = c_i.repeat(2*n_sample,1,1)
        # don't drop context at test time
        context_mask = torch.zeros((2*n_sample)).view(2*n_sample,1,1).to(device)
        context_mask[n_sample:] = 1.  # makes second half of batch context free

        x_i_store = []  # keep track of generated steps in case want to plot something
        print()
        for i in range(1000, 0, -1):
            print(f'sampling timestep {i}', end='\r')
            t_is = torch.tensor([i / 1000]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(torch.cat([x_cond, x_i], dim=1), c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                    self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train_mnist():
    # hardcoding these here
    n_epoch = 500
    batch_size = 96
    n_T = 1000 # 500  # 1000
    device = "cuda:0"
    n_conditions = 1
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = './output/'
    ws_test = [0.0]  # strength of generative guidance


    ddpm = DDPM(nn_model=ContextUnet(in_channels=6, n_feat=n_feat, n_conditions=n_conditions), betas=(1e-4, 0.02), n_T=n_T,
                device=device, drop_prob=0.1)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/ddpm_unet01_mnist_9.pth"))

    tf = transforms.Compose(
        [
            #transforms.CenterCrop(212),
            transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    train_dataset = MyDataset(image_path="/home/aidanxue/SR/Illustris_0.5_to_1.75/train",raw_image_path="/home/aidanxue/SR/Illustris_128",psf_path="/home/aidanxue/SR/kernel",transform=tf)
    val_dataset = MyDataset(image_path="/home/aidanxue/SR/Illustris_0.5_to_1.75/evaluate",raw_image_path="/home/aidanxue/SR/Illustris_128",psf_path="/home/aidanxue/SR/kernel",transform=tf)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)
        
        pbar = tqdm(train_loader)
        loss_ema = None
        # TODO:
        for x, x_c, c, lable in pbar:
            optim.zero_grad()
            x = x.to(device)
            x_c = x_c.to(device)
            c = c.to(device)
            loss = ddpm(x, x_c, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        if (ep+1) % 10 == 0:
            ddpm.eval()
            with torch.no_grad():
                n_sample = 2*n_conditions
                for w_i, w in enumerate(ws_test):
                    data_iter = iter(val_loader)
                    raw_image, cond_img, psf, sig = next(data_iter)
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3,128,128), device, x_cond=cond_img,psf=psf, guide_w=w)
                    # TODO
                    grid = make_grid(x_gen, nrow=10)
                    c_store = str(round(float(sig[0]), 2))
                    save_image(grid, save_dir + f"SR_image_ep{ep}_w{w}_{c_store}.png".format())
                    save_image(cond_img, save_dir + f"cond_img_ep{ep}_w{w}_{c_store}.png")
                    save_image(raw_image, save_dir + f"raw_img_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}_{c_store}.png")

        # optionally save model
        if save_model and (ep+1) % 50 == 0:
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_mnist()
