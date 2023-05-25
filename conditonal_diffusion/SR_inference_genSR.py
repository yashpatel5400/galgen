from SR_psf_conditioned_diffusion import DDPM, ContextUnet, MyDataset
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from torchvision import models, transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import cv2

def inference(model_path, device='cuda:0', num_gen=1):
    save_dir = './output_3.5_SR/'
    tf = transforms.Compose(
    [
        #transforms.CenterCrop(212),
        transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ]
    )
    val_dataset = MyDataset(image_path="/home/aidanxue/dataset/Illustris_3.5_splited/evaluate",raw_image_path="/home/aidanxue/dataset/Illustris_128",psf_path="/home/aidanxue/dataset/Illustris_0.5_to_1.75/kernel",transform=tf)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
    ws_test = [0.0]
    # Load the model
    model = DDPM(nn_model=ContextUnet(in_channels=6, n_feat=128, n_conditions=1), betas=(1e-4, 0.02), n_T=1000,
                device=device, drop_prob=0.1)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    with torch.no_grad():
        n_sample = 1
        for w_i, w in enumerate(ws_test):
            buff_raw = True
            data_iter = iter(val_loader)
            raw_image, cond_img, psf, sig = next(data_iter)
            for i in range (0, num_gen):
                x_gen, x_gen_store = model.sample(n_sample, (3,128,128), device, x_cond=cond_img,psf=psf, guide_w=w)
                c_store = str(round(float(sig[0]), 2))
                raw_image = raw_image.to(device)
                cond_img = cond_img.to(device)
                if buff_raw:
                    save_image(raw_image, save_dir + f"raw_images.png")
                    save_image(cond_img, save_dir + f"cond_images_{c_store}.png")
                    buff_raw = False
                save_image(x_gen, save_dir + f"{i}_inference_res_{c_store}.png")

if __name__ == "__main__":
    model_path = "/home/aidanxue/galgen/conditonal_diffusion/3.5_model_499.pth"
    num_gen = 10
    inference(model_path, device='cuda:0', num_gen=num_gen)