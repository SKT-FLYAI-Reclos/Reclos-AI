from .network import U2NET
from .options import opt

import os
from PIL import Image
import cv2
# import gdown
import argparse
import numpy as np
import glob

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict

from accelerate import Accelerator
from pathlib import Path
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("No checkpoints at given path")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("checkpoints loaded from path: {}".format(checkpoint_path))
    return model

class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"

def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)

def generate_mask(im_name, input_image, output_dir, net, device = 'cpu'): # palette, 
    img = input_image # 경로로 받아도 무방
    img_size = img.size
    img = img.resize((768, 1024), Image.BICUBIC) # Image의 resize(가로, 세로): 우리 사이즈에 맞게 수정, 원본(768, 768)  /   BICUBIC - 보간법
    image = apply_transform(img)
    image_tensor = torch.unsqueeze(image, 0) # 

    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    # Mask non-background pixels - 0이 아닌 부분 마스킹 처리
    mask = (output_arr != 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask[0], mode='L')
    mask_img = mask_img.resize(img_size, Image.BICUBIC)
    mask_img.save(os.path.join(output_dir, f'{im_name}')) # - mask 저장
    
    return mask_img   


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        print("model not found")
       
    else:
        print("Model already exists.")

def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def initialize(**kwargs):
    print("\n--- Initializing Cloth-Mask ---\n")
    SETTINGS = argparse.Namespace(**kwargs)
    print(f"CLOTH SEGMENTATION SETTINGS: {SETTINGS}")
    
    # cuda
    # Setup accelerator and device.
    accelerator = Accelerator(mixed_precision=SETTINGS.mixed_precision) # 알아서 적합한 환경 설정
    device = accelerator.device
    
    # Create an instance of your model
    model = load_seg_model(SETTINGS.checkpoint_path, device=device)
    
    return {
        "device": device,
        "model" : model,
    }

def convert_jpg_to_png(jpg_img_path):
    # JPG 파일 열기
    jpg_image = Image.open(jpg_img_path)
    
    # 파일 이름 변경 (확장자 제외)
    png_img_path = jpg_img_path.rsplit('.', 1)[0] + '.png'
    
    # PNG 파일로 저장
    jpg_image.save(png_img_path, 'PNG')

    return png_img_path

def run_inference(img_path, output_dir, INIT_VARS=None):
    device = INIT_VARS["device"]
    model = INIT_VARS["model"]
    
    mask_img_name = "clothseg.png"
    convert_jpg_to_png(img_path)
    
    img = Image.open(img_path).convert('RGB') # 이미지 RGB로 열기    
    
    generate_mask(mask_img_name, img, output_dir, net=model, device=device)
    return os.path.join(output_dir, mask_img_name)
    
