from network import U2NET

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
from options import opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
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

def generate_mask(im_name, input_image, net, device = 'cpu'): # palette, 

    img = input_image
    img_size = img.size
    img = img.resize((768, 1024), Image.BICUBIC) # Image의 resize(가로, 세로): 우리 사이즈에 맞게 수정, 원본(768, 768)  /   BICUBIC - 보간법
    image = apply_transform(img)
    image_tensor = torch.unsqueeze(image, 0) # 

    os.makedirs(args.output, exist_ok=True)
    
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
    mask_img.save(os.path.join(args.output, f'{im_name}.jpg')) # - mask 저장
    
    return mask_img

def change_background(original_img, mask_img): ## 여기서 부터
    # 원본 이미지를 넘파이 배열로 변환
    original_np = np.array(original_img)
    mask_np = np.array(mask_img)
    
    # Mask non-background pixels
    mask_np = mask_np[:, :, 0]  # Convert to single channel
    mask_np = mask_np // 255  # Convert to binary mask

    # Make masked pixels white
    original_np[mask_np != 1] = [255, 255, 255]
    
    result_img = Image.fromarray(original_np)
    
    return result_img   


def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        print("please model download!")
       
    else:
        print("Model already exists.")

def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(args):
    device = 'cuda:0' if args.cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(args.checkpoint_path, device=device)

    im_name, _ = os.path.splitext(os.path.basename(args.image)) # 파일명 생성
    img = Image.open(args.image).convert('RGB') # 이미지 RGB로 열기
    mask_img = generate_mask(im_name, img, net=model, device=device)
    print("결과가 저장되었습니다")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='./model/cloth_segm.pth', help='Path to the checkpoint file')
    parser.add_argument("--output", type=str, default='../images/output/cloth-mask', help='Path to the output file')
    args = parser.parse_args()

    main(args)