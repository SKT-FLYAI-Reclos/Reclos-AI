import os
from PIL import Image
import argparse
import torch

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision import models


def initialize(**kwargs):
    print("\n--- Initialize Classification ---\n")
    
    SETTINGS = argparse.Namespace(**kwargs)
    print(f"CLOTH SEGMENTATION SETTINGS: {SETTINGS}")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    
    # Initialize the pre-trained model
    model_state_dict = torch.load(SETTINGS.model_path, map_location=torch.device(device))
    
    model = models.resnet101(weights=models.ResNet101_Weights)
    # 마지막 Fully Connected Layer를 교체합니다.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 상의와 하의를 구분하므로 출력 크기는 2입니다.
    
    model.load_state_dict(model_state_dict)
    model.eval()  # Set the model to evaluation mode

    model = model.to(device)
    return {
        'device': device,
        "transforms": transform,
        'model': model,
        }


def run_inference(test_img, INIT_VARS=None):
    transform = INIT_VARS['transforms']
    model = INIT_VARS['model']
    device = INIT_VARS['device']
    
    # test_img = Image.open(img_path) # 기본값 RGB

    test_img_transformed = transform(test_img)
    input_image = test_img_transformed.unsqueeze(0).to(device) # (batch_size, channel_size, width, height)
    test_output = model(input_image)
    _, predicted = torch.max(test_output.data, 1)
    is_upper = True if predicted.item() == 0 else False
    
    return is_upper


CLASSIFICATION = {
    # 'mixed_precision' : False,
    "model_path": "..\\..\\resnet101_rgb_v1.pth",
}
CLASSIFIY_INIT_VARS = initialize(**CLASSIFICATION)
test_img = Image.open("../../../classification/cloth/upper_body/000001_1.jpg") # 기본값 RGB
a = run_inference(test_img, INIT_VARS=CLASSIFIY_INIT_VARS)
print(a)