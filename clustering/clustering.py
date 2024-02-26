import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
from torchvision.models import resnet50
from sklearn.cluster import KMeans
import json

from sklearn.metrics.pairwise import euclidean_distances

def initialize(**kwargs):
    print("\n--- Initialize Clustering ---\n")
    
    SETTINGS = argparse.Namespace(**kwargs)
    upper_path = SETTINGS.dataset_path + "/upper_body/images/"
    lower_path = SETTINGS.dataset_path + "/lower_body/images/"
    
    # Load image numbers list
    img_list_upper = []
    for img in os.listdir(upper_path):
        if img.endswith("1.jpg"):
            img_list_upper.append(img)

    img_list_lower = []
    for img in os.listdir(lower_path):
        if img.endswith("1.jpg"):
            img_list_lower.append(img)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), # convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the pre-trained model
    model = resnet50(pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Adapt the model to use it as a feature extractor
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # Load the features
    features_upper = np.load(SETTINGS.upper_model_path)
    features_lower = np.load(SETTINGS.lower_model_path)

    # Clustering
    kmeans_upper = KMeans(n_clusters=100, random_state=22)
    kmeans_upper.fit_predict(features_upper)

    kmeans_lower = KMeans(n_clusters=100, random_state=22)
    kmeans_lower.fit_predict(features_lower)

    # Load the cluster groups dict back from the JSON file
    with open(SETTINGS.cluster_upper_result_path, 'r') as json_file:
        cluster_groups_upper = json.load(json_file)
    with open(SETTINGS.cluster_lower_result_path, 'r') as json_file:
        cluster_groups_lower = json.load(json_file)

    return {
        'device': device,
        'img_list_upper': img_list_upper,
        'img_list_lower': img_list_lower,
        'transform': transform,
        'model': model,
        'features_upper': features_upper,
        'kmeans_upper': kmeans_upper,
        'cluster_groups_upper': cluster_groups_upper,
        'features_lower': features_lower,
        'kmeans_lower': kmeans_lower,
        'cluster_groups_lower': cluster_groups_lower
    }

def convert_png_to_jpg_if_needed(img_path):
    # 파일 확장자 확인
    filename, file_extension = os.path.splitext(img_path)
    if file_extension.lower() == '.png':
        # PNG 이미지를 열기
        png_image = Image.open(img_path)
        # RGB 모드로 변환하고 투명도를 흰색 배경으로 대체
        if png_image.mode in ('RGBA', 'LA'):
            # 투명도 채널 확인 및 흰색 배경 이미지 생성
            background = Image.new('RGB', png_image.size, (255, 255, 255))
            # PNG 이미지 위에 흰색 배경 합성
            background.paste(png_image, mask=png_image.split()[3])  # 3은 알파 채널
            png_image = background
        # JPG 파일로 저장 (임시 파일 이름 생성)
        # jpg_img_path = filename + '_white_bg.jpg' # 범순
        jpg_img_path = '.\\images\\input\\ladi_vton\\upper_body\\1.jpg'
        png_image.save(jpg_img_path, 'JPEG', quality=90)
        print('convert png to jpg with white background')
        return jpg_img_path
    else:
        # 이미지 경로가 PNG가 아닐 경우 원본 경로 반환
        return img_path
    
def run_inference(is_upper, img_path, INIT_VARS=None):
    transform = INIT_VARS['transform']
    model = INIT_VARS['model']
    device = INIT_VARS['device']
    
    if is_upper: # upper body
        img_list = INIT_VARS['img_list_upper']
        features = INIT_VARS['features_upper']
        kmeans = INIT_VARS['kmeans_upper']
        cluster_groups = INIT_VARS['cluster_groups_upper']
        
    else: # lower body
        img_list = INIT_VARS['img_list_lower']
        features = INIT_VARS['features_lower']
        kmeans = INIT_VARS['kmeans_lower']
        cluster_groups = INIT_VARS['cluster_groups_lower']
    
    # test_img = plt.imread(img_path)
    jpg_img_path = convert_png_to_jpg_if_needed(img_path)
    # 변환된 JPG 이미지 읽기
    test_img = plt.imread(jpg_img_path)    

    test_img_transformed = transform(Image.fromarray(test_img))
    test_img_feature = model(test_img_transformed.unsqueeze(0).to(device)).flatten(start_dim=1).detach().cpu().numpy()

    # Predict the cluster for the test image
    test_cluster = kmeans.predict(test_img_feature) # predict the cluster number for the test image
    # test image 클러스터에 속하는 이미지들의 파일명
    test_cluster_img_list = cluster_groups[str(test_cluster[0])]                ### TODO
    test_cluster_img_idx_list = [i for i, j in enumerate(img_list) if j in test_cluster_img_list]

    ### 유클리디안 거리로 유사도 계산
    reference_feature = test_img_feature  # 기준 이미지의 특징 벡터
    cluster_features = features[test_cluster_img_idx_list]    # test 클러스터 내 모든 이미지의 특징 벡터

    # 유클리디안 거리 계산
    distances = euclidean_distances(reference_feature, cluster_features)[0]

    # 유클리디안 거리가 가까운순으로 index 정렬
    sorted_indices = np.argsort(distances)  # 오름차순으로 정렬

    # 거리가 가장 가까운 순서대로 test cluster 내 이미지 정렬한 이미지 리스트
    test_cluster_img_list = np.array(test_cluster_img_list)

    # 정렬된 인덱스를 사용하여 예측된 클러스터에 유사한 이미지 순서대로 접근

    # 정렬된 인덱스를 사용하여 유사한 이미지 순서대로 접근
    sorted_images_list = test_cluster_img_list[sorted_indices]

    return {
        'sorted_images_list': sorted_images_list
    }

# # initialize()
# a = run_inference(True, "C:/Users/jonghui/Downloads/DressCode/cloth/upper_body/000001_1.jpg", INIT_VARS=initialize())
# print(a)