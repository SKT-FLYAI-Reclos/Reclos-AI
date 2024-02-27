# %%
### Grey Scale로 clustering하기 - Upper_Body

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %% 
dataset_path = "C:/reClos/dataset/DressCode/cloth/upper_body/" # 데이터 경로

img_list = []
for img in os.listdir(dataset_path):
    if img.endswith("1.jpg"): # 1.jpg로 끝나면 추가
        img_list.append(img)

print(img_list)

# %%
print(len(img_list))

# %%
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))] # 존재하는 파일이면 해당 파일명들 반환

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx]) # 전체 파일 경로
        # image = Image.open(img_name).convert('RGB')
        image = Image.open(img_name).convert('L') # load image grey scale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageDataset(directory=dataset_path, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# %%
import torch
from torchvision.models import resnet50

# Initialize the pre-trained model
model = resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Adapt the model to use it as a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device: cuda')
else:
    device = torch.device("cpu")
    print('device: cpu')
model = model.to(device)

# Feature extraction function
def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move input data to the GPU
            outputs = model(inputs).flatten(start_dim=1)
            features.append(outputs.cpu().numpy())  # Move the tensors back to CPU for numpy conversion
    return np.concatenate(features, axis=0)

# Extract features
features_u = extract_features(dataloader, model, device)

# %%
features_u.shape

# %%
#save model
np.save('./features_grey_upper.npy', features_u)

# %%
# load model
features_u = np.load('./features_grey_upper.npy')

# %%
from sklearn.cluster import KMeans

# Clustering
kmeans_u = KMeans(n_clusters=100, random_state=22) # 100개의 cluster로 나누기
clusters_u = kmeans_u.fit_predict(features_u) # 각 이미지가 어떤 cluster에 속하는지

# %%
print(len(clusters_u))
print(clusters_u)
print(len(set(clusters_u)))

# %%
cluster_groups_u = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i, cluster in enumerate(clusters_u):
    # print(i, cluster)
    if cluster not in cluster_groups_u:
        cluster_groups_u[cluster] = []
    
    cluster_groups_u[cluster].append(img_list[i]) # 파일명 저장

# %%
def print_cluster_img(num):
    for i in range(15):
        img = plt.imread(dataset_path + cluster_groups_u[num][i])
        plt.subplot(3, 5, i+1)
        plt.axis('off')
        plt.imshow(img)

# %%
for i in range(30):
    plt.figure()
    plt.axis('off')
    plt.title(f'image cluster {i}')
    print_cluster_img(i)

# %%
cluster_groups_u[5]

# %% [markdown]
# ### Predict

# %%
test_img = plt.imread('C:/reClos/dataset/DressCode/cloth/upper_body/007204_1.jpg')

test_img_transformed = transform(Image.fromarray(test_img))
test_img_feature = model(test_img_transformed.unsqueeze(0).to(device)).flatten(start_dim=1).detach().cpu().numpy()
# Plot the test image
plt.imshow(test_img)
plt.axis('off')
plt.show()

# Predict the cluster for the test image
test_cluster = kmeans_u.predict(test_img_feature) # predict the cluster number for the test image
print("Test Image Cluster:", test_cluster)

# Plot the images in the same cluster as the test image
print_cluster_img(test_cluster[0]) # [0] 리스트 벗겨주기


# %%
cluster_groups_u[test_cluster[0]] # 특정 그룹 선정

# %%
# 예측 클러스터에 속하는 이미지들의 파일명을 저장하는 리스트 생성
test_cluster_img_list = cluster_groups_u[test_cluster[0]]

print("test image 클러스터에 속하는 이미지들의 파일명:", test_cluster_img_list)

test_cluster_img_idx_list = [i for i, j in enumerate(img_list) if j in test_cluster_img_list]
test_cluster_img_idx_list # idx list

# %% [markdown]
# ### Cosine Similarity로 유사도 계산

# %%
from sklearn.metrics.pairwise import cosine_similarity

# 기준 이미지의 특징 벡터와 test 클러스터 내 이미지들의 특징 벡터
reference_feature = test_img_feature  # 기준 이미지의 특징 벡터


cluster_features = features_u[test_cluster_img_idx_list]    # test 클러스터 내 모든 이미지의 특징 벡터
cluster_features.shape

# 코사인 유사도 계산
similarities = cosine_similarity(reference_feature, cluster_features)[0]

# %%
similarities.shape

# %%
type(features_u)

# %%
# 유사도에 따라 인덱스 정렬
sorted_indices = np.argsort(similarities)[::-1][0]  # 내림차순으로 정렬
print(sorted_indices.shape)
sorted_indices

# %%
type(sorted_indices)

# %%
test_cluster_img_list = np.array(test_cluster_img_list)
test_cluster_img_list[sorted_indices]  # cosine 유사도가 높은 순서대로 test cluster 내 이미지 정렬한 이미지 리스트

# %%
# 정렬된 인덱스를 사용하여 유사한 이미지 순서대로 접근
sorted_images_list = test_cluster_img_list[sorted_indices]

# 결과 사용, 예를 들어 가장 유사한 이미지부터 출력
for i in range(15):
    img = plt.imread(dataset_path + sorted_images_list[i])
    plt.subplot(3, 5, i+1)
    plt.axis('off')
    plt.imshow(img)

# %% [markdown]
# ### 유클리디안 거리로 유사도 계산

# %%
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

reference_feature = test_img_feature  # 기준 이미지의 특징 벡터
cluster_features_u = features_u[test_cluster_img_idx_list]    # test 클러스터 내 모든 이미지의 특징 벡터

# 유클리디안 거리 계산
distances = euclidean_distances(reference_feature, cluster_features_u)[0]
distances.shape

# %%
# 거리에 따라 index 정렬
sorted_indices = np.argsort(distances)  # 오름차순으로 정렬
print(sorted_indices.shape)
sorted_indices

# %%
test_cluster_img_list = np.array(test_cluster_img_list)
test_cluster_img_list[sorted_indices]  # 거리가 가장 가까운 순서대로 test cluster 내 이미지 정렬한 이미지 리스트

# %%
# 정렬된 인덱스를 사용하여 유사한 이미지 순서대로 접근
sorted_images_list = test_cluster_img_list[sorted_indices]

dataset_path = "C:/reClos/dataset/DressCode/cloth/upper_body/"
# 결과 사용, 예를 들어 가장 유사한 이미지부터 출력
for i in range(15):
    img = plt.imread(dataset_path + sorted_images_list[i])
    plt.subplot(3, 5, i+1)
    plt.axis('off')
    plt.imshow(img)

# %%


# %%


# %%


# %%
############################################################################################################

# %%
### Grey Scale로 clustering하기 - Lower_Body

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
dataset_path = "C:/reClos/dataset/DressCode/cloth/lower_body/"

img_list = []
for img in os.listdir(dataset_path):
    if img.endswith("1.jpg"):
        img_list.append(img)

print(img_list)

# %%
print(len(img_list))

# %%
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        # image = Image.open(img_name).convert('RGB')
        image = Image.open(img_name).convert('L') # load image grey scale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageDataset(directory=dataset_path, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# %%
import torch
from torchvision.models import resnet50

# Initialize the pre-trained model
model = resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Adapt the model to use it as a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device: cuda')
else:
    device = torch.device("cpu")
    print('device: cpu')
model = model.to(device)

# Feature extraction function
def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move input data to the GPU
            outputs = model(inputs).flatten(start_dim=1)
            features.append(outputs.cpu().numpy())  # Move the tensors back to CPU for numpy conversion
    return np.concatenate(features, axis=0)

# Extract features
features_l = extract_features(dataloader, model, device)

# %%
features_l.shape

# %%
#save model
np.save('./features_grey_lower.npy', features_l)

# %%
import numpy as np
# load model
features_l = np.load('./features_grey_lower.npy')

# %%
from sklearn.cluster import KMeans

# Clustering
kmeans_l = KMeans(n_clusters=100, random_state=22)
clusters_l = kmeans_l.fit_predict(features_l)

# %%
print(len(clusters_l))
print(clusters_l)
print(len(set(clusters_l)))

# %%
cluster_groups_l = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i, cluster in enumerate(clusters_l):
    # print(i, cluster)
    if cluster not in cluster_groups_l:
        cluster_groups_l[cluster] = []
    
    cluster_groups_l[cluster].append(img_list[i])

# %%
def print_cluster_img(num):
    for i in range(15):
        img = plt.imread(dataset_path + cluster_groups_l[num][i])
        plt.subplot(3, 5, i+1)
        plt.axis('off')
        plt.imshow(img)

# %%
for i in range(30):
    plt.figure()
    plt.axis('off')
    plt.title(f'image cluster {i}')
    print_cluster_img(i)

# %%


# %% [markdown]
# ### Predict

# %%
test_img = plt.imread('C:/reClos/dataset/DressCode/cloth/lower_body/013582_1.jpg')

test_img_transformed = transform(Image.fromarray(test_img))
test_img_feature = model(test_img_transformed.unsqueeze(0).to(device)).flatten(start_dim=1).detach().cpu().numpy()
# Plot the test image
plt.imshow(test_img)
plt.axis('off')
plt.show()

# Predict the cluster for the test image
test_cluster = kmeans_l.predict(test_img_feature) # predict the cluster number for the test image
print("Test Image Cluster:", test_cluster)

# Plot the images in the same cluster as the test image
print_cluster_img(test_cluster[0])


# %%
cluster_groups_l[test_cluster[0]]

# %%
# 예측 클러스터에 속하는 이미지들의 파일명을 저장하는 리스트 생성
test_cluster_img_list = cluster_groups_l[test_cluster[0]]

print("test image 클러스터에 속하는 이미지들의 파일명:", test_cluster_img_list)

test_cluster_img_idx_list = [i for i, j in enumerate(img_list) if j in test_cluster_img_list]
test_cluster_img_idx_list # idx list

# %%
### 유클리드 거리로 정렬

# %%
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

reference_feature = test_img_feature  # 기준 이미지의 특징 벡터
cluster_features = features_l[test_cluster_img_idx_list]    # test 클러스터 내 모든 이미지의 특징 벡터

# 유클리디안 거리 계산
distances = euclidean_distances(reference_feature, cluster_features)[0]
distances.shape

# %%
# 거리에 따라 index 정렬
sorted_indices = np.argsort(distances)  # 오름차순으로 정렬
print(sorted_indices.shape)
sorted_indices

# %%
test_cluster_img_list = np.array(test_cluster_img_list)
test_cluster_img_list[sorted_indices]  # 거리가 가장 가까운 순서대로 test cluster 내 이미지 정렬한 이미지 리스트

# %%
# 정렬된 인덱스를 사용하여 유사한 이미지 순서대로 접근
sorted_images_list = test_cluster_img_list[sorted_indices]

# 결과 사용, 예를 들어 가장 유사한 이미지부터 출력
for i in range(15):
    img = plt.imread(dataset_path + sorted_images_list[i])
    plt.subplot(3, 5, i+1)
    plt.axis('off')
    plt.imshow(img)

# %%


# %%


# %%


# %%
### Grey Scale로 clustering하기 - Lower_Body with Model image

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
dataset_path = "C:/reClos/dataset/DressCode/cloth/lower_body"

img_list = []
for img in os.listdir(dataset_path):
    if img.endswith("1.jpg"):
        img_list.append(img)

print(img_list)

# %%
print(len(img_list))

# %%
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        # image = Image.open(img_name).convert('RGB')
        image = Image.open(img_name).convert('L') # load image grey scale
        if self.transform:
            image = self.transform(image)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageDataset(directory=dataset_path, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# %%
import torch
from torchvision.models import resnet50

# Initialize the pre-trained model
model = resnet50(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Adapt the model to use it as a feature extractor
model = torch.nn.Sequential(*(list(model.children())[:-1]))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('device: cuda')
else:
    device = torch.device("cpu")
    print('device: cpu')
model = model.to(device)

# Feature extraction function
def extract_features(dataloader, model, device):
    features = []
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)  # Move input data to the GPU
            outputs = model(inputs).flatten(start_dim=1)
            features.append(outputs.cpu().numpy())  # Move the tensors back to CPU for numpy conversion
    return np.concatenate(features, axis=0)

# Extract features
features_l = extract_features(dataloader, model, device)

# %%
features_l.shape

# %%
#save model
np.save('./features_grey_lower.npy', features_l)

# %%
import numpy as np
# load model
features_l = np.load('./features_grey_lower.npy')

# %%
from sklearn.cluster import KMeans

# Clustering
kmeans_l = KMeans(n_clusters=150, random_state=22)
clusters_l = kmeans_l.fit_predict(features_l)

# %%
print(len(clusters_l))
print(clusters_l)
print(len(set(clusters_l)))

# %%
cluster_groups = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i, cluster in enumerate(clusters_l):
    # print(i, cluster)
    if cluster not in cluster_groups:
        cluster_groups[cluster] = []
    
    cluster_groups[cluster].append(img_list[i])

# %%
def print_cluster_img(num):
    for i in range(15):
        img = plt.imread(dataset_path + cluster_groups[num][i])
        plt.subplot(3, 5, i+1)
        plt.axis('off')
        plt.imshow(img)

# %%
for i in range(30):
    plt.figure()
    plt.axis('off')
    plt.title(f'image cluster {i}')
    print_cluster_img(i)

# %%
for i in range(30,60):
    plt.figure()
    plt.axis('off')
    plt.title(f'image cluster {i}')
    print_cluster_img(i)

# %%


# %%
### 다른 코드

# %%
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# 사전 학습된 ResNet 모델을 사용하여 특징 추출
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # ResNet 모델 불러오기
        self.model = models.resnet18(pretrained=True)
        # 모델의 마지막 층 제거
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

# 옷 사진 데이터셋
class ClothesDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# 데이터셋 로드 및 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset_path = "C:/reClos/dataset/DressCode/cloth/lower_body/"

img_paths = [] # 이미지 경로 리스트
for img in os.listdir(dataset_path):
    if img.endswith("1.jpg"):
        img_paths.append(dataset_path + img)

print(img_paths)

# image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', '...']  
dataset = ClothesDataset(img_paths, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# 특징 추출
feature_extractor = FeatureExtractor().eval()  # 평가 모드로 설정
features_l2 = []
for images in dataloader:
    with torch.no_grad():
        output = feature_extractor(images)
    features_l2.append(output.cpu().numpy())

features_l2 = np.vstack(features_l2)

# K-means 클러스터링
n_clusters = 40  # 클러스터의 수
kmeans_l2 = KMeans(n_clusters=n_clusters, random_state=0).fit(features_l2)
labels = kmeans_l2.labels_

# Extract features
features_l2 = extract_features(dataloader, model, device)

# Clustering
kmeans_l2 = KMeans(n_clusters=150, random_state=22)
clusters_l2 = kmeans_l2.fit_predict(features_l2)


# %%
#save model
np.save('./features_grey_lower2.npy', features_l2)

# %%
# load model
features_l2 = np.load('./features_grey_lower2.npy')

# %%
cluster_groups = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
for i, cluster in enumerate(clusters_l2):
    # print(i, cluster)
    if cluster not in cluster_groups:
        cluster_groups[cluster] = []
    
    cluster_groups[cluster].append(img_list[i])

# %%
def print_cluster_img(num):
    for i in range(15):
        img = plt.imread(dataset_path + cluster_groups[num][i])
        plt.subplot(3, 5, i+1)
        plt.axis('off')
        plt.imshow(img)

# %%
for i in range(30):
    plt.figure()
    plt.axis('off')
    plt.title(f'image cluster {i}')
    print_cluster_img(i)

# %%



