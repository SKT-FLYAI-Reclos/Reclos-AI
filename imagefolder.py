import glob
import shutil
import os

# 이미지가 있는 폴더 경로
source_folder = 'C:/Users/jonghui/Downloads/DressCode/lower_body/images'
# 복사할 대상 폴더 경로
target_folder = 'C:/Users/jonghui/Downloads/DressCode/cloth/lower_body'

# target_folder가 없으면 생성
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# '_1.jpg'로 끝나는 모든 파일 찾기
for file_path in glob.glob(f'{source_folder}/*_1.jpg'):
    # 복사할 대상 파일의 경로 생성
    target_path = os.path.join(target_folder, os.path.basename(file_path))
    # 파일 복사
    shutil.copy(file_path, target_path)