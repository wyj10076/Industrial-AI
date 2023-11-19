import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score

import os
import cv2
import numpy as np


# 이미지를 불러오는 함수
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# 이미지를 전처리하는 함수
def preprocess_image(image):
    # 이미지를 0-1 범위로 스케일링
    image = image / 255.0
    # 모델이 입력으로 받는 크기로 이미지 크기 조절
    image = cv2.resize(image, (32, 32))
    # 4차원 배열로 변환 (batch_size, channel, width, height)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return image

# 모델 불러오기
model = torch.load('k_cross_CNN.pt')

# 이미지가 저장된 디렉토리 경로 지정
image_dir = './images'

# 클래스명 (라벨명) 리스트
class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'none', 'Random', 'Scratch']

# 예측 결과와 실제 라벨값 저장할 리스트 초기화
y_true = []
y_pred = []

# 모든 클래스의 이미지에 대해 예측 수행
for i, class_name in enumerate(class_names):
    print(f'Predicting {class_name} class')
    images = load_images_from_folder(os.path.join(image_dir, class_name))
    for image in tqdm(images):
        # 이미지 전처리
        image = preprocess_image(image)
        # 모델에 이미지 입력하여 예측 수행
        with torch.no_grad():
            output = model(torch.Tensor(image))
            pred = output.argmax(dim=1).item()
        # 예측 결과와 실제 라벨값 저장
        y_true.append(i)
        y_pred.append(pred)

# 모델 로드
model = torch.load('k_cross_CNN.pt')

classnames = os.walk("./images").__next__()[1]
# root[0], dirs[1], files[2]

# 예측을 수행할 입력 데이터
input_data = torch.from_numpy(data['test_data']).float()

# 입력 데이터를 모델에 적용하여 예측값 계산
with torch.no_grad():
    output = model(input_data)

# 확률값으로 예측된 레이블 계산
_, predicted = torch.max(output.data, 1)

# 예측된 레이블과 실제 레이블을 비교하여 F1-score 계산
f1 = f1_score(data['test_label'], predicted.numpy(), average='macro')

# F1-score 출력
print('F1-score:', f1)