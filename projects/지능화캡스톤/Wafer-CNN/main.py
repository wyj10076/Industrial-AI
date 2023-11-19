# ========================================================================
# Wafer map 검사 CNN 모델의 평가 (Confusion matrix)
# 작성일 : 2023-04-03
# ========================================================================
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             recall_score, precision_score, f1_score)
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from model import ConvAutoEncoder, CNN



# confusion matrix 그리는 함수
def plot_confusion_matrix(
        y_pred, y_true, labels,             # 예측값, 실측값, labels = classnames
        title='Confusion Matrix',           # 그래프 제목
        cmap=plt.cm.Blues,                  # plt.cm.get_cmap('Blues') (old version)
        normalize=False,                    # percentage로 표현
        norm_on_model=False):               # 정규화 시 예측값 기준으로 계산

    if normalize:
        if norm_on_model:
            cm = confusion_matrix(y_true, y_pred, normalize='pred')    # ndarray, x축 (y_pred), y축 (y_true)
        else:
            cm = confusion_matrix(y_true, y_pred, normalize='true')    # ndarray, x축 (y_pred), y축 (y_true)
    else:
        cm = confusion_matrix(y_true, y_pred)
    thresh = cm.max() / 2.

    plt.imshow(cm, cmap=cmap)      # interpolation='nearest'
    plt.title(title)
    plt.colorbar()

    con_mat = confusion_matrix(y_true, y_pred)

    ticks = np.arange(len(labels))  # 눈금
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels, rotation=0, ha='right')         # va : vertical alignment

    if normalize:
        if norm_on_model:  # 모델의 예측값(column)을 기준으로 정규화
            n = con_mat.sum(0)
            for i in range(con_mat.shape[0]):
                for j in range(con_mat.shape[1]):
                    plt.text(j, i, '{:.1f}%'.format(
                        con_mat[i, j] * 100 / n[j]), ha="center", color="w" if cm[i, j] > thresh else "k")
        else:   # 데이터의 참값(row)을 기준으로 정규화
            n = con_mat.sum(1)
            for i in range(con_mat.shape[0]):
                for j in range(con_mat.shape[1]):
                    plt.text(j, i, '{:.1f}%'.format(
                        con_mat[i, j] * 100 / n[i]), ha="center", color="w" if cm[i, j] > thresh else "k")

    else:
        for i in range(con_mat.shape[0]):
            for j in range(con_mat.shape[1]):
                plt.text(j, i, con_mat[i, j], ha="center", va='center', color="w" if cm[i, j] > thresh else "k")

    # plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('Actual label', fontsize=12)
    plt.tight_layout()
    plt.show()


def calculate_cm_metrics(y_pred, y_true):
    # Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)   # 매개변수 순서 : y축, x축

    # 클래스별 TP, FN, FP, TN 값 계산
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1_score = np.zeros(n)
    for i in range(n):
        tp = tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    return accuracy_score(y_pred, y_true), precision, recall, f1_score


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        # 파일의 확장자가 이미지인지 확인
        if filename.endswith(".bmp"):
            img_path = os.path.join(folder_path, filename)
            images.append(img_path)
    return images

######################################################################

import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
plt.style.use('seaborn')
sns.set(font_scale=2)


import torch
import torchvision
from PIL import Image

##CUDA > GPU 사용여부 체크
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Running {0}".format(DEVICE))


# 데이터 로드
df = pd.read_pickle("data/LSWMD.pkl")
print(df.info())

# 초기 데이터 index별 분포도 확인
lot_index = np.unique(df.waferIndex, return_counts=True)
plt.bar(lot_index[0], lot_index[1], color='gold', align = 'center', alpha = 0.5)
plt.title("wafer Index dist")
plt.xlabel("index num")
plt.ylabel("frequency")
plt.ylim(30000, 35000)
plt.tight_layout()
plt.show()

# 불량검증에 사용하지 않는 컬럼이므로 'waferIndex' 컬럼 삭제처리
df = df.drop(['waferIndex'], axis = 1)

#add wafermapDim column because waferMap dim is different each other.
# 2차원 배열로 이루어진 waferMap을 np.size 메소드로 크기 변환
def find_dim(x):
    dim0=np.size(x,axis=0)
    dim1=np.size(x,axis=1)
    return dim0,dim1
df['waferMapDim']=df.waferMap.apply(find_dim)
print("df.sample(5) : ", df.sample(5))


#failureType 문자열을 0~8 숫자로 매핑
df['failureNum'] = df.failureType
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
df=df.replace({'failureNum':mapping_type})


#데이터 분포 정리
#df_withlabel : labeled wafer
#df_withpattern : labeled & patterned wafer
#df_nonpatter : labeled but non-patterned wafer
df_withlabel = df[(df['failureType']!=0)]
df_withlabel =df_withlabel.reset_index() #labeled index.
df_withpattern = df_withlabel[(df_withlabel['failureType'] != 'none')]
df_withpattern = df_withpattern.reset_index() #patterned index.
df_nonpattern = df_withlabel[(df_withlabel['failureType'] == 'none')] #nonpatterned index
print(df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]) # >> (172950, 25519, 147431)


fig,ax = plt.subplots(1,2, figsize = (15,5))
colors = ['blue', 'green', 'red']
num_wafers=[len(df['waferMap'])-df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0]]
labels = ['no label', 'label & pattern', 'label & non-pattern']
ax[0].pie(num_wafers, explode=(0.1,0,0), labels = labels, colors = colors, autopct = '%1.1f%%', shadow=True)
sns.countplot(x='failureNum', data=df_withpattern, ax=ax[1])
ax[1].set_title("failure type frequency")
ax[1].set_ylabel("number of patterned wafers")
plt.subplots_adjust(wspace = 1.5)
plt.tight_layout()
plt.show()

df_withlabel['waferMapDim'].value_counts() #use two dim (25,27) & (26,26)
print("df_withlabel['waferMapDim'].value_counts()", df_withlabel['waferMapDim'].value_counts())


#extract (25,27) & (26,26) waferMapDim data
def subwafer(sw,label):
  Dim0 = np.size(sw, axis=1)
  Dim1 = np.size(sw, axis=2)
  sub_df = df_withlabel.loc[df_withlabel['waferMapDim'] == (Dim0, Dim1)]
  sub_wafer = sub_df['waferMap'].values
  sw = sw.to(torch.device('cuda'))
  for i in range(len(sub_df)):
    waferMap = torch.from_numpy(sub_df.iloc[i,:]['waferMap'].reshape(1, Dim0, Dim1))
    waferMap = waferMap.to(torch.device('cuda'))
    sw = torch.cat([sw, waferMap])
    label.append(sub_df.iloc[i,:]['failureType'][0][0])
  x = sw[1:]
  y = np.array(label).reshape((-1,1))
  del waferMap, sw
  return x, y


sw0 = torch.ones((1, 25, 27))
sw1 = torch.ones((1, 26, 26))
label0 = list()
label1 = list()

x0, y0 = subwafer(sw0, label0) #about 13s
x1, y1 = subwafer(sw1, label1) #about 13s

print('x0.shape, x1.shape',x0.shape, x1.shape)
print('y0.shape, y1.shape',y0.shape, y1.shape)

#add RGB space for one-hot encoding
# 0: non wafer -> R, 1: normal die -> G, 2: defect die -> B|
IMAGE_SIZE=56
def rgb_sw(x):
  Dim0 = np.size(x, axis=1)
  Dim1 = np.size(x, axis=2)
  new_x = np.zeros((len(x), Dim0, Dim1, 3))
  x = torch.unsqueeze(x,-1)
  x = x.to(torch.device('cpu'))
  x = x.numpy()
  for w in range(len(x)):
      for i in range(Dim0):
          for j in range(Dim1):
              new_x[w, i, j, int(x[w, i, j])] = 1 #0,1,2 추출
  return new_x

rgb_x0 = rgb_sw(x0) #about 8s each line.
rgb_x1 = rgb_sw(x1)

del x0, x1 #delete useless data


#To use two dim, we have to resize these data.
def resize(x):
  rwm = torch.ones((1,IMAGE_SIZE,IMAGE_SIZE,3))
  for i in range(len(x)):
    rwm = rwm.to(torch.device('cuda'))
    a = Image.fromarray(x[i].astype('uint8')).resize((IMAGE_SIZE,IMAGE_SIZE))
    a = np.array(a).reshape((1,IMAGE_SIZE,IMAGE_SIZE,3))
    a = torch.from_numpy(a)
    a = a.to(torch.device('cuda'))
    rwm = torch.cat([rwm, a])
  x = rwm[1:]
  del rwm
  return x

#56x56픽셀로 채널이 추가된 4차원 텐서가 얻어짐
resized_x0 = resize(rgb_x0)
resized_x1 = resize(rgb_x1)


plt.imshow(rgb_x0[1000])
plt.tight_layout()
plt.show()

plt.imshow(torch.argmax(resized_x0[1000],axis=2).cpu().numpy())
plt.tight_layout()
plt.show()
# reszied_wafer is added some noise. but I think for classification pattern, some noise is not important.


del rgb_x0, rgb_x1 #delete useless data

resized_wm = torch.cat([resized_x0, resized_x1])
label_wm = np.concatenate((y0,y1)) #concatenate To use all data.

del y0,y1,resized_x0, resized_x1

#Convolutional Autoencoder
# parameter
args = {
	'BATCH_SIZE': 16,
        'LEARNING_RATE': 0.001,
        'NUM_EPOCH': 10
        }

resized_wm = resized_wm.permute(0,3,1,2)

#데이터셋을 배치사이즈로 slice처리
train_loader  = torch.utils.data.DataLoader(resized_wm, args['BATCH_SIZE'], shuffle=True)

print("resized_wm.shape, label_wm.shape",resized_wm.shape, label_wm.shape)


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from model import ConvAutoEncoder



model = ConvAutoEncoder().to(DEVICE)
print(model)
criterion = nn.MSELoss()
print("Auto Encoder criterion : " , criterion)
optimizer = torch.optim.Adam(model.parameters(), lr=args['LEARNING_RATE'])
print("Auto Encoder optimizer : " , optimizer)

# summary 생략

steps = 0
print(resized_wm.shape)
total_steps = len(train_loader)
losses =[]
iterations = []
for epoch in range(args['NUM_EPOCH']):
    running_loss = 0.0
    for i,wafer in enumerate(train_loader):
        steps += 1
        wafer = wafer.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(wafer)
        loss = criterion(outputs, wafer) #autoencoder loss : compare input & output
        loss.backward()
        running_loss += loss.item()*wafer.shape[0]
        optimizer.step()
        if steps % total_steps == 0:
            model.eval()
            print('Epoch: {}/{}'.format(epoch+1, args['NUM_EPOCH']),
                 "=> loss : %.3f"%(running_loss/total_steps))
            steps = 0
            iterations.append(i)
            losses.append(running_loss / total_steps)
            model.train()

del wafer, optimizer, loss

faulty_case = np.unique(label_wm)
print(faulty_case)


import torch.nn.init
# augment function define
def gen_data(wafer, label):
  # gen_x = torch.zeros((1, 3, 56, 56))
  gen_x = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE))
  with torch.no_grad():
    encoded_x = model.encoder(wafer).to(torch.device('cpu'))
    # dummy array for collecting noised wafer
    # Make wafer until total # of wafer to 2000
    for i in range((3000//len(wafer)) + 1):
      noised_encoded_x = (encoded_x + torch.from_numpy(np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 32, 14, 14))).to(torch.device('cpu'))).to(DEVICE)
      noised_decoded_x = model.decoder(noised_encoded_x.float()).to(torch.device('cpu'))
      gen_x = torch.cat([gen_x, noised_decoded_x], axis=0)

    # also make label vector with same length
    gen_y = np.full((len(gen_x), 1), label)
    # return date without 1st dummy data.
  del encoded_x, noised_encoded_x, noised_decoded_x
  return gen_x[1:], gen_y[1:]

# Augmentation for all faulty case.
for f in faulty_case :
    # skip none case
    if f == 'none' :
        continue
    gen_x, gen_y = gen_data(resized_wm[np.where(label_wm==f)[0]].to(DEVICE), f)
    resized_wm = torch.cat([resized_wm.to(torch.device('cpu')), gen_x], axis=0)
    label_wm = np.concatenate((label_wm, gen_y))


print('After Generate resized_wm shape : {}, label_wm shape : {}'.format(resized_wm.shape, label_wm.shape))
del gen_x, gen_y


none_idx = np.where(label_wm=='none')[0][np.random.choice(len(np.where(label_wm=='none')[0]), size=27150, replace=False)]
EdgeLoc_idx = np.where(label_wm=='Edge-Loc')[0][np.random.choice(len(np.where(label_wm=='Edge-Loc')[0]), size=1100, replace=False)]
Center_idx = np.where(label_wm=='Center')[0][np.random.choice(len(np.where(label_wm=='Center')[0]), size=2500, replace=False)]
Loc_idx = np.where(label_wm=='Loc')[0][np.random.choice(len(np.where(label_wm=='Loc')[0]), size=600, replace=False)]


delete_idx = np.concatenate((none_idx, EdgeLoc_idx, Center_idx, Loc_idx))
print(delete_idx.shape)
print(resized_wm.shape)

remove_wm = np.delete(resized_wm.detach().cpu().numpy(), delete_idx, axis=0)
resized_wm = torch.from_numpy(remove_wm)

del_idx = np.concatenate((none_idx, EdgeLoc_idx, Center_idx, Loc_idx))
label_wm = np.delete(label_wm, del_idx, axis=0)

n, bins, patches = plt.hist(label_wm, bins=9)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 최종 데이터 클래스별 개수
# print("Center : {}".format(len(np.where(label_wm=='Center')[0])))
# print("Donut : {}".format(len(np.where(label_wm=='Donut')[0])))
# print("Edge-Loc : {}".format(len(np.where(label_wm=='Edge-Loc')[0])))
# print("Edge-Ring : {}".format(len(np.where(label_wm=='none')[0])))
# print("Loc : {}".format(len(np.where(label_wm=='Loc')[0])))
# print("Near-full : {}".format(len(np.where(label_wm=='Near-full')[0])))
# print("Random : {}".format(len(np.where(label_wm=='Random')[0])))
# print("Scratch : {}".format(len(np.where(label_wm=='Scratch')[0])))
# print("none : {}".format(len(np.where(label_wm=='none')[0])))

# 증량한 데이터 이미지 파일화
# for index in range(len(resized_wm)):
#     img = resized_wm[index]
#     img = transforms.ToPILImage()(img).convert("RGB")
#     img.save("images/{}/image_{}.jpg".format(label_wm[index][0], index))

# one-hot-encoding
for i, l in enumerate(faulty_case):
    label_wm[label_wm==l] = i
    print('i : {}, l : {}'.format(i, l))
def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)

'''
원핫인코딩 전  [['2']
 ['2']
 ['2']
 ...
 ['7']
 ['7']
 ['7']]
'''

#one_hot_encoding : 학습을 위한 라벨 데이터를 서로 독립적인 이진 변수로 표현
label_wm = _to_one_hot(torch.as_tensor(np.int64(label_wm)), num_classes=9)

'''
원핫인코딩 후  
tensor([[[0, 0, 1,  ..., 0, 0, 0]],
        [[0, 0, 1,  ..., 0, 0, 0]],
        [[0, 0, 1,  ..., 0, 0, 0]],
        ...,
        [[0, 0, 0,  ..., 0, 1, 0]],
        [[0, 0, 0,  ..., 0, 1, 0]],
        [[0, 0, 0,  ..., 0, 1, 0]]])
'''

import torch.utils.data as data


class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


from sklearn import model_selection

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(resized_wm, label_wm , test_size=0.2)
dataset_train = BasicDataset(train_X,train_Y)
dataset_test = BasicDataset(test_X, test_Y)
dataset = ConcatDataset([dataset_train, dataset_test])


from model import CNN


CNN = CNN().to(DEVICE)
print(CNN)

args = {
	'BATCH_SIZE': 256,
        'LEARNING_RATE': 0.005,
        'NUM_EPOCH': 20
        }

criterion = torch.nn.CrossEntropyLoss().to(DEVICE) # 비용 함수에 소프트맥스 함수 포함되어져 있음.
print("CNN criterion : " , criterion)
optimizer = torch.optim.Adam(CNN.parameters(), lr=args['LEARNING_RATE'])
print("CNN optimizer : " , optimizer)
torch.manual_seed(42)
splits = KFold(n_splits=5, shuffle = True, random_state = 42)
foldperf={}


def train_epoch(model, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).reshape(args['BATCH_SIZE'], 9)
        optimizer.zero_grad()
        output = model(images)
        labels = torch.argmax(labels, dim=1)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        predictions = torch.argmax(output, 1)
        train_correct += (predictions == labels).sum().item()
    print(train_correct)
    return train_loss, train_correct


def valid_epoch(model, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).reshape(args['BATCH_SIZE'], 9)
        output = model(images)
        labels = torch.argmax(labels, dim=1)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        predictions = torch.argmax(output, 1)
        val_correct += (predictions == labels).sum().item()
    return valid_loss, val_correct


def eval_model(model, dataloader):
    classes = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

    model.eval()
    confusion_matrix = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    class_accuracy = {}
    for i in range(len(classes)):
        class_accuracy[classes[i]] = 100 * confusion_matrix[i, i] / confusion_matrix[i, :].sum()

    return class_accuracy




for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=args['BATCH_SIZE'], sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=args['BATCH_SIZE'], sampler=test_sampler, drop_last=True)
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(args['NUM_EPOCH']):
        train_loss, train_correct=train_epoch(CNN,train_loader,criterion,optimizer)
        test_loss, test_correct=valid_epoch(CNN,test_loader,criterion)
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                             args['NUM_EPOCH'],
                                                                                                             train_loss,
                                                                                                             test_loss,
                                                                                                             train_acc,
                                                                                                             test_acc))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

    foldperf['fold{}'.format(fold+1)] = history

test_loader = DataLoader(dataset, batch_size=args['BATCH_SIZE'], sampler=test_sampler, drop_last=True)
print(eval_model(model, test_loader))

torch.save(model,'k_cross_CNN.pt')

testl_f,tl_f,testa_f,ta_f=[],[],[],[]
k=5
for f in range(1,k+1):

     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))

     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))

print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))

print(tl_f)

# accuracy plot
plt.plot(ta_f)
plt.plot(testa_f)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('fold')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()

# loss plot
plt.plot(tl_f)
plt.plot(testl_f)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('fold')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.show()

