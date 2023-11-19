import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        # Decoder
        self.tran_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            nn.ReLU())

        self.tran_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid())

    def encoder(self, x):
        encode = self.cnn_layer1(x)
        encode = self.cnn_layer2(encode)
        return encode

    def decoder(self, x):
        decode = self.tran_cnn_layer1(x)
        decode = self.tran_cnn_layer2(decode)
        return decode

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob1 = 0.2
        self.keep_prob2 = 0.5

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # 1번째 conv layer : 입력 층 3, 출력 32, Relu, Poolling으로 MAX 직용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # 2번째 conv layer : 입력 층 32, 출력 64, Relu, Poolling으로 MAX 직용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 3번째 conv layer : 입력 층 64, 출력 128, Relu, Polling으로 Max 적용.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        self.fc1 = nn.Linear(8 * 8 * 128, 1250, bias=True)  # fully connected,
        nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = nn.Sequential(
            self.fc1,
            nn.ReLU())  # dropout 적용

        self.fc2 = nn.Linear(1250, 9, bias=True)  # 오류패턴 9개로 출력 9
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)  # fully conntect를위해 flatten을 함.
        out = self.layer4(out)
        out = self.fc2(out)
        return out


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3),
#             nn.ReLU(),
#         )
#
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.conv5 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.conv6 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.conv7 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)
#
#         self.conv8 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#         )
#
#         self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=1)
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(128 * 3 * 3, 4608),
#             nn.ReLU(),
#         )
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(4608, 512),
#             nn.ReLU(),
#         )
#
#         self.output = nn.Sequential(
#             nn.Linear(512, 9),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.maxpool2(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.maxpool3(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.maxpool4(x)
#         x = self.conv8(x)
#         x = self.maxpool5(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.output(x)
#         return x