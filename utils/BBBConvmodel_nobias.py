import torch.nn as nn
from .BBBlayers_nobias import BBBConv2d, BBBLinearFactorial, FlattenLayer


class BBBAlexNet(nn.Module):
    def __init__(self, outputs, inputs):
        # create AlexNet with probabilistic weights
        super(BBBAlexNet, self).__init__()

        # FEATURES
        self.conv1 = BBBConv2d(inputs, 64, kernel_size=11, stride=4, padding=2)
        self.conv1a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = BBBConv2d(64, 192, kernel_size=5, padding=2)
        self.conv2a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = BBBConv2d(192, 384, kernel_size=3, padding=1)
        self.conv3a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(384),
        )
        self.conv4 = BBBConv2d(384, 256, kernel_size=3, padding=1)
        self.conv4a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
        )
        self.conv5 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv5a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # CLASSIFIER
        self.flatten = FlattenLayer(256 * 6 * 6)

        self.drop1 = nn.Dropout()
        self.fc1 = BBBLinearFactorial(256 * 6 * 6, 4096)
        self.soft1 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout()
        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.soft2 = nn.ReLU(inplace=True)

        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.conv1, self.conv1a, self.conv2, self.conv2a, self.conv3, self.conv3a, self.conv4, self.conv4a,
                  self.conv5, self.conv5a, self.flatten, self.drop1, self.fc1, self.soft1, self.drop2, self.fc2, self.soft2, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
                print('conv', x.size())

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
                print('x', x.size())
        logits = x
        print('logits', logits)
        return logits, kl


class BBBLeNet(nn.Module):
    def __init__(self, outputs, inputs):
        super(BBBLeNet, self).__init__()
        self.conv1 = BBBConv2d(inputs, 6, 5, stride=1)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = BBBConv2d(6, 16, 5, stride=1)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = FlattenLayer(5 * 5 * 16)
        self.fc1 = BBBLinearFactorial(5 * 5 * 16, 120)
        self.soft3 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(120, 84)
        self.soft4 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(84, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.flatten, self.fc1, self.soft3, self.fc2, self.soft4, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl


class BBB3Conv3FC(nn.Module):
    def __init__(self, outputs, inputs):
        super(BBB3Conv3FC, self).__init__()
        self.conv1 = BBBConv2d(inputs, 32, 5, stride=1, padding=2)
        self.soft1 = nn.Softplus()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, stride=1, padding=2)
        self.soft2 = nn.Softplus()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, stride=1, padding=1)
        self.soft3 = nn.Softplus()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten = FlattenLayer(2 * 2 * 128)
        self.fc1 = BBBLinearFactorial(2 * 2 * 128, 1000)
        self.soft5 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(1000, 1000)
        self.soft6 = nn.Softplus()

        self.fc3 = BBBLinearFactorial(1000, outputs)

        layers = [self.conv1, self.soft1, self.pool1, self.conv2, self.soft2, self.pool2,
                  self.conv3, self.soft3, self.pool3, self.flatten, self.fc1, self.soft5,
                  self.fc2, self.soft6, self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        print('logits', logits)
        return logits, kl

class BBBVGG16(nn.Module):
    def __init__(self, outputs, inputs):
        # create AlexNet with probabilistic weights
        super(BBBVGG16, self).__init__()

        # FEATURES
        # block 1
        self.conv1 = BBBConv2d(inputs, 64, kernel_size=3, stride=1, padding=2)
        self.conv1a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv2 = BBBConv2d(64, 64, kernel_size=3, stride=1, padding=2)
        self.conv2a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 2

        self.conv3 = BBBConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(384),
        )
        self.conv4 = BBBConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 3
        self.conv5 = BBBConv2d(128, 256, kernel_size=3, padding=1)
        self.conv5a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv6 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv6a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        
        self.conv7 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv7a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 4
        self.conv8 = BBBConv2d(256, 512, kernel_size=3, padding=1)
        self.conv8a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv9 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv9a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv10 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv10a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 5
        self.conv11 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv11a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv12 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv12a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv13 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv13a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # CLASSIFIER
        self.flatten = FlattenLayer(512)

        # self.drop1 = nn.Dropout()
        self.fc1 = BBBLinearFactorial(512, 4096)
        self.soft1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout()

        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.soft2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()

        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.conv1, self.conv1a, self.conv2, self.conv2a, self.conv3, self.conv3a, self.conv4, self.conv4a,
                  self.conv5, self.conv5a, self.conv6, self.conv6a,self.conv7, self.conv7a,self.conv8, self.conv8a,
                  self.conv9, self.conv9a,self.conv10, self.conv10a,self.conv11, self.conv11a,self.conv12, self.conv12a,self.conv13, self.conv13a,
                  self.flatten, self.fc1, self.soft1,self.drop1, self.fc2, self.soft2,self.drop2,  self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
                print('conv', x.size())

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
                print('x', x.size())
        logits = x
        print('logits', logits)
        return logits, kl

class BBBVGG19(nn.Module):
    def __init__(self, outputs, inputs):
        # create AlexNet with probabilistic weights
        super(BBBVGG19, self).__init__()

        # FEATURES
        # block 1
        self.conv1 = BBBConv2d(inputs, 64, kernel_size=3, stride=1, padding=2)
        self.conv1a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv2 = BBBConv2d(64, 64, kernel_size=3, stride=1, padding=2)
        self.conv2a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(192),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 2

        self.conv3 = BBBConv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(384),
        )
        self.conv4 = BBBConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 3
        self.conv5 = BBBConv2d(128, 256, kernel_size=3, padding=1)
        self.conv5a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv6 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv6a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv7 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv7a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv8 = BBBConv2d(256, 256, kernel_size=3, padding=1)
        self.conv8a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 4
        self.conv9 = BBBConv2d(256, 512, kernel_size=3, padding=1)
        self.conv9a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv10 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv10a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv11 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv11a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv12 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv12a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 5
        self.conv13 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv13a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv14 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv14a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv15 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv15a = nn.Sequential(
            nn.ReLU(inplace=True)
        )
        self.conv16 = BBBConv2d(512, 512, kernel_size=3, padding=1)
        self.conv16a = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # CLASSIFIER
        self.flatten = FlattenLayer(512)

        # self.drop1 = nn.Dropout()
        self.fc1 = BBBLinearFactorial(512, 4096)
        self.soft1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout()

        self.fc2 = BBBLinearFactorial(4096, 4096)
        self.soft2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout()

        self.fc3 = BBBLinearFactorial(4096, outputs)

        layers = [self.conv1, self.conv1a, self.conv2, self.conv2a, self.conv3, self.conv3a, self.conv4, self.conv4a,
                  self.conv5, self.conv5a, self.conv6, self.conv6a,self.conv7, self.conv7a,self.conv8, self.conv8a,
                  self.conv9, self.conv9a,self.conv10, self.conv10a,self.conv11, self.conv11a,self.conv12, self.conv12a,
                  self.conv13, self.conv13a,self.conv14, self.conv14a,self.conv15, self.conv15a,self.conv16, self.conv16a,
                  self.flatten, self.fc1, self.soft1,self.drop1, self.fc2, self.soft2,self.drop2,  self.fc3]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl
                print('conv', x.size())

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
                print('x', x.size())
        logits = x
        print('logits', logits)
        return logits, kl