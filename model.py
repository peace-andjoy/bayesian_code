from torch.nn import Module
from torch import nn


class lenet_nobias(Module):
    def __init__(self, inputs):
        super(lenet_nobias, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 6, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120, bias=False)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10, bias=False)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class lenet(Module):
    def __init__(self, inputs):
        super(lenet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y

class fcn(Module):
    def __init__(self, width, depth, inputs_FCN, outputs):
        super(fcn, self).__init__()
        self.width = width
        self.depth = depth
        self.inputs_FCN = inputs_FCN
        self.outputs = outputs

        layers = self.get_layers()

        self.fc = nn.Sequential(
            *layers,
        )


    def get_layers(self):
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.inputs_FCN, self.width))
        layers.append(nn.ReLU(inplace=True))

        for i in range(self.depth - 2):
            layers.append(nn.Linear(self.width, self.width))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(self.width, self.outputs))

        return layers

    def forward(self, x):
        y = self.fc(x)
        return y

class VGG16(nn.Module):
    def __init__(self, inputs, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(inputs,64,kernel_size=3,padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64,64,kernel_size=3,padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #2
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #3
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #4
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
  
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #5
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        
        self.classifier = nn.Sequential(
            # nn.Flatten(512)
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
#        print(out.shape)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out = self.classifier(out)
#        print(out.shape)
        return out

class VGG19(nn.Module):
    def __init__(self, inputs,outputs):
        super(VGG19, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(inputs,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #2
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128,128,kernel_size=3,padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #3
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256,256,kernel_size=3,padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #4
            nn.Conv2d(256,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
  
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #5
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),

            nn.Conv2d(512,512,kernel_size=3,padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        
        self.classifier = nn.Sequential(
            # nn.Flatten(512)
            nn.Linear(512,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,outputs),
            )
        #self.classifier = nn.Linear(512, 10)
 
    def forward(self, x):
        out = self.features(x) 
#        print(out.shape)
        out = out.view(out.size(0), -1)
#        print(out.shape)
        out = self.classifier(out)
#        print(out.shape)
        return out