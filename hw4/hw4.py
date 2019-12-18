import numpy as np

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as datasets
import torchvision.transforms as transforms

num_epochs = 30
learning_rate = 0.001
batch_size = 120
DIM = 32
no_of_hidden_units = 196
LAMADA = 10
n_z = 100
n_classes = 10
gen_train = 1
monte_carlo_size = 50

class cnn_network(nn.Module):
    def __init__(self):
        super(cnn_network,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=(1,1), padding=2)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=0)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.4)
        self.drop4 = nn.Dropout(p=0.5)
    
        self.fc1 = nn.Linear(4*4*64, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self,x,extract_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.drop3(x)
        x = F.relu(self.bn4(self.conv7(x)))
        x = F.relu(self.bn5(self.conv8(x)))
        x = self.drop4(x)
        x = x.view(-1, 4*4*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM,scale=(0.7,1.0),ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness = 0.1*torch.randn(1),
            contrast = 0.1*torch.randn(1),
            saturation = 0.1*torch.randn(1),
            hue = 0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size= batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model = cnn_network()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)

for epoch in range(0,num_epochs):

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate/(1+epoch*0.1)

    train_accu = []
    model.train()

    for batch_idx, (X_train_batch,Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0]<batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y_train_batch.data).sum())/float(batch_size))*100.0

        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch,accuracy_epoch)


    # Heuristic Prediction Rule
    model.eval()
    # Normal Prediction
    with torch.no_grad():
        test_accu = []
        for batch_idx, (x_test_batch,y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch = Variable(x_test_batch).cuda(),Variable(y_test_batch).cuda()
            output = model(X_test_batch)
            prediction = output.data.max(1)[1]
            accuracy = (float(prediction.eq(Y_test_batch.data).sum())/float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print("Testing",accuracy_test)

#------------when using monte carlo prediction, use this part
#    # Monte Carlo Prediction
#    with torch.no_grad():
#        test_accu = []
#        for batch_idx, (X_test_batch,Y_test_batch) in enumerate(testloader):
#            X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(),Variable(Y_test_batch).cuda()
#            prediction_vector = Variable(torch.tensor(np.zeros((50,120,10)))).cuda()
#            for i in range(monte_carlo_size):
#                output = model(X_test_batch)
#                temp_prediction = output.data
#                prediction_vector[i] = temp_prediction
#            prediction = (prediction_vector.mean(0)).max(1)[1]
#            accuracy = (float(prediction.eq(Y_test_batch.data).sum())/float(batch_size))*100.0
#            test_accu.append(accuracy)
#        mc_test_acc = np.mean(test_accu)
#    print("Monte Carlo test accuracy:",mc_test_acc)


torch.save(model,"CIFAR10_Xu.model")
