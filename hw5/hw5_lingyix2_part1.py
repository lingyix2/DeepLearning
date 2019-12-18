import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os

# Define data augmentation method and load data
rgb_mean = (0.5, 0.5, 0.5)
rgb_std = (0.5, 0.5, 0.5)

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std),])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])

trainset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Define Model Structure
class convBlock(nn.Module):
    def __init__(self, inChannel, outChannel, instride, inpadding):
        super(convBlock, self).__init__()
        self.convB1 = nn.Conv2d(inChannel, outChannel, kernel_size=3,stride=instride, padding=inpadding)
        self.convB2 = nn.Conv2d(outChannel, outChannel, kernel_size=3,stride=1, padding=1)
        self.bnB = nn.BatchNorm2d(outChannel)
        self.shortcut = nn.Sequential()
        if inChannel != outChannel or instride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(inChannel, outChannel,kernel_size=1, stride=instride),nn.BatchNorm2d(outChannel))

    def forward(self, x):
        out = self.bnB(self.convB1(x))
        out = F.relu(out)
        out = self.bnB(self.convB2(out))
        shortcut = self.shortcut(x)
        out = out + shortcut
        return out

class ResNetNetwork(nn.Module):
    def __init__(self):
        super(ResNetNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(p=0.2)

        self.conv21 = convBlock(32, 32, instride=1, inpadding=1)
        self.conv22 = convBlock(32, 32, instride=1, inpadding=1)
        self.conv31 = convBlock(32, 64, instride=2, inpadding=1)
        self.conv32 = convBlock(64, 64, instride=1, inpadding=1)
        self.conv33 = convBlock(64, 64, instride=1, inpadding=1)
        self.conv34 = convBlock(64, 64, instride=1, inpadding=1)
        self.conv41 = convBlock(64, 128, instride=2, inpadding=1)
        self.conv42 = convBlock(128, 128, instride=1, inpadding=1)
        self.conv43 = convBlock(128, 128, instride=1, inpadding=1)
        self.conv44 = convBlock(128, 128, instride=1, inpadding=1)
        self.conv51 = convBlock(128, 256, instride=2, inpadding=1)
        self.conv52 = convBlock(256, 256, instride=1, inpadding=1)

        self.fc = nn.Linear(256*2*2, 100)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn(x)
        x = F.relu(x)

        x = self.drop(x)

        x = self.conv21(x)
        x = self.conv22(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.conv34(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.conv44(x)

        x = self.conv51(x)
        x = self.conv52(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = x.view(-1, 256*2*2)
        x = self.fc(x)

        return x

res = ResNetNetwork()
res.cuda()
res = torch.nn.DataParallel(res, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# ------------------------------------------Define optimizer
inilr = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(res.parameters(), lr=0.001)

#-------------------------------------------Training procedure
num_epoches = 40

for epoch in range(num_epoches):
    for param_group in optimizer.param_groups:
        param_group["lr"] = (0.9**epoch)*0.001

    test_size = 0.0
    train_size = 0.0
    test_acc = 0.0
    train_count = 0.0

    for i, data in enumerate(trainloader, start=0):
        tra_inputs, tra_labels = data
        optimizer.zero_grad()
        tra_outputs = res(tra_inputs)

        tra_labels = tra_labels.to(torch.device("cuda"))
        loss = loss_function(tra_outputs, tra_labels)
        loss.backward()

        if(epoch >= 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if('step' in state and state['step'] >= 1024):
                        state['step'] = 1000
        optimizer.step()

        train_size += tra_labels.size(0)
        _, predicted = torch.max(tra_outputs.data, 1)
        train_count += (predicted == tra_labels).sum().item()

    with torch.no_grad():
        for tes_data in testloader:
            tes_image, tes_lab = tes_data
            tes_lab = tes_lab.to(torch.device("cuda"))
            tes_out = res(tes_image)
            _, tes_predicted = torch.max(tes_out.data, 1)
            test_size += tes_lab.size(0)
            test_acc += (tes_predicted == tes_lab).sum().item()

    train_accuracy = train_count/train_size
    test_accuracy = test_acc/test_size
    print('epoch number: '+str(epoch+1))
    print('test accuracy: '+str(test_accuracy))

#--------------------------------------------------Save model
torch.save(res.state_dict(), 'lingyix2ResNet1.pt')
