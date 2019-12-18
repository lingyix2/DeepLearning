import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os

# Define dara augmentation method and load data
rgb_mean = (0.5, 0.5, 0.5)
rgb_std = (0.5, 0.5, 0.5)
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(rgb_mean, rgb_std),
])

test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])

trainset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR100(root='~/scratch/Data4', train=False, download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

#load previous trained resnet model
file_dir = os.getcwd() + '/' + 'resnet18-5c106cde.pth'
def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(torch.utils.model_zoo.load_url(file_dir, model_dir='./'))
    return model

# Build and Modify the preRes
preRes = resnet18()
preRes.fc = nn.Linear(in_features=512, out_features=100, bias=True)

preRes.cuda()
preRes = torch.nn.DataParallel(preRes, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True

# Define Loss function and optimizer
inilr = 0.001
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(preRes.parameters(), lr=inilr)

# Training procedure
num_epoches = 40

# Define Upsample method
up = nn.Upsample(scale_factor=7, mode='bilinear')

for epoch in range(num_epoches):
    for param_group in optimizer.param_groups:
        param_group["lr"] = (0.9**epoch)*inilr
    train_count = 0.0
    train_size = 0.0
    test_size = 0.0
    test_acc = 0.0

    for i, data in enumerate(trainloader, start=0):

        tra_inputs, tra_labels = data
        tra_inputs = up(tra_inputs)

        optimizer.zero_grad()

        tra_outputs = preRes(tra_inputs)

        tra_labels = tra_labels.to(torch.device("cuda"))
        loss = loss_function(tra_outputs, tra_labels)
        loss.backward()

        if (epoch >= 0):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if ('step' in state and state['step'] >= 1024):
                        state['step'] = 1000

        optimizer.step()

        train_size += tra_labels.size(0)
        _, predicted = torch.max(tra_outputs.data, 1)
        train_count += (predicted == tra_labels).sum().item()

    with torch.no_grad():
        for tes_data in testloader:
            tes_image, tes_lab = tes_data
            tes_image = up(tes_image)
            tes_lab = tes_lab.to(torch.device("cuda"))
            tes_out = preRes(tes_image)
            _, tes_predicted = torch.max(tes_out.data, 1)
            test_size += tes_lab.size(0)
            test_acc += (tes_predicted == tes_lab).sum().item()

    train_accuracy = train_count/train_size
    test_accuracy = test_acc/test_size
    print('epoch number: '+str(epoch+1))
    print('test accuracy: '+str(test_accuracy))

# Save model
torch.save(preRes.state_dict(), 'lingyix2ResNet2.pt')
