import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
BS = 4
epochs = 10
tsf = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

tsf_test = transforms.Compose([
    transforms.Resize((256, 256)),    
    transforms.ToTensor(),
])
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tsf)
train_dataset = datasets.ImageFolder(root='../../../imagenet-mini/train', transform=tsf)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BS,
    num_workers=4,
    shuffle=True,
    pin_memory=(device == "cuda"),
)

classes = len(train_dataset.classes)
in_data_side_len = train_dataset[0][0].shape[1]

# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tsf_test)
test_dataset = datasets.ImageFolder(root='../../../imagenet-mini/val', transform=tsf_test)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BS,
    num_workers=4,
    shuffle=True,
    pin_memory=(device == "cuda"),
)
print(f'data shape: ({in_data_side_len}, {in_data_side_len})')
print(f'classes number: {classes}')

import torch.nn as nn
import torch.optim as optim

from VQLIC.functions import (
    AverageMeter,
)
from VQLIC.models import (
    get_model,
    get_variable_dc,
)
# MOD the switch in experiment 2
switch = 0
cb = [128, 128, 32, 32]
dim = [[16, 16, 16, 16], [8, 8, 24, 24]]
mode = [0, 2]
model = get_model('classifier')
net = model(2, classes, in_data_side_len, BS, 128, dim[switch], 4, cb).to(device)
ckpt = torch.load('variable_dims.tar')
print(ckpt.keys())
net.in_net.load_state_dict(ckpt[f'mode={mode[switch]}'])
parameters = {
    n
    for n, p in net.named_parameters()
    if n.startswith("cls")
}
params_dict = dict(net.named_parameters())
optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4,)
lrs = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", eps = 1e-8, cooldown = 10, verbose = True)
criterion = nn.CrossEntropyLoss()

import time
start = time.time()
for epoch in range(epochs):
    stamp = AverageMeter()
    for img, label in train_dataloader:
        if (not len(label) == BS):
            break
        optimizer.zero_grad()
        img = img.to(device)
        label = label.to(device)
        predict = net(img)
        print(predict.shape)
        loss = criterion(predict, label)
        loss.backward()
        optimizer.step()
        stamp.update(loss)
        print(predict.max())
        print(predict.min())
        break
    lrs.step(stamp.avg)
    strr = f'{stamp.avg}'
    print(f'{epoch}: {strr}')
    if optimizer.param_groups[0]['lr'] < 1e-7:
        break
    if epoch % 10 == 0:
        torch.save(net.state_dict(), 'cls.tar')
end = time.time()
print(f'{(end - start) / 60} min')

correct = 0
top_5_correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        if (not len(labels) == BS):
            break
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        # top 1
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        # top 5
        outputs_clone = outputs.clone()        
        for i in range(5):
            _, predicted_ = torch.max(outputs_clone.data, 1)
            top_5_correct += (predicted_ == labels).sum().item()
            outputs_clone[range(BS), predicted] = -1
        # total
        total += labels.size(0)
        break
print(f'Top-1 Accuracy: {100 * correct // total} %')
print(f'Top-5 Accuracy: {100 * top_5_correct // total} %')
print(outputs[0][:10])
print(predicted[:10])
print(labels[:10])