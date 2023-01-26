import sys
import argparse
import time

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-in","--in-subvector-num", type=int, default=2, help="setting",) 
    parser.add_argument("-e", "--epochs", default=3000, type=int, help="Number of epochs (default: %(default)s)",)
    parser.add_argument("-bs","--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("-ckpt","--checkpoint", type=str, default="", help="Load check point")
    parser.add_argument("-fd","--full-decode", action="store_true", help="full decode(T) or latent(F) classification",)
    parser.add_argument("-ft","--fine-turing", action="store_true", help="full decode(T) or latent(F) classification",)
    parser.add_argument("-rsvq","--reset-vq", action="store_true", help="full decode(T) or latent(F) classification",)
    
    
    args = parser.parse_args(argv)
    return args    

def test(args, net, test_dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            if (not len(labels) == args.batch_size):
                break
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            # top 1
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            # total
            total += labels.size(0)
    acc = 100 * correct / total
    return acc

def main(argv):
    args = parse_args(argv)
    tmp = '_fd' if args.full_decode else '_nfd'

    tsf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomRotation(degrees=20),
    #     transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tsf_test = transforms.Compose([
    #     transforms.Resize((256, 256)),    
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tsf)
    # train_dataset = datasets.ImageFolder(root='../../../imagenet-mini/train', transform=tsf)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    classes = len(train_dataset.classes)
    in_data_side_len = train_dataset[0][0].shape[1]

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tsf_test)
    # test_dataset = datasets.ImageFolder(root='../../../imagenet-mini/val', transform=tsf_test)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
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
    switch = 1
    cb = [128, 128, 32, 32]
#     dim = [16, 16, 16, 16] # mode = 0
    dim = [8, 8, 24, 24] # mode = 2
    
    if args.fine_turing:
        model = get_model('classifier_FT')
    elif args.full_decode:
        model = get_model('classifier_FD')
    else:
        model = get_model('classifier')
    net = model(args.in_subvector_num, classes, in_data_side_len, args.batch_size, 128, dim, 4, cb, args.reset_vq).to(device)
    ckpt = torch.load('variable_dims.tar')
    print(ckpt.keys())
    net.in_net.load_state_dict(ckpt['mode=2'])
    if not args.checkpoint == "":
        ckpt = torch.load(args.checkpoint)
        net.load_state_dict(ckpt)
    parameters = {
        n
        for n, p in net.named_parameters()
        if n.startswith("cls")
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=1e-4,)
    lrs = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", eps = 1e-9, cooldown = 10, verbose = True)
    criterion = nn.CrossEntropyLoss()
    
    start = time.time()
    for epoch in range(args.epochs):
        stamp = AverageMeter()
        for img, label in train_dataloader:
            if (not len(label) == args.batch_size):
                break
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            predict = net(img)
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()
            stamp.update(loss)
        lrs.step(stamp.avg)
        strr = f'{stamp.avg}'
        strrr = f'{epoch}, {strr}'        
        print(f'{strrr}')
        if epoch % 10 == 0:
            torch.save(net.state_dict(),f'cls_{args.in_subvector_num}{tmp}.tar')
            acc = test(args, net, test_dataloader)
            strrr = strrr + f', {acc}'
        with open(f'cls_{args.in_subvector_num}{tmp}.csv', 'a') as f:
            strrr = strrr + '\n'
            f.write(strrr)
        if optimizer.param_groups[0]['lr'] < 1e-8:
            break
    acc = test(args, net, test_dataloader)
    
    end = time.time()
    spent = (end - start) / 60
    print(f'{spent} min')
    
    with open(f'cls_{args.in_subvector_num}{tmp}.csv', 'a') as f:
        f.write(f'acc, {acc}\n')
        f.write(f'time, {spent}\n')
            
if __name__ == "__main__":
    main(sys.argv[1:])