import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os

from model import Net  # Assuming Net is defined in model.py

def main():
    # Initialize the distributed environment.
    dist.init_process_group(backend='nccl')

    # Get the current GPU id and total number of GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Data loading and preprocessing
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 8
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # Use DistributedSampler for distributed data loading
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=2, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Initialize model
    net = Net().to(device)

    # Wrap model with DistributedDataParallel
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    for epoch in range(2):  # loop over the dataset multiple times
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the model (only on the master node)
    if rank == 0:
        PATH = './cifar_net.pth'
        torch.save(net.state_dict(), PATH)

if __name__ == '__main__':
    main()
