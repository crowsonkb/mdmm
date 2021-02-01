#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

import mdmm

BATCH_SIZE = 50
EPOCHS = 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    tf = transforms.ToTensor()
    train_set = datasets.MNIST('data/mnist', download=True, transform=tf)
    train_dl = data.DataLoader(train_set, BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_set = datasets.MNIST('data/mnist', train=False, download=True, transform=tf)
    val_dl = data.DataLoader(train_set, BATCH_SIZE, num_workers=2, pin_memory=True)

    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 10),
    ).to(device)

    crit = nn.CrossEntropyLoss()

    def make_constraint(layer):
        return mdmm.EqConstraint(lambda: layer.weight.abs().mean(), 1, damping=20)

    constraints = []
    for layer in model:
        if hasattr(layer, 'weight'):
            constraints.append(make_constraint(layer))

    mdmm_module = mdmm.MDMM(constraints)
    opt = mdmm_module.make_optimizer(model.parameters())

    def train():
        model.train()
        i = 0
        losses = []
        for inputs, targets in train_dl:
            i += 1
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = crit(outputs, targets)
            losses.append(loss)
            lagrangian, norms = mdmm_module(loss)
            opt.zero_grad()
            lagrangian.backward()
            opt.step()
            if i % 100 == 0:
                print(f'{i} {sum(losses[-100:]) / 100:g}')
                print('Layer weight norms:', *(f'{norm.item():g}' for norm in norms))

    def val():
        print('Validating...')
        model.eval()
        losses = []
        for inputs, targets in val_dl:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = crit(outputs, targets)
            losses.append(loss * len(outputs))
        loss = sum(losses) / len(val_set)
        print(f'Validation loss: {loss.item():g}')

    try:
        for epoch in range(1, EPOCHS + 1):
            print('Epoch', epoch)
            train()
            val()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
