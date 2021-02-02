#!/usr/bin/env python3

"""An MNIST classifier using MDMM to enforce layer norm constraints."""

import argparse
import csv
from functools import partial

import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms

import mdmm

BATCH_SIZE = 50
EPOCHS = 100


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--norm', type=float, default=1.,
                   help='the target layer norm')
    p.add_argument('--scale', type=float, default=1.,
                   help='the infeasibility scale factor')
    p.add_argument('--damping', type=float, default=20.,
                   help='the damping strength')
    p.add_argument('--lr', type=float, default=0.02,
                   help='the learning rate')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(0)

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
    print('Parameters:', sum(param.numel() for param in model.parameters()))

    crit = nn.CrossEntropyLoss()

    constraints = []
    for layer in model:
        if hasattr(layer, 'weight'):
            fn = partial(lambda x: x.weight.abs().mean(), layer)
            constraints.append(mdmm.EqConstraint(fn, args.norm,
                                                 scale=args.scale, damping=args.damping))

    mdmm_module = mdmm.MDMM(constraints)
    opt = mdmm_module.make_optimizer(model.parameters(), lr=args.lr)

    writer = csv.writer(open('mdmm_demo_mnist.csv', 'w'))
    writer.writerow(['loss', 'norm_1', 'norm_2', 'norm_3'])

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
            mdmm_return = mdmm_module(loss)
            writer.writerow([loss.item(), *(norm.item() for norm in mdmm_return.fn_values)])
            opt.zero_grad()
            mdmm_return.value.backward()
            opt.step()
            if i % 100 == 0:
                print(f'{i} {sum(losses[-100:]) / 100:g}')
                print('Layer weight norms:',
                      *(f'{norm.item():g}' for norm in mdmm_return.fn_values))

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
