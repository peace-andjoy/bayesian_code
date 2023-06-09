import math
import torch
import pickle
import torch.cuda
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as dsets
import os
from utils.BBBConvmodel import BBBAlexNet, BBBLeNet, BBB3Conv3FC
from utils.BBBlayers import GaussianVariationalInference
import numpy as np
import scipy.stats
cuda = torch.cuda.is_available()
import pandas as pd

'''
HYPERPARAMETERS
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--b', type=int, help="batchsize", default=32)
parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
parser.add_argument('--dataset', type=str, default="MNIST", help="Which dataset")
parser.add_argument('--epoch', type=int, default=50, help="num of epoch")

args = parser.parse_args()

is_training = True  # set to "False" to only run validation
num_samples = 10  # because of Casper's trick
batch_size = args.b
beta_type = "Blundell"
net = BBBLeNet   # LeNet or AlexNet
dataset = args.dataset  # MNIST, CIFAR-10 or CIFAR-100
num_epochs = args.epoch
p_logvar_init = 0
q_logvar_init = -10
lr = args.lr
weight_decay = 0.0005
cuda
# dimensions of input and output
if dataset == 'MNIST':    # train with MNIST
    outputs = 10
    inputs = 1
elif dataset == 'CIFAR-10':  # train with CIFAR-10
    outputs = 10
    inputs = 3
elif dataset == 'CIFAR-100':    # train with CIFAR-100
    outputs = 100
    inputs = 3

if net == BBBLeNet or BBB3Conv3FC:
    resize = 32
elif net == BBBAlexNet:
    resize = 227

'''
LOADING DATASET
'''

if dataset == 'MNIST':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = dsets.MNIST(root="data", download=True, transform=transform)
    val_dataset = dsets.MNIST(root="data", download=True, train=False, transform=transform)

elif dataset == 'CIFAR-100':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR100(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR100(root='data', download=True, train=False, transform=transform)

elif dataset == 'CIFAR-10':
    transform = transforms.Compose([transforms.Resize((resize, resize)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    train_dataset = dsets.CIFAR10(root="data", download=True, transform=transform)
    val_dataset = dsets.CIFAR10(root='data', download=True, train=False, transform=transform)


'''
MAKING DATASET ITERABLE
'''

print('length of training dataset:', len(train_dataset))
n_iterations = num_epochs * (len(train_dataset) / batch_size)
n_iterations = int(n_iterations)
print('Number of iterations: ', n_iterations)

loader_train = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

loader_val = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

'''
INSTANTIATE MODEL
'''

model = net(outputs=outputs, inputs=inputs)

if cuda:
    model.cuda()

'''
INSTANTIATE VARIATIONAL INFERENCE AND OPTIMISER
'''
vi = GaussianVariationalInference(torch.nn.CrossEntropyLoss())
optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

'''
check parameter matrix shapes
'''

# how many parameter matrices do we have?
print('Number of parameter matrices: ', len(list(model.parameters())))

for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
for i, (images, labels) in enumerate(loader_train):
    print(images.shape)
    if i == 0:
        break
'''
TRAIN MODEL
'''

logfile = os.path.join('../logs/diagnostics_{}_lr{}_b{}.txt'.format(dataset, lr, batch_size))
with open(logfile, 'w') as lf:
    lf.write('')


def run_epoch(loader, epoch, is_training=False):
    m = math.ceil(len(loader.dataset) / loader.batch_size)

    accuracies = []
    likelihoods = []
    kls = []
    losses = []

    for i, (images, labels) in enumerate(loader):
        # # Repeat samples (Casper's trick)
        # x = images.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)
        # y = labels.repeat(num_samples)

        x = images.view(-1, inputs, resize, resize)
        y = labels

        if cuda:
            x = x.cuda()
            y = y.cuda()

        if beta_type == "Blundell":
            beta = 2 ** (m - (i + 1)) / (2 ** m - 1)
        elif beta_type == "Soenderby":
            beta = min(epoch / (num_epochs//4), 1)
        elif beta_type == "Standard":
            beta = 1 / m
        else:
            beta = 0

        logits, kl = model.probforward(x)
        loss = vi(logits, y, kl, beta)
        ll = -loss.data.mean() + beta*kl.data.mean()

        if is_training:
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        _, predicted = logits.max(1)
        accuracy = (predicted.data.cpu() == y.cpu()).float().mean()

        accuracies.append(accuracy)
        losses.append(loss.data.mean())
        kls.append(beta*kl.data.mean())
        likelihoods.append(ll)

    diagnostics = {'loss': sum(losses)/len(losses),
                   'acc': sum(accuracies)/len(accuracies),
                   'kl': sum(kls)/len(kls),
                   'likelihood': sum(likelihoods)/len(likelihoods)}

    return diagnostics

for epoch in range(num_epochs):
    if is_training is True:
        diagnostics_train = run_epoch(loader_train, epoch, is_training=True)
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_train = dict({"type": "train", "epoch": epoch}, **diagnostics_train)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_train)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_train))
            lf.write(str(diagnostics_val))
    else:
        diagnostics_val = run_epoch(loader_val, epoch)
        diagnostics_val = dict({"type": "validation", "epoch": epoch}, **diagnostics_val)
        print(diagnostics_val)

        with open(logfile, 'a') as lf:
            lf.write(str(diagnostics_val))
model.state_dict().keys()

torch.save(model, "../results/lenet_b{}_lr{}epoch{}_{}.pth".format(batch_size, lr, num_epochs, dataset))

# def compress_coordinates(means, stds, beta, bitlengths):
#     # N = len(means.ravel()) = len(stds.ravel())
#     # C = len(codepoints)
#     optima = np.empty_like(means)
#     for i in range(0, 10000000, 100000):
#         if i % 100000 == 0:
#             print(i / 1000000)
#         squared_errors = (codepoints[np.newaxis, :] - means.ravel()[i:i+100000, np.newaxis])**2
#             # shape (N, C)
#         weighted_penalties = (2 * beta) * stds.ravel()[i:i+100000, np.newaxis]**2 * bitlengths[np.newaxis, :]
#             # shape (N, C)
#         optima_idxs = np.argmin(squared_errors + weighted_penalties, axis=1)
#         optima.ravel()[i:i+100000] = codepoints[optima_idxs]
# #     optima_lengths = bitlengths[optima_idxs].reshape(means.shape)
#     return optima #, None # optima_lengths
# model.state_dict().keys()
# all_names = model.state_dict().keys()
# compressed_list = []

# name = ['layers.0.qw_', 'layers.0.qb_', 
#             'layers.3.qw_', 'layers.3.qb_', 
#               'layers.7.qw_','layers.7.qb_',
#               'layers.9.qw_',  'layers.9.qb_', 
#               'layers.11.qw_', 'layers.11.qb_']

# results = [['full', 'full', diagnostics_train['acc'].numpy(), diagnostics_val['acc'].numpy()]]
# betas = [0.01, 0.1, 1]
# bitlengths = [2, 5, 10]

# for max_codepoint_length in bitlengths:
#     for beta_ in betas:
#         for i in name:
#             vecs_u = model.state_dict()['{}mean'.format(i)].numpy()
#             if '{}logvar'.format(i) in all_names:
#                 stds_u = np.exp(model.state_dict()['{}logvar'.format(i)].numpy())
#             else:
#                 stds_u = np.exp(model.state_dict()['{}si'.format(i)].numpy())
#             empirical_mean = 0
#             empirical_std = np.sqrt(np.mean(vecs_u.ravel()**2))
#             codepoints_and_lengths = [
#                 (scipy.stats.norm.ppf(codepoint_xi, scale=empirical_std), length)
#                 for length in range(max_codepoint_length+1)
#                 for codepoint_xi in np.arange(0.5**(length+1), 1, 0.5**length)
#             ]
#             codepoints = np.array([codepoint for codepoint, _ in codepoints_and_lengths])
#             lengths = np.array([length for _, length in codepoints_and_lengths])
#             compressed = compress_coordinates(vecs_u,stds_u,beta_,lengths)
#             compressed = torch.from_numpy(compressed)
#             compressed_list.append(compressed)
#         len(compressed_list)
#         # lenet
#         from model import lenet
#         cpr_model=lenet()

#         cpr_model.state_dict().keys()
#         name_cpr=cpr_model.state_dict().keys()
#         cpr_state_dict = dict(zip(name_cpr, compressed_list))
#         cpr_model.load_state_dict(cpr_state_dict)
#         # 用压缩后的模型计算loss，acc等
#         def evaluate(loader, epoch=1):
#             m = math.ceil(len(loader.dataset) / loader.batch_size)

#             accuracies = []

#             for i, (images, labels) in enumerate(loader):
#                 # # Repeat samples (Casper's trick)
#                 # x = images.view(-1, inputs, resize, resize).repeat(num_samples, 1, 1, 1)
#                 # y = labels.repeat(num_samples)

#                 x = images.view(-1, inputs, resize, resize)
#                 y = labels

#                 if cuda:
#                     x = x.cuda()
#                     y = y.cuda()

#                 logits = cpr_model(x)
#                 _, predicted = torch.max(logits.data, 1)
#                 accuracy = (predicted.data.cpu() == y.cpu()).float().mean()

#                 accuracies.append(accuracy)

#             diagnostics = {'acc': sum(accuracies)/len(accuracies)}

#             return diagnostics
#         diagnostics_cpr_train = evaluate(loader_train)
#         diagnostics_cpr_val = evaluate(loader_val)
#         results.append([max_codepoint_length, beta_, diagnostics_cpr_train['acc'].numpy(), diagnostics_cpr_val['acc'].numpy()])
# results = pd.DataFrame(results)
# results.to_csv("./results/lenet_b{}_lr{}_{}.csv".format(batch_size, lr, dataset))
print('end!')