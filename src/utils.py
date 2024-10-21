import torch
import torchvision
import os
from datetime import timedelta
import numpy as np
import random
from omegaconf import OmegaConf

from src import ff_mnist, ff_model

def get_accuracy(opt, output, target):
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size

def get_data(opt, partition):
    dataset = ff_mnist.FF_MNIST(opt, partition)
    
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(dataset, batch_size=opt.input.batch_size, generator=g, drop_last=True, shuffle=True, num_workers=4)


def get_MNIST_partition(opt, partition, colab=False):
    if colab:
        path = './data/'
    else:
        path = os.path.join(os.getcwd(), opt.input.path)
    if partition in ['train', 'val', 'train_val']:
        mnist = torchvision.datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
    elif partition in ['test']:
        mnist = torchvision.datasets.MNIST(
            path,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
    else:
        raise NotImplementedError
    
    if partition == 'train':
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == 'val':
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))
    
    return mnist

def update_learning_rate(opt, optimizer, epoch):
    """Update the learning rate of the optimizer."""
    optimizer.param_groups[0]['lr'] = get_linear_cooldown_lr(opt, epoch, opt.training.learning_rate)
    optimizer.param_groups[1]['lr'] = get_linear_cooldown_lr(opt, epoch, opt.training.downstream_learning_rate)
    return optimizer


def get_linear_cooldown_lr(opt, epoch, lr):
    """Decrease the learning rate linearly after the halfway point."""
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def dict_to_cude(dict):
    for k, v in dict.items():
        dict[k] = v.cuda(non_blocking=True)


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cude(inputs)
        labels = dict_to_cude(labels)
    return inputs, labels


def log_results(results, outputs, num_steps_per_epoch):
    for k, v in outputs.items():
        if isinstance(v, float):
            results[k] += v / num_steps_per_epoch
        else:
            results[k] += v.item() / num_steps_per_epoch
    return results

def print_results(partition, iteration_time, outputs, epoch=None):
    if epoch is not None:
        print(f'Epoch: {epoch} \t', end='')
    
    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )

    if outputs is not None:
        for k, v in outputs.items():
            print(f'{k}: {v:.4f} \t', end='')
    print()

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt

def get_model_and_optimizer(opt):
    model = ff_model.FFModel(opt)
    if "cuda" in opt.device:
        model = model.cuda()
    print(model, '\n')

    main_model_params = [
        p for p in model.parameters() if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                'params': main_model_params,
                'lr': opt.training.learning_rate,
                'weight_decay': opt.training.weight_decay,
                'momemtum': opt.training.momentum
            },
            {
                'params': model.linear_classifier.parameters(),
                'lr': opt.training.downstream_learning_rate,
                'weight_decay': opt.training.downstream_weight_decay,
                'momentum': opt.training.momentum
            }
        ]
    )
    return model, optimizer


