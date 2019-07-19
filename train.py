import argparse
import random
import math
import copy

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def init_freeze(model):
    for p in model.parameters():
        p.frozen = False

def requires_grad(model, flag=True):
    for p in model.parameters():
        if not p.frozen:
            p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


# def mse(input, target):
#     return torch.mean((input - target) ** 2)
#
#
# def get_weight_distance(w_full, w_freeze):
#     distances= []
#     for k in w_freeze:
#         distances.append(mse(w_full[k], w_freeze[k]))

    return torch.tensor(distances).mean()

def train(args, dataset, generator, discriminator):
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    # gen_freeze_dict = {}
    # disc_freeze_dict = {}

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if resolution == args.init_size or final_progress:
            alpha = 1

        check = args.phase * 2

        if used_sample > check:
            if step > 2:
                # # Weight matching!
                # for n in range(step + 1):
                #     gen_key = 'match.' + str(n)
                #     for k in generator.module.state_dict():
                #         if gen_key in k:
                #             gen_freeze_dict[k] = copy.deepcopy(generator.module.state_dict()[k])
                #             gen_freeze_dict[k].requires_grad=False
                #
                # for n in range(step + 2):
                #     disc_key = 'match.' + str(8 - n)
                #     for k in discriminator.module.state_dict():
                #         if disc_key in k:
                #             disc_freeze_dict[k] = copy.deepcopy(discriminator.module.state_dict()[k])
                #             disc_freeze_dict[k].requires_grad=False
                #
                # if step == 3:
                #     for k in generator.module.state_dict():
                #         if k.startswith('style'):
                #             gen_freeze_dict[k] = copy.deepcopy(generator.module.state_dict()[k])
                #             gen_freeze_dict[k].requires_grad=False

                # Weight matching!
                for n in range(step + 1):
                    gen_key = 'match.' + str(n)
                    for name, k in generator.named_parameters():
                        if gen_key in name:
                            k.requires_grad = False
                            k.frozen = True

                for n in range(step + 2):
                    disc_key = 'match.' + str(8 - n)
                    for name, k in discriminator.named_parameters():
                        if disc_key in name:
                            k.requires_grad = False
                            k.frozen = True

                if step == 3:
                    print('Freezing Style network now!')
                    for name, k in generator.named_parameters():
                        if name.startswith('module.style'):
                            k.requires_grad = False
                            k.frozen = True

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True

            else:
                alpha = 0

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict()
                },
                f'/scratch/bunk/results/torchgan/checkpoint/train_step-{step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_predict).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_predict.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                    (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()
            disc_loss_val = (real_predict - fake_predict).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            disc_loss_val = (real_predict + fake_predict).item()

        # if step > 1:
        #     disc_loss = nn.MSELoss()
        #     disc_losses = []
        #     for k in disc_freeze_dict:
        #         tmp = disc_loss(discriminator.module.state_dict()[k], disc_freeze_dict[k])
        #         if not torch.isnan(tmp):
        #             disc_losses.append(tmp)
        #
        #     # change
        #     disc_freeze_loss = Variable(torch.tensor(disc_losses).mean(), requires_grad=False)
        #
        #     #disc_freeze_loss.backward()
        #     disc_loss_freeze_val = disc_freeze_loss.item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                loss = -predict.mean()

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            loss.backward()
            gen_loss_val = loss.item()

            # if step > 1:
            #     gen_loss = nn.MSELoss()
            #     gen_losses = []
            #
            #     for k in gen_freeze_dict:
            #         tmp = gen_loss(generator.module.state_dict()[k], gen_freeze_dict[k]) * 10000
            #         if not torch.isnan(tmp):
            #             gen_losses.append(tmp)
            #
            #     # CHANGE DEBUG
            #     gen_freeze_loss = Variable(torch.tensor(gen_losses).mean(), requires_grad=False)
            #
            #     #gen_freeze_loss.backward()
            #     gen_loss_freeze_val = gen_freeze_loss.item()

            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))

            with torch.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            torch.randn(gen_j, code_size).cuda(), step=step, alpha=alpha
                        ).data.cpu()
                    )

            if step < 4:
                utils.save_image(
                    torch.cat(images, 0),
                    f'/scratch/bunk/results/torchgan/sample/{str(i + 1).zfill(6)}.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                )
            else:
                utils.save_image(
                    torch.cat(images, 0),
                    f'/scratch/bunk/results/torchgan/sample/{str(i + 1).zfill(6)}-' + str(gen_loss_freeze_val) + '-' + str(disc_loss_freeze_val) + '.png',
                    nrow=gen_i,
                    normalize=True,
                    range=(-1, 1),
                    )


        if (i + 1) % 10000 == 0:
            torch.save(
                g_running.state_dict(), f'/scratch/bunk/results/torchgan/checkpoint/{str(i + 1).zfill(6)}.model'
            )

        # if step > 1:
        #     state_msg = (
        #         f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; GF:{gen_loss_freeze_val:.3f}; D: {disc_loss_val:.3f}; DF:{disc_loss_freeze_val:.3f}; '
        #         f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        #     )
        # else:
        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=512, type=int, help='max image size')
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    generator = nn.DataParallel(StyledGenerator(code_size)).cuda()
    discriminator = nn.DataParallel(Discriminator()).cuda()
    g_running = StyledGenerator(code_size).cuda()
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5,), (0.5)),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    init_freeze(generator)
    init_freeze(discriminator)

    train(args, dataset, generator, discriminator)
