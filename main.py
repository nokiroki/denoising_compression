from __future__ import print_function
import matplotlib.pyplot as plt

import os

import numpy as np
from utils import crop_image, pil_to_np, get_image, get_noisy_image,\
    plot_image_grid, get_noise, np_to_torch, torch_to_np, get_params
from models import skip, optimize

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

imsize = -1
PLOT = False
sigma = 50
sigma_ = sigma / 255

fname = 'data/F16_GT.png'

# Image loading
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)

img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

if PLOT:
    plot_image_grid([img_np, img_noisy_np], 4, 6)
# -------------------

# Setup block
INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 20.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

num_iter = 3000
input_depth = 32
figsize = 4

skip_n33d = 128
skip_n33u = 128
skip_n11 = 4
num_scales = 5
upsample_mode = 'bilinear'
downsample_mode = 'stride'
n_channels = 3
act_fun = 'LeakyReLU'

net = skip(input_depth, n_channels,
           num_channels_down=[skip_n33d] * num_scales,
           num_channels_up=[skip_n33u] * num_scales,
           num_channels_skip=[skip_n11] * num_scales,
           upsample_mode=upsample_mode, downsample_mode=downsample_mode,
           need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

net = net.to('cuda')

# Коммент насчёт работоспособности - ниже будут три строчки, которые я заменил
# Я не уверен, нужны ли они и правильно ли я заменил, но если будет падать, попробуй вернуть как было
# net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).to(device)

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
print('Number of params: %d' % s)

# Loss
# mse = torch.nn.MSELoss().type(dtype)
mse = torch.nn.MSELoss()

# img_noisy_torch = np_to_torch(img_noisy_np).type
img_noisy_torch = np_to_torch(img_noisy_np).to(device)
# --------------------


# Optimize
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0


def closure():
    global i, out_avg, psrn_noisy_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])

    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
    i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1),
                         np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
# ---------------------


out_np = torch_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13)