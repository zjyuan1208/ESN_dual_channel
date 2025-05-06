#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

# Imports
import torch
# from TimeSeriesDataset import TimeSeriesDataset
# from EuESN_maml import EuESN_maml
from echotorch.datasets.TimeSeriesDataset import TimeSeriesDataset
import echotorch.nn.reservoir as etrs
import echotorch.utils
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch.nn.functional as F
import json


import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# eval state
eval_state = True
mask_state = False
if eval_state is True:
    train_weight_state = False
else:
    train_weight_state = True


num_layer = 1

# shift = 160
shift = 200

# Length of training samples
train_sample_length = 24000
# train_sample_length = 640


# Length of test samples
test_sample_length = 29800
# test_sample_length = 640

# How many training/test samples
n_train_samples = 1000
# n_train_samples = 20
n_test_samples = 12

# Batch size (how many sample processed at the same time?)
batch_size = 40
# batch_size = 12

# Reservoir hyper-parameters
spectral_radius = 0.6
leaky_rate = [0.03, 0.3]
input_dim = 2
output_dim = 2
reservoir_size = 4800
connectivity = 1.0
ridge_param = 0.0000001

# Predicted/target plot length
# plot_length = 30000
# plot_length = 4966
plot_length = 15000

# Use CUDA?
use_cuda = False
use_cuda = torch.cuda.is_available() if use_cuda else False

pt_file = f'./euler_esn_{leaky_rate[0]}_woadapt_wotranspose_{n_train_samples}samples_noise_mse.pt'

# Manual seed initialisation
np.random.seed(1)
torch.manual_seed(1)

def resample(signal, new_length):
    original_length = signal.shape[1]
    scale_factor = new_length / original_length
    # Reshape the signal to (batch_size * n_feature, time_step)
    reshaped_signal = signal.permute(0, 2, 1).reshape(-1, original_length)
    # Resample the signal using linear interpolation
    resampled_signal = F.interpolate(reshaped_signal.unsqueeze(1), scale_factor=scale_factor, mode='linear')
    resampled_signal = resampled_signal.squeeze(1).reshape(signal.shape[0], signal.shape[2], new_length)
    resampled_signal = resampled_signal.permute(0, 2, 1)
    return resampled_signal


def threshold(tensor, threshold_value):
    tensor[tensor > threshold_value] = tensor.max()
    tensor[tensor <= threshold_value] = 0
    return tensor


esn = etrs.EuESN_maml(input_dim=input_dim, n_reservoir1=reservoir_size, n_reservoir2=reservoir_size, n_out=output_dim, connectivity=connectivity, spectral_radius=spectral_radius,
                 leaky_rate1=leaky_rate[0], leaky_rate2=leaky_rate[0], downsample_factor=40, prediction_horizon=int(shift/40), eval_state=True)

# Transfer in the GPU if possible
if use_cuda:
    esn.to(device)
# end if

if __name__ == '__main__':
    # extractor = parallelTestModule.ParallelExtractor()
    # extractor.runInParallel(numProcesses=2, numThreads=4)

    if eval_state is False:
        print('start')
        T1 = time.time()
        train_dataset = TimeSeriesDataset(train_sample_length, n_train_samples, shift_time=shift, input_dim=input_dim, output_dim=output_dim, extra_dim=None, eval=False)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        # For each batch
        for data in trainloader:
            # Inputs and outputs
            inputs, targets = data

            # Transform data to Variables
            inputs, targets = Variable(inputs), Variable(targets)

            inputs = resample(inputs, int(inputs.size(1) / 6))
            targets = resample(targets, int(targets.size(1) / 6))

            # inputs = threshold(inputs)
            # targets = threshold(targets, threshold_value=80)

            # threshold_value = 30
            # targets = threshold(targets, threshold_value)

            if use_cuda: inputs, targets = inputs.to(device), targets.to(device)

            esn(inputs, targets, eval_state=eval_state)
        # end for

        pt_path = pt_file
        # pt_path = f'/home/zhyuan/Desktop/ESN/checkpoint/adapt_c_checkpoint_with_new_data/random_esn_{leaky_rate[0]}.pt'
        torch.save(esn, pt_path)
        print('training end')

    else:
        T1 = time.time()

        test_dataset = TimeSeriesDataset(test_sample_length, n_test_samples, shift_time=shift, input_dim=input_dim,
                                         output_dim=output_dim, extra_dim=None, eval=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        dataiter = iter(testloader)
        test_u, test_y = dataiter.__next__()
        test_u, test_y = Variable(test_u), Variable(test_y)

        test_u = resample(test_u, int(test_u.size(1) / 6))
        test_y = resample(test_y, int(test_y.size(1) / 6))

        # test_u = resample(test_u, int(test_u.size(1) / 10))
        # test_y = resample(test_y, int(test_y.size(1) / 10))

        # threshold_value = 70
        # test_y = threshold(test_y, threshold_value)

        # larger_tensor = torch.zeros_like(test_u[:, 0:2000, :])
        # test_u = torch.cat((test_u, larger_tensor), dim=1)
        # test_y = torch.cat((test_y, larger_tensor), dim=1)

        if use_cuda: test_u, test_y = test_u.to(device), test_y.to(device)

        new_m = torch.load(pt_file, map_location=device)
        if use_cuda: new_m = new_m.to(device)
        new_m.train_weight = train_weight_state
        # y_predicted, hidden_state, eps_c, norm_feature0, norm_feature1, target_fb, target_changed = new_m(complicated_test_u, complicated_test_y, eval_state=eval_state)
        y_predicted, hidden_state, eps_c, norm_feature0, norm_feature1, target_fb, target_changed = new_m(test_u, test_y, eval_state=eval_state)

        # y_predicted = esn(test_u)
        T2 = time.time()
        print('The testing costs %s seconds' % (T2 - T1))

        print(u"")


        save_path = f'./figure'
        os.makedirs(save_path)

        save_fig = False
        colors = ['#D4B4A1', '#80221E', '#80221E', '#AD7C59', '#B85C48', '#CABCAB']
        for i in range(n_test_samples):
            # target = test_y.detach().cpu()[i, :plot_length, 0].data
            target_fb = target_fb.reshape(n_test_samples, -1, output_dim).detach().cpu()
            target_changed = target_changed.reshape(n_test_samples, -1, output_dim).detach().cpu()

            plt.plot(target_changed.detach().cpu()[i, 33:plot_length, 0].data, colors[0], label=f'{i}_feat0_reference')
            # plt.plot(target_fb.detach().cpu()[i, 33:plot_length, 0].data, colors[3], label=f'{i}_feat0_feedback')
            plt.plot(y_predicted.detach().cpu()[i, :plot_length, 0].data, colors[1], label=f'{i}_feat0_pred')
            plt.legend()
            plt.title(f'example_{i}_feature0')
            if save_fig:
                plt.savefig(f'{save_path}/fig_{leaky_rate[0]}_{i}_feature0.png')
            plt.show()



            # plt.plot(test_y.detach().cpu()[i, :plot_length, 1].data, 'r', label=f'{i}_feat1_reference')
            plt.plot(target_changed.detach().cpu()[i, 33:plot_length, 1].data, colors[0], label=f'{i}_feat1_reference')
            # plt.plot(target_fb.detach().cpu()[i, 33:plot_length, 1].data, colors[3], label=f'{i}_feat1_feedback')
            plt.plot(y_predicted.detach().cpu()[i, :plot_length, 1].data, colors[1], label=f'{i}_feat1_pred')
            plt.legend()
            plt.title(f'example_{i}_feature1')
            if save_fig:
                plt.savefig(f'{save_path}/fig_{leaky_rate[0]}_{i}_feature1.png')
            plt.show()
