# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
# Imports
import torch
from torch.utils.data.dataset import Dataset
import scipy.io
# import matplotlib
import mat73
# import torchaudio
import warnings
import scipy.signal
import torch.nn.functional as F


warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)


def resample(signal, new_length):
    original_length = signal.shape[1]
    scale_factor = new_length / original_length
    # Reshape the signal to (batch_size * n_feature, time_step) for interpolation
    reshaped_signal = signal.permute(0, 2, 1).reshape(-1, original_length)
    # Resample the signal using linear interpolation
    resampled_signal = F.interpolate(reshaped_signal.unsqueeze(1), scale_factor=scale_factor, mode='linear')
    # Reshape the resampled signal back to (batch_size, n_feature, new_length)
    resampled_signal = resampled_signal.squeeze(1).reshape(signal.shape[0], signal.shape[2], new_length)
    # Transpose the signal to (batch_size, new_length, n_feature) for consistent shape
    resampled_signal = resampled_signal.permute(0, 2, 1)
    return resampled_signal


# 10th order NARMA task
class TimeSeriesDataset(Dataset):
    # Constructor
    def __init__(self, sample_len, n_samples, shift_time, input_dim, output_dim, extra_dim, eval=False):
        """
        Constructor
        :param sample_len: Length of the time-series in time steps.
        :param n_samples: Number of samples to generate.
        """
        # Properties
        self.sample_len = sample_len
        self.n_samples = n_samples
        self.shift_time = shift_time
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.extra_dim = extra_dim


        # Generate data set
        self.inputs, self.outputs = self._generate(eval)
    # end __init__

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return self.n_samples
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        return self.inputs[idx], self.outputs[idx]
    # end __getitem__

    ##############################################
    # PRIVATE
    ##############################################

    # Generate
    def _generate(self, eval=False):
        """
        Generate dataset
        :return:
        """

        # data = scipy.io.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_train2.mat')
        # data_test = scipy.io.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_test2_new.mat')
        #
        # Xtr = torch.FloatTensor(data['series']).transpose(-1, -2)
        # Xtr_test = torch.FloatTensor(data_test['series']).transpose(-1, -2)
        #
        #
        # if eval is False:
        #     ins = Xtr[:, 0:self.sample_len, 0:self.input_dim]
        #     if self.extra_dim is not None:
        #         ins_target = Xtr[:, :, 2:][:self.n_samples][0:self.sample_len, :]
        #
        #     # Xtr = resample(ins, int(ins.size(1) / 6))
        #     # ins = Xtr
        #
        #
        #     if self.extra_dim is not None:
        #         outputs = ins_target[:, self.shift_time:self.sample_len + self.shift_time, :]
        #     else:
        #         outputs = Xtr[:, self.shift_time:self.sample_len + self.shift_time, 0:self.input_dim]
        #         # outputs = torch.cat([outputs, torch.zeros([outputs.size(0), self.shift_time, outputs.size(-1)])], dim=1)
        #
        #
        #     inputs = list(ins)
        #     outputs = list(outputs)
        # else:
        #     ins = Xtr_test[:, 0:self.sample_len, 0:self.input_dim]
        #     # Xtr_test = resample(ins, int(ins.size(1) / 6))
        #     # ins = Xtr_test
        #
        #
        #     if self.extra_dim is not None:
        #         ins_target = Xtr_test[:, :, 2:][0:self.n_samples][0:self.sample_len, :]
        #
        #     if self.extra_dim is not None:
        #         outputs = ins_target[:, self.shift_time:self.sample_len + self.shift_time, :]
        #     else:
        #         outputs = Xtr_test[:, self.shift_time:self.sample_len + self.shift_time, 0:self.input_dim]
        #         # outputs = torch.cat([outputs, torch.zeros([outputs.size(0), self.shift_time, outputs.size(-1)])], dim=1)
        #
        #
        #     inputs = list(ins)
        #     outputs = list(outputs)

        # New dataloader
        # data_train_ori = scipy.io.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_train4.mat')
        # data_train_target = scipy.io.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_target4.mat')

        # data_train_ori = mat73.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_train4_5000samples.mat')
        # data_train_target = mat73.loadmat(r'/home/zhyuan/Desktop/ESN/data/rhythm_target4_5000samples.mat')

        data_test = scipy.io.loadmat(r'./rhythm_test2_new.mat')


        # Xtr = torch.FloatTensor(data_train_ori['series']).transpose(-1, -2)
        # Xtr_target = torch.FloatTensor(data_train_target['target']).transpose(-1, -2)
        Xtr_test = torch.FloatTensor(data_test['series']).transpose(-1, -2)



        if eval is False:
            ins = Xtr[:, 0:self.sample_len, 0:self.input_dim]
            # ins = Xtr[:, 0:self.sample_len, 1:2]
            outputs = Xtr_target[:, 0:self.sample_len, 0:self.output_dim]
            # outputs = Xtr_target[:, 0:self.sample_len, 1:2]

            inputs = list(ins)
            outputs = list(outputs)
        else:
            # test set (12)
            ins = Xtr_test[:, 0:self.sample_len, 0:self.input_dim]
            # ins = Xtr_test[:, 0:self.sample_len, 1:2]
            outputs = Xtr_test[:, self.shift_time:self.sample_len + self.shift_time, 0:self.output_dim]
            # outputs = Xtr_test[:, self.shift_time:self.sample_len + self.shift_time, 1:2]

            inputs = list(ins)
            outputs = list(outputs)


            # ins = Xtr[:, 0:self.sample_len, 0:self.input_dim]
            # outputs = Xtr_target[:, 0:self.sample_len, 0:self.output_dim] * 5
            #
            # inputs = list(ins)
            # outputs = list(outputs)

            # flashback_order = [4, 5, 4, 3, 4, 6, 4] # this set can show it has flashback and strange thing is 33 slow works
            # # flashback_order = [9, 10, 11, 10, 4]
            # new_input_list = []
            # new_output_list = []
            # for i in range(len(flashback_order)):
            #     new_input_list.append(inputs[flashback_order[i]])
            #     new_output_list.append(outputs[flashback_order[i]])
            # inputs = new_input_list
            # outputs = new_output_list



        return inputs, outputs

