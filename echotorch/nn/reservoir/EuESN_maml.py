import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
# import torchaudio.functional as F
# import torchaudio
import tqdm
import torch.optim as optim
import scipy.signal
import copy
import math
import warnings
import random
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
# from echotorch.nn.linear.RRCell import RRCell

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SparseMatrixGenerator(nn.Module):
    def __init__(self, connectivity=1.0, spectral_radius=0.6, apply_spectral_radius=True,
                 scale=1.0, mean=0.0, std=1.0, minimum_edges=0):
        super(SparseMatrixGenerator, self).__init__()

        self.connectivity = connectivity,
        self.apply_spectral_radius = apply_spectral_radius,
        self.scale = scale,
        self.mean = mean,
        self.std = std,
        self.minimum_edges = minimum_edges
        self.spectral_radius_value = spectral_radius

    # Compute spectral radius of a square 2-D tensor
    def spectral_radius(self, m):
        """
        Compute spectral radius of a square 2-D tensor
        :param m: squared 2D tensor
        :return:
        """
        return torch.max(torch.abs(torch.linalg.eig(m)[0])).item()

    def generate(self, size, sparse=False, dtype=torch.float64):
        """
        Generate the matrix
        :param: Matrix size (row, column)
        :param: Data type to generate
        :return: Generated matrix
        """
        # Params
        # connectivity = self.get_parameter('connectivity')
        # mean = self.get_parameter('mean')
        # std = self.get_parameter('std')

        # Full connectivity if none
        if self.connectivity is None:
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=self.mean[0], std=self.std[0])
        else:
            # Generate matrix with entries from norm
            w = torch.zeros(size, dtype=dtype)
            w = w.normal_(mean=self.mean[0], std=self.std[0])

            # Generate mask from bernoulli
            mask = torch.zeros(size, dtype=dtype)
            mask.bernoulli_(p=self.connectivity[0])

            # Add edges until minimum is ok
            while torch.sum(mask) < self.minimum_edges:
                # Random position at 1
                x = torch.randint(high=size[0], size=(1, 1))[0, 0].item()
                y = torch.randint(high=size[1], size=(1, 1))[0, 0].item()
                mask[x, y] = 1.0

            # Mask filtering
            w *= mask

        # Scale
        # w *= self.get_parameter('scale')

        # Set spectral radius
        # If two dim tensor, square matrix and spectral radius is available
        if w.ndimension() == 2 and w.size(0) == w.size(1) and self.apply_spectral_radius:
            # If current spectral radius is not zero
            if self.spectral_radius(w) > 0.0:
                w = (w / self.spectral_radius(w)) * self.spectral_radius_value
            else:
                warnings.warn("Spectral radius of W is zero (due to small size), spectral radius not changed")

        if sparse is True:
            return self.to_sparse(w)
        else:
            return w.float()

    # To sparse matrix
    @staticmethod
    def to_sparse(m):
        """
        To sparse matrix
        :param m:
        :return:
        """
        # Rows, columns and values
        rows = torch.LongTensor()
        columns = torch.LongTensor()
        values = torch.FloatTensor()

        # For each row
        for i in range(m.shape[0]):
            # For each column
            for j in range(m.shape[1]):
                if m[i, j] != 0.0:
                    rows = torch.cat((rows, torch.LongTensor([i])), dim=0)
                    columns = torch.cat((columns, torch.LongTensor([j])), dim=0)
                    values = torch.cat((values, torch.FloatTensor([m[i, j]])), dim=0)

        # Indices
        indices = torch.cat((rows.unsqueeze(0), columns.unsqueeze(0)), dim=0)

        return torch.sparse.FloatTensor(indices, values)


class EulerMatrixGenerator(nn.Module):
    def __init__(self, leaky_rate=0.03, n=40, c=None):
        super(EulerMatrixGenerator, self).__init__()
        setup_seed(1234)

        self.leaky_rate = leaky_rate
        self.n = n

        self.CN = 0.1
        self.c0 = 300
        self.crr = 0.8
        # self.crr = 0


        if c is None:
            self.c = self.c0 * torch.ones([n, n]).to(device)
            self.dc = -250 / n
            # self.dc = 0
            # self.dc = -10 / n
            for i in range(n):
                self.c[:, i] = self.c[:, i] + self.dc * (i - 1)
            self.c = self.c - self.crr * torch.rand([n, n]).to(device) * self.c
        else:
            self.c = c
        self.dc = 0

        self.dt = 1.0
        # self.dt = 0.01
        self.dx = self.dt / self.CN * torch.max(torch.max(self.c)) * math.sqrt(2)
        self.dy = self.dx


        self.k = 0.1
        self.k = self.k + 2.0 * torch.rand([n, n])
        self.dk = - 0.1 / n
        # k = k / 100
        self.dk = self.dk / 100

        self.k = torch.zeros([n, n])
        if torch.cuda.is_available():
            self.k = self.k.to(device)

        self.k = self.k * 0.0
        self.dk = self.dk * 0.0
        self.kp = 0.0001


        self.Nx1 = torch.zeros((n, n))
        self.Nxprime1 = torch.zeros((n, n))
        self.Ny1 = torch.zeros((n, n))
        self.Nyprime1 = torch.zeros((n, n))
        self.M1 = torch.zeros((n, n))
        self.Mprime1 = torch.zeros((n, n))
        self.Rx1 = torch.zeros((n, n))
        self.Rx2 = torch.zeros((n, n))
        self.Sx1 = torch.zeros((n, n))
        self.Sx2 = torch.zeros((n, n))
        self.Ry1 = torch.zeros((n, n))
        self.Sy1 = torch.zeros((n, n))
        if torch.cuda.is_available():
            self.Nx1 = self.Nx1.to(device)
            self.Nxprime1 = self.Nxprime1.to(device)
            self.Ny1 = self.Ny1.to(device)
            self.Nxprime1 = self.Nxprime1.to(device)
            self.M1 = self.M1.to(device)
            self.Mprime1 = self.Mprime1.to(device)
            self.Rx1 = self.Rx1.to(device)
            self.Rx2 = self.Rx2.to(device)
            self.Sx1 = self.Sx1.to(device)
            self.Sx2 = self.Sx2.to(device)
            self.Ry1 = self.Ry1.to(device)
            self.Sy1 = self.Sy1.to(device)

        self.M = torch.zeros((self.n * self.n, self.n * self.n))
        self.Mprime = torch.zeros((self.n * self.n, self.n * self.n))
        self.Nx = torch.zeros((self.n * self.n, self.n * self.n))
        self.Nxprime = torch.zeros((self.n * self.n, self.n * self.n))
        self.Ny = torch.zeros((self.n * self.n, self.n * self.n))
        self.Nyprime = torch.zeros((self.n * self.n, self.n * self.n))
        self.Rx = torch.zeros((self.n * self.n, self.n * self.n))
        self.Sx = torch.zeros((self.n * self.n, self.n * self.n))
        self.Ry = torch.zeros((self.n * self.n, self.n * self.n))
        self.Sy = torch.zeros((self.n * self.n, self.n * self.n))
        self.Minv = torch.zeros_like(self.M)
        self.Nxinv = torch.zeros_like(self.Nx)
        self.Nyinv = torch.zeros_like(self.Ny)
        self.A = torch.zeros((3 * self.n * self.n, 3 * self.n * self.n))
        self.W = torch.zeros_like(self.A)

        if torch.cuda.is_available():
            self.M = self.M.to(device)
            self.Mprime = self.Mprime.to(device)
            self.Nx = self.Nx.to(device)
            self.Nxprime = self.Nxprime.to(device)
            self.Ny = self.Ny.to(device)
            self.Nyprime = self.Nyprime.to(device)
            self.Rx = self.Rx.to(device)
            self.Sx = self.Sx.to(device)
            self.Ry = self.Ry.to(device)
            self.Sy = self.Sy.to(device)
            self.Minv = self.Minv.to(device)
            self.Nxinv = self.Nxinv.to(device)
            self.Nyinv = self.Nyinv.to(device)
            self.A = self.A.to(device)
            self.W = self.W.to(device)

        self.dkx = torch.zeros_like(self.c).to(device)
        self.dky = torch.zeros_like(self.c).to(device)


    def change_diagonal(self, matrix):
        diagonal = torch.diag(matrix)
        reciprocal = 1.0 / diagonal
        matrix.diagonal().copy_(reciprocal)
        return matrix

    def generate(self, n=40, eps_c=None, hidden_states=None, pos=None, neg=None):
        setup_seed(1234)

        if eps_c is not None:
            self.c = self.c * (1 + eps_c)

        self.k_x = 0
        self.k_x = self.k_x * torch.ones_like(self.c)
        self.dk_x = 0

        self.k_y = 0
        self.k_y = self.k_y * torch.ones_like(self.c)
        self.dk_y = 0


        if pos is not None:
            if neg is not None:
                self.dkx[int(pos/self.n), int(pos%self.n)] -= 1
                self.dky[int(pos/self.n), int(pos%self.n)] -= 1
                self.dkx[int(neg/self.n), int(neg%self.n)] += 1
                self.dky[int(neg/self.n), int(neg%self.n)] += 1
        if hidden_states is not None:
            hidden_states_size = (400, 4800)
            baseline_energy_size = 4800

            # Generate random hidden states and baseline energy for testing
            # hidden_states = torch.randn(hidden_states_size)
            # baseline_energy = torch.randn(baseline_energy_size)
            energy_list = []
            for i in range(10):
                save_path = f'/home/zhyuan/Desktop/ESN/timeserie_prediction/avg_states/avg_energy_tensor_{i}.pt'
                loaded_tensor = torch.load(save_path)
                # print(loaded_tensor.shape)

                # for i in range(batch_size):
                time_series_data = loaded_tensor[:1600]
                energy_list.append(time_series_data)

            energy = torch.stack(energy_list)
            baseline_energy = torch.mean(energy, dim=0)

            P = hidden_states[:, 0: 1600]
            ox = hidden_states[:, 1600: 3200]
            oy = hidden_states[:, 3200: 4800]

            # Apply FFT along the feature dimension
            freq_domain_data = torch.fft.fft(hidden_states.detach().cpu(), dim=0)
            energy = torch.abs(freq_domain_data) ** 2

            # Calculate the average energy for each feature
            average_energy = torch.mean(energy, dim=0).to(device)

            # Define indices for P, ox, and oy
            P_indices = torch.arange(1600)
            ox_indices = torch.arange(1600, 3200)
            oy_indices = torch.arange(3200, 4800)

            # Compare P's energy with baseline energy
            P_energy = average_energy[P_indices]
            baseline_P_energy = baseline_energy[P_indices]

            tau = 400
            dt = 6

            # Calculate the exponential factor for integration
            exponential_factor = torch.exp(-torch.arange(1600 - 1).float() * dt / tau)


            exponential_factor = exponential_factor.to(device)

            # Perform the integration using the trapezoidal rule
            accumulated_ox = 0.5 * dt * torch.sum((ox[:, 0:1599] ** 2 + ox[:, 1:1600] ** 2) * exponential_factor)
            accumulated_oy = 0.5 * dt * torch.sum((oy[:, 0:1599] ** 2 + oy[:, 1:1600] ** 2) * exponential_factor)

            # print("Accumulated ox:", accumulated_ox.item())
            # print("Accumulated oy:", accumulated_oy.item())
            average_baseline_threshold = torch.mean(baseline_P_energy)

            if accumulated_ox <= average_baseline_threshold and accumulated_oy <= average_baseline_threshold:
                self.dkx -= 0.0000001 * accumulated_ox
                self.dky -= 0.0000001 * accumulated_oy


            # Add positive value everywhere in dkx and dky
            self.dkx += torch.where(P_energy > average_baseline_threshold, 0.00001, 0).reshape(self.n, self.n)
            self.dky += torch.where(P_energy > average_baseline_threshold, 0.00001, 0).reshape(self.n, self.n)


        for i in range(n):
            self.M1[i, i] = 1 / self.dt - self.kp / 2
            self.Mprime1[i, i] = 1 / self.dt + self.kp / 2

        for i in range(1, self.n + 1):
            # if dk != 0:
            for ii in range(1, self.n + 1):
                self.Nx1[ii - 1, ii - 1] = 1 / self.dt + (self.k_x[ii - 1, i - 1] + self.dkx[ii - 1, i - 1] * (i - 1)) / 2
                self.Nxprime1[ii - 1, ii - 1] = 1 / self.dt - (self.k_x[ii - 1, i - 1] + self.dkx[ii - 1, i - 1] * (i - 1)) / 2
                self.Ny1[ii - 1, ii - 1] = 1 / self.dt + (self.k_y[ii - 1, i - 1] + self.dky[ii - 1, i - 1] * (i - 1)) / 2
                self.Nyprime1[ii - 1, ii - 1] = 1 / self.dt - (self.k_y[ii - 1, i - 1] + self.dky[ii - 1, i - 1] * (i - 1)) / 2

            self.M[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.M1
            self.Mprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Mprime1
            self.Nx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nx1
            self.Nxprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nxprime1
            self.Ny[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Ny1
            self.Nyprime[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Nyprime1

            # if dc != 0:
            for ii in range(1, self.n + 1):
                self.Rx1[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Rx2[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sx1[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sx2[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dx
                self.Sy1[ii - 1, ii - 1] = -(self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                if ii > 1:
                    self.Sy1[ii - 1, (ii - 2) % n] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                self.Ry1[ii - 1, ii - 1] = (self.c[ii - 1, i - 1] + self.dc * (i - 1)) / self.dy
                if ii < n:
                    self.Ry1[ii - 1, ii % n] = -(self.c[ii % n, i - 1] + self.dc * (i - 1)) / self.dy
            self.Rx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Rx1
            self.Sx[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Sx1
            if i < n:
                self.Rx[(i - 1) * self.n:i * self.n, (i % n) * n:((i % n) + 1) * n] = self.Rx2
                self.Sx[(i % n) * n:((i % n) + 1) * n, (i - 1) * self.n:i * self.n] = self.Sx2
            self.Ry[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Ry1
            self.Sy[(i - 1) * self.n:i * self.n, (i - 1) * self.n:i * self.n] = self.Sy1

        self.Minv = torch.inverse(self.M)
        self.Nxinv = torch.inverse(self.Nx)
        self.Nyinv = torch.inverse(self.Ny)

        self.A = torch.vstack([
            torch.hstack([self.Minv @ self.Mprime + self.Minv @ self.Rx @ self.Nxinv @ self.Sx + self.Minv @ self.Ry @ self.Nyinv @ self.Sy,
                          self.Minv @ self.Rx @ self.Nxinv @ self.Nxprime,
                          self.Minv @ self.Ry @ self.Nyinv @ self.Nyprime]),
            torch.hstack(
                [self.Nxinv @ self.Sx, self.Nxinv @ self.Nxprime, torch.zeros((n ** 2, n ** 2), dtype=torch.float).to(device)]),
            torch.hstack(
                [self.Nyinv @ self.Sy, torch.zeros((n ** 2, n ** 2), dtype=torch.float).to(device), self.Nyinv @ self.Nyprime])
        ])


        self.W = (self.A - (1 - self.leaky_rate) * torch.eye(n ** 2 * 3).to(device)) / self.leaky_rate

        return self.A, self.W, self.c, self.k




def detect_non_empty_zero_regions(time_series):
    zero_regions = []
    in_zero_region = False
    start_idx = None

    for i, value in enumerate(time_series):
        if value < 10:
            if not in_zero_region:
                start_idx = i
                in_zero_region = True
        else:
            if in_zero_region:
                if start_idx != i - 10:  # Exclude zero regions with no beats
                    zero_regions.append((start_idx, i))
                in_zero_region = False

    return zero_regions


def generate_negative_sine_wave(length, frequency, amplitude):
    x = torch.linspace(0, 2 * torch.pi, length)
    return -amplitude * torch.sin(frequency * x)

def replace_zero_regions_with_sine_wave(time_series, zero_regions, frequency=1, amplitude=1):
    modified_series = time_series.clone()
    for start, end in zero_regions:
        if start > 0 and end < len(time_series) - 1:  # Ensure we have non-zero regions on both sides
            sine_wave = generate_negative_sine_wave(end - start, frequency, amplitude)
            modified_series[start:end] = sine_wave
    return modified_series


def threshold(tensor, threshold_value):
    # tensor[tensor > threshold_value] = tensor.max()
    tensor[tensor > threshold_value] = 1
    tensor[tensor <= threshold_value] = 0
    return tensor

def generate_periodical_signals(length=5000, pulse_interval=140):
    # Generate a periodic signal with a pulse of 1 every pulse_interval steps
    original_signal = np.zeros(length)
    original_signal[::pulse_interval] = 1

    # Skip a pulse every two beats
    skip_signal = original_signal.copy()
    skip_signal[2 * pulse_interval::3 * pulse_interval] = 0


    return original_signal, skip_signal


def activation_f(x):

    return torch.tanh(x)


class EuESN_maml(nn.Module):
    def __init__(self, input_dim, n_reservoir1, n_reservoir2, n_out, connectivity=0.1, spectral_radius=0.6,
                 leaky_rate1=0.03, leaky_rate2=0.7, downsample_factor=40, prediction_horizon=5, mask_state=True, eval_state=True):
        super(EuESN_maml, self).__init__()

        self.input_dim = input_dim
        self.n_reservoir1 = n_reservoir1
        self.n_reservoir2 = n_reservoir2
        self.n_out = n_out
        self.connectivity = connectivity
        self.spectral_radius = spectral_radius
        self.leaky_rate1 = leaky_rate1
        self.leaky_rate2 = leaky_rate2
        self.downsample_factor = downsample_factor
        self.prediction_horizon = prediction_horizon
        self.output_learning_rate = 0.01
        self.momentum = 0.001
        self.shift_time = 200
        self.mask_state = mask_state
        self.eval_state = eval_state

        self.n = int(math.sqrt(int(self.n_reservoir1 / 3)))

        # random
        weight_generator = SparseMatrixGenerator(connectivity=self.connectivity, spectral_radius=self.spectral_radius,
                                                 apply_spectral_radius=True,
                                                 scale=1.0, mean=0.0, std=1.0, minimum_edges=0)

        self.w_weight_generator = EulerMatrixGenerator()

        self.Win1 = weight_generator.generate((input_dim, n_reservoir1))
        nn.init.xavier_uniform_(self.Win1)

        self.W_fb = weight_generator.generate((input_dim, n_reservoir1))
        nn.init.xavier_uniform_(self.W_fb)



        self.epskx = torch.zeros([self.n * self.n, 1])
        self.epsky = torch.zeros([self.n * self.n, 1])
        self.A, self.W_res1, self.c, self.k = self.w_weight_generator.generate(n=self.n)

        self.b_res1 = torch.randn([n_reservoir1, 1])
        nn.init.sparse_(self.b_res1, sparsity=0.1, std=0.01)
        self.b_res1 = self.b_res1.squeeze(-1)

        self.W_out = torch.randn(n_out, int(self.n_reservoir1/3))
        nn.init.xavier_uniform_(self.W_out)

        if torch.cuda.is_available():
            self.Win1 = self.Win1.to(device)
            self.W_fb = self.W_fb.to(device)
            self.W_res1 = self.W_res1.to(device)
            self.b_res1 = self.b_res1.to(device)
            self.W_out = self.W_out.to(device)

        self.leaky_rate = leaky_rate1

    def change_diagonal(self, matrix):
        diagonal = torch.diag(matrix)
        reciprocal = 1.0 / diagonal
        matrix.diagonal().copy_(reciprocal)
        return matrix


    def smape_loss(self, output, target):

        nonzero_mask = target != 0
        target = target[nonzero_mask]
        output = output[nonzero_mask]

        numerator = torch.abs(output - target)
        denominator = torch.abs(output) + torch.abs(target)
        elementwise_smape = torch.div(numerator, denominator)

        nan_mask = torch.isnan(elementwise_smape)
        loss = elementwise_smape[~nan_mask].mean() * 200

        return loss


    def task_metalearn(self, input, target_data, update_time_step, update_state):
        """ Perform gradient descent for one task in the meta-batch. """
        # best version
        # num_updates = 40
        # update_lr = 0.1
        num_updates = 20
        update_lr = 0.1
        loss = 0
        if update_state:
            for j in range(num_updates - 1):
                output = self.forward_meta(input, target_data, seq_len=update_time_step)
                # loss = self.smape_loss(output, target_data)
                loss = nn.MSELoss()(output, target_data)
                grads = torch.autograd.grad(loss, self.W_out, create_graph=True)
                self.W_out = torch.nn.Parameter(self.W_out - update_lr * grads[0])
        return loss


    def update_c(self, eps_c=0):
        self.A, self.W_res1, self.c, self.k = self.w_weight_generator.generate(n=self.n, eps_c=eps_c)

    def update_c_k(self, eps_c=0, hidden_states=None, pos=None, neg=None):
        self.A, self.W_res1, self.c, self.k = self.w_weight_generator.generate(n=self.n, eps_c=eps_c, hidden_states=hidden_states, pos=pos, neg=neg)


    def forward_meta(self, input_data, target_data, seq_len):
        self.Win1[:, 40:] = 0

        # eval state
        target_data = target_data.reshape(-1, self.input_dim)
        x1 = torch.zeros(seq_len, self.n_reservoir1)
        if torch.cuda.is_available():
            x1 = x1.to(device)

        updata_state = False

        eps_c_sum = 0
        eps_c_lst = []

        for t in range(int(seq_len)):


            u = input_data[t, :]

            if t % seq_len == 0:
                x1[t, :] = self.leaky_rate1 * F.tanh(u @ self.Win1 + self.b_res1)
            else:
                # self.b_res1 = torch.randn([self.n_reservoir1, 1])
                # nn.init.sparse_(self.b_res1, sparsity=0.1, std=0.01)
                # self.b_res1 = self.b_res1.squeeze(-1).to(device)
                x1[t, :] = (1 - self.leaky_rate1) * x1[t - 1, :] + self.leaky_rate1 * torch.tanh(u @ self.Win1 +
                                                                                                 x1[t - 1,
                                                                                                 :] @ self.W_res1 +
                                                                                                 self.b_res1)

        y_out = x1[:, :int(self.n_reservoir1 / 3)] @ self.W_out.t()

        # y_out = y_out.reshape(seq_len, self.n_out)

        return y_out

    def forward(self, input_data, target_data=None, eval_state=False):
        batch_size = input_data.size(0)
        seq_len = input_data.size(1)

        input_data = input_data.reshape(-1, self.input_dim)

        self.Win1[:, 40:] = 0


        if eval_state is False:
            x1 = torch.zeros(batch_size, seq_len, self.n_reservoir1)

            if torch.cuda.is_available():
                x1 = x1.to(device)
                input_data = input_data.to(device)

            w_out_layer = nn.Linear(in_features=int(self.n_reservoir1 / 3), out_features=self.n_out, bias=False)
            w_out_layer.weight = torch.nn.Parameter(self.W_out.detach().cpu())
            if torch.cuda.is_available():
                w_out_layer = w_out_layer.to(device)
            optimizer = optim.SGD(w_out_layer.parameters(), lr=self.output_learning_rate, momentum=self.momentum)
            # criterion = nn.MSELoss()
            update_steps = 40
            self.W_out = w_out_layer.weight


            with tqdm.tqdm(total=batch_size) as pbar:
                eps_c_lst = []


                for b in range(batch_size):
                    for t in range(seq_len):
                        u = input_data[b * seq_len + t, :].detach()

                        if t == 0:
                            # x1[b, t, :] = self.leaky_rate1 * F.tanh(u @ self.Win1 + self.b_res1)
                            x1[b, t, :] = self.leaky_rate1 * activation_f(u @ self.Win1 + self.b_res1)
                        else:
                            # Add some noise
                            self.b_res1 = torch.randn([self.n_reservoir1, 1])
                            nn.init.sparse_(self.b_res1, sparsity=0.1, std=0.01)
                            self.b_res1 = self.b_res1.squeeze(-1).to(device)

                            # x1[b, t, :] = (1 - self.leaky_rate1) * x1[b, t - 1, :] + self.leaky_rate1 * torch.tanh(
                            #     u @ self.Win1 + x1[b, t - 1, :] @ self.W_res1 + self.b_res1)
                            x1[b, t, :] = (1 - self.leaky_rate1) * x1[b, t - 1, :] + self.leaky_rate1 * activation_f(
                                u @ self.Win1 + x1[b, t - 1, :] @ self.W_res1 + self.b_res1)


                            torch.cuda.empty_cache()


                    pbar.update(1)

                    criterion = nn.MSELoss()

                    # self.W_out = w_out_layer.weight
                for i in range(batch_size):
                    # out = w_out_layer(x1[i, :, 40:80].detach())
                    out = w_out_layer(x1[i, :, :int(self.n_reservoir1 / 3)].detach())
                    # # threshold
                    # target_data[i] = torch.where(target_data[i] > 80, torch.tensor(80).to(device), torch.tensor(0).to(device))
                    # target_data[i] = target_data[i] / 5
                    # target_data[i][target_data[i] == 0] = -100
                    # minus = target_data[i] - out
                    # minus[target_data[i] != 0] = 0
                    # loss = criterion(target_data[i] - torch.abs(out), torch.zeros_like(out).to(device))
                    loss = criterion(out, target_data[i])
                    # loss = self.smape_loss(out, target_data[i]) + self.smape_loss_inverse(out, target_data[i])
                    # print(self.smape_loss(out, target_data[i]))
                    # print(self.smape_loss_inverse(out, target_data[i]))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # self.output(x1[i, :, :int(self.n_reservoir1 / 3)].detach(), target_data[i])
            # self.W_out = w_out_layer.weight

            # y_out = x1[:, :, 40:80] @ self.W_out.t()
            # y_out = torch.relu(x1[:, :, :int(self.n_reservoir1/3)] @ self.W_out.t())
            y_out = x1[:, :, :int(self.n_reservoir1 / 3)] @ self.W_out.t()
            # y_out = x1[:, :, :int(self.n_reservoir1 / 3)] @ self.output.w_out.t().to(device)
            y_out = y_out.view(batch_size, seq_len, self.n_out)

            return y_out, x1.reshape(batch_size, seq_len, -1)
        else:
            # eval state
            input_shift = copy.deepcopy(target_data).reshape(-1, self.input_dim)
            for i in range(len(target_data)):
                for j in range(target_data.size(-1)):
                    # zero_regions = self.detect_non_empty_zero_regions(target_data[i, j])
                    zero_regions = detect_non_empty_zero_regions(target_data[i, :, j])
                    # target_data[i, j] = self.replace_zero_regions_with_sine_wave(time_series=target_data[i, j], zero_regions=zero_regions, frequency=0.5, amplitude=60)
                    target_data[i, :,  j] = replace_zero_regions_with_sine_wave(time_series=target_data[i, :, j], zero_regions=zero_regions, frequency=0.5, amplitude=80)
            target_data = target_data.reshape(-1, self.input_dim)
            x1 = torch.zeros(batch_size * seq_len, self.n_reservoir1)
            if torch.cuda.is_available():
                x1 = x1.to(device)


            update_steps = 400

            y_predicted = torch.zeros_like(input_data.reshape(-1, self.input_dim))
            y_predicted_early = torch.zeros_like(input_data.reshape(-1, self.input_dim))
            y_predicted_late = torch.zeros_like(input_data.reshape(-1, self.input_dim))


            # eps_c = -0.1
            # self.update_c(eps_c=eps_c)

            updata_state = False
            self.update_output = True
            increase_after_1st_stop_state = False
            feedback_time = 0
            want_fb = True


            eps_c_sum = 0
            eps_c_lst = []

            y_out_tensor = torch.zeros_like(target_data)
            y_complicated_out = torch.zeros_like(target_data)
            stop_flag_lst = []
            stop_flag_lst.append(0)
            loss_lst = []
            # replaced_input = torch.zeros_like(input_data)
            replaced_input = input_data.clone().detach()
            after_stop_reference = input_data.clone().detach()
            # feedback_input = input_data.clone().detach()
            feedback_input = torch.zeros_like(input_data)
            # replaced_input = input_data
            complicated_rhythm_target = torch.zeros([batch_size, update_steps, 1])

            # Instantiate the model and set up the optimizer
            skip_model = SkipBeatModel().to(device)
            optimizer = optim.Adam(skip_model.parameters(), lr=0.01)
            criterion = nn.NLLLoss()  # Negative Log Likelihood Loss for classification

            for t in range(int(batch_size * seq_len)):


                u = input_data[t, :]

                if t % seq_len == 0:
                    print(f'sample {t/seq_len} is done!')
                    self.update_output = True

                    feedback_time = 0
                    x1[t, :] = self.leaky_rate1 * activation_f(u @ self.Win1 + self.b_res1)
                    if updata_state:
                        print(eps_c_sum)
                        self.update_c(eps_c=-eps_c_sum)
                        eps_c_sum = 0
                        print(f'sample {t / seq_len} is finished')
                else:
                    # self.b_res1 = torch.randn([self.n_reservoir1, 1])
                    # nn.init.sparse_(self.b_res1, sparsity=0.1, std=0.01)
                    # self.b_res1 = self.b_res1.squeeze(-1).to(device)
                    # self.W_out = torch.nn.Parameter(torch.rand_like(self.W_out)).to(device)
                    if self.update_output:
                        x1[t, :] = (1 - self.leaky_rate1) * x1[t - 1, :] + self.leaky_rate1 * activation_f(u @ self.Win1
                                                                                                      +
                                                                                                     x1[t - 1,
                                                                                                     :] @ self.W_res1 +
                                                                                                     self.b_res1)
                    else:

                        if want_fb:
                            if feedback_time == 0:
                                if y_out_tensor[self.last_peak_prediction, 0] >= 0:
                                    feedback_input[t+32, 0] = y_out_tensor[self.last_peak_prediction-1, 0]
                                feedback_time += 1
                            else:
                                if y_out_tensor[self.last_peak_prediction+feedback_time, 0] >= 0:
                                    if self.last_peak_prediction+feedback_time <= feedback_input.size(0)-100:
                                        feedback_input[t+32, 0] = y_out_tensor[self.last_peak_prediction+feedback_time-1, 0]
                                feedback_time += 1

                        x1[t, :] = (1 - self.leaky_rate1) * x1[t - 1, :] + self.leaky_rate1 * activation_f(replaced_input[t, :] @ self.Win1 +
                                                                                                         feedback_input[t, :] @ self.Win1 +
                                                                                                         x1[t - 1,
                                                                                                         :] @ self.W_res1 +
                                                                                                         self.b_res1)



                    if t % update_steps == 0 and t / update_steps != 0:
                        if increase_after_1st_stop_state:
                            target_sin = copy.deepcopy(after_stop_reference).reshape(batch_size, -1, self.input_dim)
                            for i in range(len(target_sin)):
                                for j in range(target_sin.size(-1)):
                                    zero_regions = detect_non_empty_zero_regions(target_sin[i, :, j])
                                    target_sin[i, :, j] = replace_zero_regions_with_sine_wave(
                                        time_series=target_sin[i, :, j], zero_regions=zero_regions, frequency=0.5,
                                        amplitude=80)
                            target_sin = target_sin.reshape(-1, self.input_dim)
                            self.task_metalearn(input_data[t - update_steps:t],
                                                target_sin[t - update_steps+33:t+33],
                                                update_time_step=update_steps, update_state=True)
                        else:
                            self.task_metalearn(input_data[t-update_steps:t],
                                                target_data[t-update_steps:t],
                                                update_time_step=update_steps, update_state=self.update_output)





                    y_out_tensor[t, :] = x1[t, :int(self.n_reservoir1 / 3)] @ self.W_out.t()
                    if t % seq_len > 2400 and self.update_output:
                        prediction_peaks, _ = find_peaks(y_out_tensor[t - 3*update_steps:t+10, 0].detach().cpu().numpy(), height=5)
                        target_peaks, _ = find_peaks(input_shift[t - 3*update_steps:t+30, 0].detach().cpu().numpy(), height=5)
                        target_peaks = torch.tensor(target_peaks)
                        input_intervals = torch.diff(target_peaks)

                        # Calculate the last peak in prediction and its closest peak in input signal
                        if len(prediction_peaks) > 0 and len(target_peaks) > 0:
                            last_peak_prediction = prediction_peaks[-1]
                            closest_peak_input = torch.argmin(torch.abs(target_peaks - last_peak_prediction))

                            # Compare distances
                            if torch.abs(last_peak_prediction - target_peaks[closest_peak_input]) >= input_intervals[0]-3:
                                self.last_peak_prediction = t - 3 * update_steps + last_peak_prediction
                                self.interval = input_intervals
                                self.update_output = False



            y_out = y_out_tensor.reshape(batch_size, seq_len, self.n_out)


            return y_out, x1.reshape(batch_size, seq_len, -1), torch.tensor(eps_c_lst), y_predicted_early, y_predicted_late, feedback_input, replaced_input
