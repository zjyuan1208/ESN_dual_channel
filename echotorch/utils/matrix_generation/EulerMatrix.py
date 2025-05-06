import torch
import echotorch.utils
import numpy as np
from .MatrixGenerator import MatrixGenerator
from .MatrixFactory import matrix_factory
import torch
import math
import matplotlib.pyplot as plt
import warnings



class EulerMatrixGenerator(MatrixGenerator):
    """
    Generate matrix it normally distributed weights.
    """

    # Constructor
    def __init__(self, **kwargs):
        """
        Constructor
        :param kwargs: Parameters of the generator
        """
        # Set default parameter values
        super(EulerMatrixGenerator, self).__init__(
        )

        # Set parameters
        self._set_parameters(args=kwargs)
    # end __init__

    # Generate the matrix
    def generate(self, size, dtype=torch.float64):
        leaky_rate = 0.03
        CN = 0.4
        n = 40
        c0 = 300
        cr = 200
        c = c0 - cr * torch.rand([n, n])
        dc = -100/n

        ctest = c0 - cr + dc * (n - 1)

        dt = 1.0
        dx = dt / CN * torch.max(torch.max(c)) * math.sqrt(2)
        dy = dx

        k = 0.1
        k = k + 2.0 * torch.rand([n, n])
        k = k / 10
        dk = - 0.01 / n

        Nx1 = np.zeros((n, n))
        Nxprime1 = np.zeros((n, n))
        Ny1 = np.zeros((n, n))
        Nyprime1 = np.zeros((n, n))
        M1 = np.zeros((n, n))
        Mprime1 = np.zeros((n, n))
        Rx1 = np.zeros((n, n))
        Rx2 = np.zeros((n, n))
        Sx1 = np.zeros((n, n))
        Sx2 = np.zeros((n, n))
        Ry1 = np.zeros((n, n))
        Sy1 = np.zeros((n, n))

        for i in range(n):
            M1[i, i] = 1 / dt
            Mprime1[i, i] = 1 / dt
            Nx1[i, i] = 1 / dt + k[i, i] / 2
            Nxprime1[i, i] = 1 / dt - k[i, i] / 2
            Ny1[i, i] = 1 / dt + k[i, i] / 2
            Nyprime1[i, i] = 1 / dt - k[i, i] / 2
            Rx1[i, i] = c[i, i] / dx
            Rx2[i, i] = -c[i, i] / dx
            Sx1[i, i] = -c[i, i] / dx
            Sx2[i, i] = c[i, i] / dx
            Sy1[i, i] = -c[i, i] / dy
            if i > 0:
                Sy1[i, (i - 1) % n] = c[i, i] / dy
            Ry1[i, i] = c[i, i] / dy
            if i < n - 1:
                Ry1[i, i % n + 1] = -c[i, i] / dy

        M = np.zeros((n * n, n * n))
        Mprime = np.zeros((n * n, n * n))
        Nx = np.zeros((n * n, n * n))
        Nxprime = np.zeros((n * n, n * n))
        Ny = np.zeros((n * n, n * n))
        Nyprime = np.zeros((n * n, n * n))
        Rx = np.zeros((n * n, n * n))
        Sx = np.zeros((n * n, n * n))
        Ry = np.zeros((n * n, n * n))
        Sy = np.zeros((n * n, n * n))

        for i in range(1, n + 1):
            if dk != 0:
                for ii in range(1, n + 1):
                    Nx1[ii - 1, ii - 1] = 1 / dt + (k[ii - 1, ii - 1] + dk * (i - 1)) / 2
                    Nxprime1[ii - 1, ii - 1] = 1 / dt - (k[ii - 1, ii - 1] + dk * (i - 1)) / 2
                    Ny1[ii - 1, ii - 1] = 1 / dt + (k[ii - 1, ii - 1] + dk * (i - 1)) / 2
                    Nyprime1[ii - 1, ii - 1] = 1 / dt - (k[ii - 1, ii - 1] + dk * (i - 1)) / 2
            M[(i - 1) * n:i * n, (i - 1) * n:i * n] = M1
            Mprime[(i - 1) * n:i * n, (i - 1) * n:i * n] = Mprime1
            Nx[(i - 1) * n:i * n, (i - 1) * n:i * n] = Nx1
            Nxprime[(i - 1) * n:i * n, (i - 1) * n:i * n] = Nxprime1
            Ny[(i - 1) * n:i * n, (i - 1) * n:i * n] = Ny1
            Nyprime[(i - 1) * n:i * n, (i - 1) * n:i * n] = Nyprime1

            if dc != 0:
                for ii in range(1, n + 1):
                    Rx1[ii - 1, ii - 1] = (c[ii - 1, ii - 1] + dc * (i - 1)) / dx
                    Rx2[ii - 1, ii - 1] = -(c[ii - 1, ii - 1] + dc * (i - 1)) / dx
                    Sx1[ii - 1, ii - 1] = -(c[ii - 1, ii - 1] + dc * (i - 1)) / dx
                    Sx2[ii - 1, ii - 1] = (c[ii - 1, ii - 1] + dc * (i - 1)) / dx
                    Sy1[ii - 1, ii - 1] = -(c[ii - 1, ii - 1] + dc * (i - 1)) / dy
                    if ii > 1:
                        Sy1[ii - 1, (ii - 2) % n] = (c[ii - 1, ii - 1] + dc * (i - 1)) / dy
                    Ry1[ii - 1, ii - 1] = (c[ii - 1, ii - 1] + dc * (i - 1)) / dy
                    if ii < n:
                        Ry1[ii - 1, ii % n] = -(c[ii - 1, ii - 1] + dc * (i - 1)) / dy
            Rx[(i - 1) * n:i * n, (i - 1) * n:i * n] = Rx1
            Sx[(i - 1) * n:i * n, (i - 1) * n:i * n] = Sx1
            if i < n:
                Rx[(i - 1) * n:i * n, i * n:(i + 1) * n] = Rx2
                Sx[i * n:(i + 1) * n, (i - 1) * n:i * n] = Sx2
            Ry[(i - 1) * n:i * n, (i - 1) * n:i * n] = Ry1
            Sy[(i - 1) * n:i * n, (i - 1) * n:i * n] = Sy1

        Minv = np.linalg.inv(M)
        Nxinv = np.linalg.inv(Nx)
        Nyinv = np.linalg.inv(Ny)

        A = np.vstack([
            np.hstack([Minv @ Mprime + Minv @ Rx @ Nxinv @ Sx + Minv @ Ry @ Nyinv @ Sy, Minv @ Rx @ Nxinv @ Nxprime,
                       Minv @ Ry @ Nyinv @ Nyprime]),
            np.hstack([Nxinv @ Sx, Nxinv @ Nxprime, np.zeros((n ** 2, n ** 2))]),
            np.hstack([Nyinv @ Sy, np.zeros((n ** 2, n ** 2)), Nyinv @ Nyprime])
        ])

        A = torch.FloatTensor(A)

        W = (A - (1 - leaky_rate) * torch.eye(n ** 2 * 3)) / leaky_rate


        return W
    # end generate


# Add
matrix_factory.register_generator("euler", EulerMatrixGenerator)

