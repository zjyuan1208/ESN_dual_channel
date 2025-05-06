import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class EigenMatrixGenerator(nn.Module):
    """
    Generates weight matrices using an eigendecomposition-based approach.
    This is a trainable matrix generator with learnable parameters.
    """

    def __init__(self, leaky_rate=0.03, n=40, c=None):
        super(EigenMatrixGenerator, self).__init__()
        self.leaky_rate = leaky_rate
        self.n = n
        grid_size = n * n
        self.grid_size = grid_size

        # Constants and parameters
        self.CN = 0.1
        self.c0 = 300
        self.crr = 0.8
        self.dc = -250 / n
        self.dt = 1.0

        # Initialize c as a trainable parameter
        if c is None:
            c_init = self.c0 * torch.ones([n, n], device=device)
            for i in range(n):
                c_init[:, i] = c_init[:, i] + self.dc * (i - 1)
            c_init = c_init - self.crr * torch.rand_like(c_init) * c_init
            self.c = nn.Parameter(c_init, requires_grad=True)
        else:
            self.c = nn.Parameter(c.clone(), requires_grad=True)

        # Initialize dx as a trainable matrix parameter
        dx_init = self.dt / self.CN * torch.max(torch.max(c_init)) * math.sqrt(2) * torch.ones([n, n], device=device)
        self.dx = nn.Parameter(dx_init, requires_grad=True)
        self.dy = nn.Parameter(dx_init.clone(), requires_grad=True)

        # Initialize damping parameters
        self.k = nn.Parameter(torch.zeros([n, n], device=device), requires_grad=True)
        self.kp = nn.Parameter(0.0001 * torch.ones([n, n], device=device), requires_grad=True)

        # Additional learnable parameters
        self.dkx = nn.Parameter(torch.zeros_like(self.c), requires_grad=True)
        self.dky = nn.Parameter(torch.zeros_like(self.c), requires_grad=True)

        # Print parameter information
        print("EigenMatrixGenerator initialized with trainable parameters:")
        for name, param in self.named_parameters():
            print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")

        # Initialize cached matrices at the correct size
        self._init_caches()

    def _init_caches(self):
        """Initialize cache buffers with the correct size"""
        n = self.n
        grid_size = n * n
        state_size = 3 * grid_size

        # Initialize empty buffers with correct shape
        self.register_buffer('cached_A', torch.zeros(state_size, state_size, device=device))
        self.register_buffer('cached_W', torch.zeros(state_size, state_size, device=device))
        self.register_buffer('last_c', torch.zeros(n * n, device=device))
        self.register_buffer('last_k', torch.zeros(n * n, device=device))
        self.register_buffer('last_dx', torch.zeros(n * n, device=device))
        self.register_buffer('last_dy', torch.zeros(n * n, device=device))

    def get_positive_parameters(self):
        """Get positive versions of parameters with appropriate constraints."""
        # For wave speed, use softplus for positivity
        c_pos = F.softplus(self.c)

        # For damping parameters, ensure positivity
        kp_pos = F.softplus(self.kp)
        k_pos = F.softplus(self.k)

        # For dx and dy, ensure positivity
        dx_pos = F.softplus(self.dx)
        dy_pos = F.softplus(self.dy)

        # Simplified processing to match original implementation
        return c_pos.flatten(), k_pos.flatten(), k_pos.flatten(), kp_pos.flatten(), dx_pos.flatten(), dy_pos.flatten()

    def construct_eigendecomposition(self, c, k_x, k_y, kp, dx, dy):
        """
        Constructs the eigendecomposition matrices for the system.

        Args:
            c: Wave speed parameter (flattened)
            k_x: X-direction damping parameter (flattened)
            k_y: Y-direction damping parameter (flattened)
            kp: Pressure damping parameter (flattened)
            dx: X-direction spatial step (flattened)
            dy: Y-direction spatial step (flattened)

        Returns:
            Tuple of (Lambda, P, P_inv) for eigendecomposition
        """
        n = self.n
        grid_size = n * n
        state_size = 3 * grid_size  # [p, ox, oy]

        # Calculate spatial frequencies based on grid spacing
        # Using fixed values instead of dynamic calculations to maintain performance
        xi_magnitude = torch.tensor(np.pi / (2 * 1.0), device=c.device)

        # Calculate eigenvalue components for simplified model
        lambda1 = 1.0 / (1.0 + self.dt * k_x)
        real_part = 0.5 * (1.0 / (1.0 + self.dt * kp) + 1.0 / (1.0 + self.dt * k_x))
        imag_part = c * self.dt * xi_magnitude / torch.sqrt((1.0 + self.dt * kp) * (1.0 + self.dt * k_x))

        # Complex conjugate pair for oscillatory modes
        lambda2 = torch.complex(real_part, imag_part)
        lambda3 = torch.complex(real_part, -imag_part)

        # Create Lambda vector (diagonal values)
        Lambda = torch.zeros(state_size, dtype=torch.complex64, device=c.device)
        Lambda[:grid_size] = lambda1
        Lambda[grid_size:2 * grid_size] = lambda2
        Lambda[2 * grid_size:] = lambda3

        # Calculate eigenvectors
        beta = self.dt / (1.0 + self.dt * k_x)
        alpha = lambda2 - lambda1

        # Avoid division by 0
        alpha = alpha + 1e-6

        v2_1 = torch.ones(grid_size, device=c.device, dtype=torch.complex64)
        v2_2 = beta * 1j * xi_magnitude / alpha
        v2_3 = beta * 1j * xi_magnitude / alpha

        v3_1 = torch.ones(grid_size, device=c.device, dtype=torch.complex64)
        v3_2 = -v2_2.conj()
        v3_3 = -v2_3.conj()

        # Construct full modal eigenvector matrix P (state_size x state_size)
        P = torch.zeros((state_size, state_size), dtype=torch.complex64, device=c.device)

        # Assign basis vectors for 3 modal groups
        P[0:grid_size, 0:grid_size] = torch.diag(
            torch.ones(grid_size, dtype=torch.complex64, device=c.device))  # λ1 mode
        P[0:grid_size, grid_size:2 * grid_size] = torch.diag(v2_1)  # λ2 mode
        P[0:grid_size, 2 * grid_size:] = torch.diag(v3_1)  # λ3 mode

        P[grid_size:2 * grid_size, grid_size:2 * grid_size] = torch.diag(v2_2)
        P[2 * grid_size:, grid_size:2 * grid_size] = torch.diag(v2_3)

        P[grid_size:2 * grid_size, 2 * grid_size:] = torch.diag(v3_2)
        P[2 * grid_size:, 2 * grid_size:] = torch.diag(v3_3)

        # Orthogonal or Hermitian inverse
        P_inv = P.conj().transpose(0, 1)

        return Lambda, P, P_inv

    def construct_matrices_from_eigendecomposition(self, Lambda, P, P_inv):
        """
        Construct the A and W matrices from eigendecomposition.

        Args:
            Lambda: Diagonal matrix of eigenvalues
            P: Matrix of eigenvectors
            P_inv: Inverse of P

        Returns:
            Tuple of (A, W) matrices for the system
        """
        n = self.n
        grid_size = n * n
        state_size = 3 * grid_size

        # Construct A matrix from eigendecomposition
        # A = P @ diag(Lambda) @ P_inv
        Lambda_diag = torch.diag_embed(Lambda)
        A_complex = P @ Lambda_diag @ P_inv

        # Take real part for the final matrix
        A = A_complex.real

        # Compute W from A using the leaky rate formula
        W = (A - (1 - self.leaky_rate) * torch.eye(state_size, device=A.device)) / self.leaky_rate

        return A, W

    def generate(self, n=None):
        """
        Generate the weight matrices based on current parameters.
        Returns A, W, c, k matrices.
        """
        if n is None:
            n = self.n

        # Get constrained positive parameters
        c_pos, k_x, k_y, kp_pos, dx_pos, dy_pos = self.get_positive_parameters()

        # Check if we need to recompute matrices (parameters changed)
        params_changed = not (torch.allclose(c_pos, self.last_c) and
                              torch.allclose(k_x, self.last_k) and
                              torch.allclose(dx_pos, self.last_dx) and
                              torch.allclose(dy_pos, self.last_dy))

        if params_changed:
            # Construct eigendecomposition
            Lambda, P, P_inv = self.construct_eigendecomposition(c_pos, k_x, k_y, kp_pos, dx_pos, dy_pos)

            # Build A and W matrices from eigendecomposition
            A, W = self.construct_matrices_from_eigendecomposition(Lambda, P, P_inv)

            # Update cached values
            self.cached_A = A
            self.cached_W = W
            self.last_c = c_pos.clone()
            self.last_k = k_x.clone()
            self.last_dx = dx_pos.clone()
            self.last_dy = dy_pos.clone()
        else:
            # Use cached values
            A = self.cached_A
            W = self.cached_W

        return A, W, c_pos.reshape(n, n), self.k


class SimpleESN(nn.Module):
    """A simplified Echo State Network with a trainable reservoir"""

    def __init__(self, input_dim, reservoir_size, output_dim,
                 leaky_rate=0.03, n_grid=40):
        super(SimpleESN, self).__init__()

        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.output_dim = output_dim
        self.leaky_rate = leaky_rate

        # Grid size for the EigenMatrixGenerator
        # reservoir_size must be divisible by 3 (for p, ox, oy state components)
        self.n_grid = n_grid

        # Create trainable weight generator
        self.weight_generator = EigenMatrixGenerator(
            leaky_rate=leaky_rate,
            n=n_grid,
            c=None
        )

        # Generate initial reservoir weights
        self.A, self.W_res, self.c, self.k = self.weight_generator.generate(n=n_grid)

        # Input weights (input to reservoir)
        self.Win = torch.randn(input_dim, reservoir_size)
        nn.init.xavier_uniform_(self.Win)

        # For focused input, you can make most of the input weights zero
        # Example: self.Win[:, 40:] = 0  # Only connect to first 40 neurons

        # Bias term for reservoir
        self.b_res = torch.zeros(reservoir_size)

        # Output weights (reservoir to output)
        self.W_out = torch.randn(output_dim, reservoir_size)
        nn.init.xavier_uniform_(self.W_out)

        # Move everything to device
        if torch.cuda.is_available():
            self.Win = self.Win.to(device)
            self.W_res = self.W_res.to(device)
            self.b_res = self.b_res.to(device)
            self.W_out = self.W_out.to(device)

    def train_readout(self, X, Y, lr=0.01, num_epochs=20):
        """Train the output layer using simple gradient descent

        Args:
            X: Input states from reservoir (batch_size, seq_len, reservoir_states)
            Y: Target outputs (batch_size, seq_len, output_dim)
            lr: Learning rate
            num_epochs: Number of training epochs
        """
        batch_size = X.size(0)
        seq_len = X.size(1)
        actual_state_size = X.size(2)  # Get the actual state size from X

        # Print dimensions for debugging
        print(f"Training readout layer:")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"W_out current shape: {self.W_out.shape}")

        # Resize W_out to match the actual state size
        if self.W_out.shape[1] != actual_state_size:
            print(f"Resizing W_out from {self.W_out.shape} to [{self.output_dim}, {actual_state_size}]")
            new_W_out = torch.zeros(self.output_dim, actual_state_size, device=device)

            # Copy existing weights where possible
            min_cols = min(self.W_out.shape[1], actual_state_size)
            new_W_out[:, :min_cols] = self.W_out[:, :min_cols]

            # Initialize new weights randomly
            if min_cols < actual_state_size:
                nn.init.xavier_uniform_(new_W_out[:, min_cols:])

            self.W_out = new_W_out

        # Create a linear layer with the output weights
        readout = nn.Linear(actual_state_size, self.output_dim, bias=False)
        readout.weight = nn.Parameter(self.W_out)

        if torch.cuda.is_available():
            readout = readout.to(device)

        # Setup optimizer
        optimizer = torch.optim.SGD(readout.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train the output weights
        for epoch in range(num_epochs):
            total_loss = 0

            # For each batch, we'll detach X to avoid backward through the reservoir
            X_detached = X.detach()

            for i in range(batch_size):
                # Forward pass with detached inputs
                out = readout(X_detached[i])
                loss = criterion(out, Y[i])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()  # No need for retain_graph since we're using detached inputs
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / batch_size:.6f}")

        # Update the output weights
        self.W_out = readout.weight.detach()

    def update_reservoir_weights(self):
        """Update the reservoir weights by regenerating them with the trainable generator"""
        # This allows to update the reservoir based on the current state of the generator
        self.A, self.W_res, self.c, self.k = self.weight_generator.generate(n=self.n_grid)

    def scan_operation(self, Lambda, F_modal):
        """
        Associative scan-style implementation of:
            y[t] = Lambda * y[t-1] + F_modal[t]

        Uses a tree-reduction style binary operation similar to JAX's associative_scan.

        Args:
            Lambda: The transition matrix [reservoir_size, reservoir_size]
            F_modal: The input forcing term sequence [seq_len, reservoir_size]

        Returns:
            The resulting state sequence [seq_len, reservoir_size]
        """
        seq_len, state_size = F_modal.shape
        device = F_modal.device

        # Ensure Lambda is 2D - if it's 3D, we need to reshape appropriately
        if Lambda.dim() == 3:
            # Assume it's [1, state_size, state_size] and squeeze the first dimension
            Lambda = Lambda.squeeze(0)

        # For vector-based scan operation, we need a 1D transition factor
        # Let's create a diagonal Lambda for scan operation
        Lambda_diag = torch.diag(Lambda)  # Extract diagonal elements

        # Broadcast Lambda across time for the scan
        Lambda_seq = Lambda_diag.unsqueeze(0).expand(seq_len, -1)

        # Initialize tuples (A, b)
        A_seq = Lambda_seq.clone()
        b_seq = F_modal.clone()

        A_stack = [A_seq]
        b_stack = [b_seq]

        while A_stack[0].size(0) > 1:
            A_prev, b_prev = A_stack[0], b_stack[0]
            L = A_prev.size(0)
            even_len = (L // 2) * 2

            A1 = A_prev[0:even_len:2]
            A2 = A_prev[1:even_len:2]
            b1 = b_prev[0:even_len:2]
            b2 = b_prev[1:even_len:2]

            A_comb = A2 * A1
            b_comb = A2 * b1 + b2

            if L % 2 == 1:
                A_comb = torch.cat([A_comb, A_prev[-1:]], dim=0)
                b_comb = torch.cat([b_comb, b_prev[-1:]], dim=0)

            A_stack.insert(0, A_comb)
            b_stack.insert(0, b_comb)

        # Reconstruct full modal state sequence
        modal_states = b_stack[0]
        for i in range(1, len(A_stack)):
            A, b = A_stack[i], b_stack[i]
            out = torch.zeros_like(b)
            out[0::2] = modal_states
            out[1::2] = A[1::2] * modal_states[0:A.size(0) // 2] + b[1::2]
            modal_states = out

        return modal_states

    def forward(self, input_data):
        """Process input data through the ESN using efficient step-by-step computation

        Args:
            input_data: Input sequences (batch_size, seq_len, input_dim)

        Returns:
            Output predictions and reservoir states
        """
        batch_size = input_data.size(0)
        seq_len = input_data.size(1)

        # Regenerate reservoir weights if the matrix generator params have changed
        # Using the trainable EigenMatrixGenerator
        self.update_reservoir_weights()

        # Print shapes for debugging
        print(f"W_res shape: {self.W_res.shape}")
        print(f"Win shape: {self.Win.shape}")
        print(f"b_res shape: {self.b_res.shape}")
        print(f"W_out shape: {self.W_out.shape}")
        print(f"Reservoir size: {self.reservoir_size}")

        # Get the actual size from W_res, which is n^2*3 from EigenMatrixGenerator
        actual_size = self.W_res.shape[0]

        # Ensure bias has the correct size
        if self.b_res.shape[0] != actual_size:
            # Create a new bias with the correct size
            new_b_res = torch.zeros(actual_size, device=device)
            # Copy values from the old bias where possible
            min_size = min(self.b_res.shape[0], actual_size)
            new_b_res[:min_size] = self.b_res[:min_size]
            self.b_res = new_b_res

        # Ensure Win has the correct output size
        if self.Win.shape[1] != actual_size:
            # Create a new input weight matrix
            new_Win = torch.zeros(self.input_dim, actual_size, device=device)
            # Copy values from the old Win where possible
            min_cols = min(self.Win.shape[1], actual_size)
            new_Win[:, :min_cols] = self.Win[:, :min_cols]
            # Initialize remaining weights if needed
            if min_cols < actual_size:
                nn.init.xavier_uniform_(new_Win[:, min_cols:])
            self.Win = new_Win

        # Initialize outputs and states tensor
        outputs = torch.zeros(batch_size, seq_len, self.output_dim, device=device)
        states = torch.zeros(batch_size, seq_len, actual_size, device=device)

        # Process each batch
        for b in range(batch_size):
            # Prepare the input
            batch_input = input_data[b]  # [seq_len, input_dim]

            # Initial state
            x_prev = torch.zeros(actual_size, device=device)

            for t in range(seq_len):
                # Compute the input projection
                input_projection = batch_input[t] @ self.Win  # [actual_size]

                # Compute current state: x_t = (1-alpha)*x_{t-1} + alpha*(Wx_{t-1} + Win*u_t + b)
                # Without tanh activation as requested
                state_update = input_projection + self.b_res
                state_recurrent = x_prev @ self.W_res

                x_t = (1 - self.leaky_rate) * x_prev + self.leaky_rate * (state_update + state_recurrent)

                # Store the state
                states[b, t] = x_t
                x_prev = x_t

            # For output weights, we need to ensure W_out has the right input size
            if self.W_out.shape[1] != actual_size:
                # Create a temporary weight matrix for this forward pass
                temp_W_out = torch.zeros(self.output_dim, actual_size, device=device)
                # Copy values from the original W_out where possible
                min_cols = min(self.W_out.shape[1], actual_size)
                temp_W_out[:, :min_cols] = self.W_out[:, :min_cols]
                # Compute outputs
                outputs[b] = states[b] @ temp_W_out.t()
            else:
                # Compute outputs with the original weights
                outputs[b] = states[b] @ self.W_out.t()

        return outputs, states

    def update_wave_parameters(self, eps_c=0.0):
        """
        Update the wave speed parameters of the matrix generator

        Args:
            eps_c: Epsilon value to adjust wave speed (can be positive or negative)
        """
        # This allows targeted adjustments to the wave parameters
        # Similar to the original EulerMatrixGenerator's update_c method
        # We modify the wave speed parameter directly
        self.weight_generator.c.data = self.weight_generator.c.data * (1 + eps_c)

        # After changing parameters, update the reservoir weights
        self.update_reservoir_weights()

    def train_matrix_generator(self, input_data, target_data, lr=0.001, epochs=10):
        """
        Train the matrix generator parameters using gradient descent

        Args:
            input_data: Input sequences (batch_size, seq_len, input_dim)
            target_data: Target outputs (batch_size, seq_len, output_dim)
            lr: Learning rate for matrix generator parameters
            epochs: Number of training epochs
        """
        print("Starting matrix generator training...")
        print(
            f"Weight generator parameters require grad: {[p.requires_grad for p in self.weight_generator.parameters()]}")

        # Verify we have trainable parameters
        trainable_params = [p for p in self.weight_generator.parameters() if p.requires_grad]
        if not trainable_params:
            print("WARNING: No trainable parameters in the weight generator!")
            return

        # Create optimizer for the matrix generator parameters
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        criterion = nn.MSELoss()

        # Ensure inputs and targets are on the right device
        inputs = input_data.to(device)
        targets = target_data.to(device)

        for epoch in range(epochs):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = self.forward(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")

            # Check if loss is valid
            if torch.isnan(loss):
                print("NaN loss detected, skipping backward pass")
                continue

            try:
                # Manually check if the loss is connected to any parameters that require gradients
                params_requiring_grad = set()
                if hasattr(loss, 'grad_fn'):
                    def traverse_graph(fn):
                        if not hasattr(fn, 'next_functions'):
                            return
                        for next_fn, _ in fn.next_functions:
                            if next_fn is not None:
                                if hasattr(next_fn, 'variable'):
                                    var = next_fn.variable
                                    if var.requires_grad:
                                        params_requiring_grad.add(var)
                                traverse_graph(next_fn)

                    traverse_graph(loss.grad_fn)

                if not params_requiring_grad:
                    print("WARNING: Loss is not connected to any parameters requiring gradients")
                    # Try to update weights directly instead
                    with torch.no_grad():
                        for param in trainable_params:
                            # Apply a small random perturbation
                            param.add_(torch.randn_like(param) * lr * 0.01)
                    continue

                # Backward pass
                loss.backward()

                # Check for valid gradients
                valid_grads = True
                for name, param in self.weight_generator.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"Parameter {name} has no gradient")
                            valid_grads = False
                        elif torch.isnan(param.grad).any():
                            print(f"Parameter {name} has NaN gradients")
                            valid_grads = False

                if valid_grads:
                    optimizer.step()

                # Update reservoir weights
                self.update_reservoir_weights()

            except Exception as e:
                print(f"Error during backward/optimization: {str(e)}")

        # Final update of reservoir weights
        self.update_reservoir_weights()

    def evaluate(self, input_data, target_data=None):
        """Run the network in evaluation mode"""
        with torch.no_grad():
            return self.forward(input_data)


# Example usage
if __name__ == "__main__":
    # Set random seed for reproducibility
    setup_seed(1234)

    # Create a trainable ESN
    input_dim = 1
    reservoir_size = 120  # Must be divisible by 3 for the eigen-based reservoir
    output_dim = 1
    n_grid = 20  # so reservoir_size = 3 * n_grid * n_grid
    esn = SimpleESN(input_dim, reservoir_size, output_dim, n_grid=n_grid)

    # Generate some dummy data
    batch_size = 2
    seq_len = 100
    X = torch.randn(batch_size, seq_len, input_dim)
    Y = torch.sin(torch.linspace(0, 4 * np.pi, seq_len)).unsqueeze(1).repeat(batch_size, 1, 1)

    # First train just the readout weights
    outputs, states = esn(X)
    esn.train_readout(states, Y)

    # Then optionally train the matrix generator parameters
    esn.train_matrix_generator(X, Y, lr=0.001, epochs=5)

    # Run in evaluation mode
    predictions, _ = esn.evaluate(X)

    print(f"Prediction shape: {predictions.shape}")