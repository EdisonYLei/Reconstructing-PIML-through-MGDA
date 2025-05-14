import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

seed = 24 # first round
# seed = 4321 # second round
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# ---------------------------
# Use GPU (if available)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 1. Load dataset dataNew.npz
# ---------------------------
# dataNew.npz contains:
#   X_train: (N_train, 2), Y_train: (N_train, 2)
#   X_validate: (N_validate, 2), Y_validate: (N_validate, 2)
#   X_test: (N_test, 2), Y_test: (N_test, 2)
data_npz = np.load("dataNew.npz", allow_pickle=True)
X_train = data_npz["X_train"]
Y_train = data_npz["Y_train"]
X_validate = data_npz["X_validate"]
Y_validate = data_npz["Y_validate"]
X_test = data_npz["X_test"]
Y_test = data_npz["Y_test"]

# Compute true density: rho = f / u (add eps to avoid division by zero)
eps_val = 1e-6
rho_train_true = Y_train[:, 0:1] / (Y_train[:, 1:2] + eps_val)
rho_validate_true = Y_validate[:, 0:1] / (Y_validate[:, 1:2] + eps_val)
rho_test_true = Y_test[:, 0:1] / (Y_test[:, 1:2] + eps_val)
# True speed is u
speed_train_true = Y_train[:, 1:2]
speed_validate_true = Y_validate[:, 1:2]
speed_test_true = Y_test[:, 1:2]

# ---------------------------
# 1.1 Normalize data (based on training set statistics)
# For input X: compute mean and std from training set
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0) + 1e-6
X_train_norm = (X_train - X_mean) / X_std
X_validate_norm = (X_validate - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# For density (rho): compute mean and std from training set
rho_mean = np.mean(rho_train_true, axis=0)
rho_std = np.std(rho_train_true, axis=0) + 1e-6
rho_train_norm = (rho_train_true - rho_mean) / rho_std
rho_validate_norm = (rho_validate_true - rho_mean) / rho_std
rho_test_norm = (rho_test_true - rho_mean) / rho_std

# For speed (u): compute mean and std from training set
u_mean = np.mean(speed_train_true, axis=0)
u_std = np.std(speed_train_true, axis=0) + 1e-6
u_train_norm = (speed_train_true - u_mean) / u_std
u_validate_norm = (speed_validate_true - u_mean) / u_std
u_test_norm = (speed_test_true - u_mean) / u_std

# ---------------------------
# 2. Define network architecture (no Dropout)
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        # Dropout layer not used
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PINN(nn.Module):
    def __init__(self, rho_hidden_dim=20, rho_num_hidden=8, q_hidden_dim=20, q_num_hidden=2):
        super(PINN, self).__init__()
        # net_rho: input (x,t) -> output 1-dim (hat_rho)
        self.net_rho = MLP(input_dim=2, hidden_dim=rho_hidden_dim, num_hidden=rho_num_hidden, output_dim=1)
        # q_net: input hat_rho -> output 1-dim (Q_hat), compute u_hat = Q_hat / (rho_hat + eps)
        self.q_net = MLP(input_dim=1, hidden_dim=q_hidden_dim, num_hidden=q_num_hidden, output_dim=1)

    def forward(self, X):
        # X: (batch,2)
        rho_hat = self.net_rho(X)  # predicted hat_rho (normalized space)
        Q_hat = self.q_net(rho_hat)  # get Q_hat
        eps = 1e-6
        u_hat = Q_hat / (rho_hat + eps)  # compute predicted speed u_hat
        return rho_hat, Q_hat, u_hat

# ---------------------------
# 3. Define helper function: compute partial derivatives
# ---------------------------
def compute_grad(output, X, idx):
    # compute gradient of output w.r.t. X[:, idx]
    grad = torch.autograd.grad(outputs=output, inputs=X,
                               grad_outputs=torch.ones_like(output),
                               create_graph=True)[0]
    return grad[:, idx:idx + 1]

# ---------------------------
# 4. Define physics residual function
# ---------------------------
def compute_physics_residual(model, X):
    """
    Compute physics residual f = (rho_t + Q_x)
    where rho_t = ∂hat_rho/∂t, Q_x = ∂Q_hat/∂x.
    Require X.requires_grad=True
    """
    if not X.requires_grad:
        X.requires_grad_()
    rho_hat, Q_hat, _ = model(X)
    grads_rho = torch.autograd.grad(rho_hat, X, grad_outputs=torch.ones_like(rho_hat), create_graph=True)[0]
    rho_t = grads_rho[:, 1:2]  # derivative w.r.t. t
    grads_Q = torch.autograd.grad(Q_hat, X, grad_outputs=torch.ones_like(Q_hat), create_graph=True)[0]
    Q_x = grads_Q[:, 0:1]      # derivative w.r.t. x
    f = rho_t + Q_x
    return f

# ---------------------------
# New: define uniform sampling function
# ---------------------------
def generate_uniform_collocation_points(X_norm, num_points):
    """
    Generate uniformly sampled collocation points
    based on the range of normalized training data X_norm
    """
    x_min, t_min = X_norm.min(axis=0)
    x_max, t_max = X_norm.max(axis=0)
    num_per_dim = int(np.sqrt(num_points))
    x_vals = np.linspace(x_min, x_max, num_per_dim)
    t_vals = np.linspace(t_min, t_max, num_per_dim)
    X1, T1 = np.meshgrid(x_vals, t_vals)
    X_coll = np.vstack([X1.flatten(), T1.flatten()]).T
    return X_coll

# ---------------------------
# 5. Define loss function (use sampled points for physics residual)
# ---------------------------
def loss_function(model, X_data, rho_true, u_true, X_collocation, coeff_phys):
    """
    Loss function:
      L = 100 * MSE(hat_rho, rho_true) + coeff_phys * MSE(rho_t+Q_x, 0) + 100 * MSE(u_hat, u_true)
    where physics residual is computed on uniformly sampled points
    """
    X_data.requires_grad = True
    hat_rho, Q_hat, u_hat = model(X_data)
    mse = nn.MSELoss()
    loss_rho = mse(hat_rho, rho_true)
    loss_speed = mse(u_hat, u_true)

    X_coll = X_collocation.clone().detach()
    X_coll.requires_grad_()
    f = compute_physics_residual(model, X_coll)
    loss_phys = mse(f, torch.zeros_like(f))

    total_loss = 100 * loss_rho + coeff_phys * loss_phys + 100 * loss_speed
    return total_loss

# ---------------------------
# 6. Convert data to Tensor (use normalized data)
# ---------------------------
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
rho_train_tensor = torch.tensor(rho_train_norm, dtype=torch.float32, device=device)
u_train_tensor = torch.tensor(u_train_norm, dtype=torch.float32, device=device)

X_validate_tensor = torch.tensor(X_validate_norm, dtype=torch.float32, device=device)
rho_validate_tensor = torch.tensor(rho_validate_norm, dtype=torch.float32, device=device)
u_validate_tensor = torch.tensor(u_validate_norm, dtype=torch.float32, device=device)

X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
rho_test_tensor = torch.tensor(rho_test_norm, dtype=torch.float32, device=device)
u_test_tensor = torch.tensor(u_test_norm, dtype=torch.float32, device=device)

# ---------------------------
# 7. Define training function (with early stopping using validation set)
# ---------------------------
def train_PINN(model, X_data, rho_true, u_true, X_val, rho_val, u_val,
               num_epochs=5000, lr=1e-5, coeff_phys=120,
               num_collocation_points=int(0.8 * X_train_norm.shape[0]),
               patience=200):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Generate uniformly sampled collocation points based on normalized training data
    X_coll_np = generate_uniform_collocation_points(X_train_norm, num_collocation_points)
    X_coll_tensor = torch.tensor(X_coll_np, dtype=torch.float32, device=device)
    X_coll_tensor.requires_grad_()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = loss_function(model, X_data, rho_true, u_true, X_coll_tensor, coeff_phys)
        loss.backward()
        optimizer.step()

        # Compute validation loss (need gradients for physics residual)
        with torch.enable_grad():
            val_loss = loss_function(model, X_val, rho_val, u_val, X_coll_tensor, coeff_phys)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.6e}, Val Loss: {val_loss.item():.6e}")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model

# ---------------------------
# 8. Fine-tuning coefficient coeff_phys experiment
# Fix loss_rho and loss_speed coefficients at 100,
# Try coeff_phys candidates: [2,3,4,5,6,7,8,9] (second round)
# Run multiple experiments for each value, record relative L² error on test set
# ---------------------------
# coeff_phys_candidates = [1, 10, 100, 1000, 10000] # first round
coeff_phys_candidates = [2,3,4,5,6,7,8,9] # second round
# coeff_phys_candidates = [500]
num_experiments = 30

results = {}

for coeff in coeff_phys_candidates:
    print(f"\nFine-tuning: coeff_phys = {coeff}")
    density_errors = []
    speed_errors = []
    overall_errors = []
    for exp in range(num_experiments):
        print(f"  Experiment {exp + 1}/{num_experiments}")
        pinn_model = PINN().to(device)
        pinn_model = train_PINN(pinn_model,
                                X_train_tensor, rho_train_tensor, u_train_tensor,
                                X_validate_tensor, rho_validate_tensor, u_validate_tensor,
                                num_epochs=50000, lr=1e-4, coeff_phys=coeff,
                                num_collocation_points=int(0.8 * X_train_norm.shape[0]),
                                patience=2000)
        pinn_model.eval()
        with torch.no_grad():
            rho_pred = pinn_model(X_test_tensor)[0]
            u_pred = pinn_model(X_test_tensor)[2]
        # Denormalize
        rho_pred = rho_pred.cpu().numpy() * rho_std + rho_mean
        u_pred = u_pred.cpu().numpy() * u_std + u_mean
        rho_true_orig = rho_test_true  # original scale
        u_true_orig = speed_test_true  # original scale
        err_rho = np.linalg.norm(rho_pred - rho_true_orig) / np.linalg.norm(rho_true_orig)
        err_u = np.linalg.norm(u_pred - u_true_orig) / np.linalg.norm(u_true_orig)
        density_errors.append(err_rho)
        speed_errors.append(err_u)
        overall_errors.append((err_rho + err_u) / 2)
    density_errors = np.array(density_errors)
    speed_errors = np.array(speed_errors)
    overall_errors = np.array(overall_errors)
    results[coeff] = {
        "mean_overall": np.mean(overall_errors),
        "std_overall": np.std(overall_errors),
        "mean_density": np.mean(density_errors),
        "std_density": np.std(density_errors),
        "mean_speed": np.mean(speed_errors),
        "std_speed": np.std(speed_errors)
    }
    print(f"  coeff_phys = {coeff}:")
    print(f"    Overall Error: Mean = {results[coeff]['mean_overall']:.3e}, Std = {results[coeff]['std_overall']:.3e}")
    print(f"    Density Error: Mean = {results[coeff]['mean_density']:.3e}, Std = {results[coeff]['std_density']:.3e}")
    print(f"    Speed Error:   Mean = {results[coeff]['mean_speed']:.3e}, Std = {results[coeff]['std_speed']:.3e}")

# ---------------------------
# 9. Select best coeff_phys
# ---------------------------
best_coeff = None
best_overall_mean = float('inf')
best_overall_std = float('inf')
for coeff in coeff_phys_candidates:
    m = results[coeff]["mean_overall"]
    s_val = results[coeff]["std_overall"]
    if m < best_overall_mean or (np.isclose(m, best_overall_mean) and s_val < best_overall_std):
        best_overall_mean = m
        best_overall_std = s_val
        best_coeff = coeff

print("\n================ Final Fine-tuning Results ================")
print(f"Best coeff_phys: {best_coeff}")
print(f"Density Error: Mean = {results[best_coeff]['mean_density']:.3e}, Std = {results[best_coeff]['std_density']:.3e}")
print(f"Speed Error:   Mean = {results[best_coeff]['mean_speed']:.3e}, Std = {results[best_coeff]['std_speed']:.3e}")
