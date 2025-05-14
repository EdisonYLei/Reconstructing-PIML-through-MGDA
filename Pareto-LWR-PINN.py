import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from min_norm_solvers_numpy import MinNormSolverNumpy, gradient_normalizers

# ---------------------------
# 1. Use GPU (if available)
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set random seeds for reproducibility
import random
seed = 4321
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# 2. Load new dataset dataNew.npz
# ---------------------------
data_npz = np.load("dataNew.npz", allow_pickle=True)
X_train = data_npz["X_train"]
Y_train = data_npz["Y_train"]
X_validate = data_npz["X_validate"]
Y_validate = data_npz["Y_validate"]
X_test = data_npz["X_test"]
Y_test = data_npz["Y_test"]

# Compute true density: ρ = f / u (add eps to avoid division by zero)
eps_val = 1e-6
rho_train_true = Y_train[:, 0:1]/(Y_train[:, 1:2] + eps_val)
rho_validate_true = Y_validate[:, 0:1]/(Y_validate[:, 1:2] + eps_val)
rho_test_true = Y_test[:, 0:1]/(Y_test[:, 1:2] + eps_val)
# True speed is u
speed_train_true = Y_train[:, 1:2]
speed_validate_true = Y_validate[:, 1:2]
speed_test_true = Y_test[:, 1:2]

# ---------------------------
# 3. Normalize data (based on training set statistics)
# ---------------------------
# Normalize input X
X_mean = np.mean(X_train, axis=0)
X_std  = np.std(X_train, axis=0) + 1e-6
X_train_norm = (X_train - X_mean) / X_std
X_validate_norm = (X_validate - X_mean) / X_std
X_test_norm  = (X_test - X_mean) / X_std

# Normalize density (ρ) target
rho_mean = np.mean(rho_train_true, axis=0)
rho_std  = np.std(rho_train_true, axis=0) + 1e-6
rho_train_norm = (rho_train_true - rho_mean) / rho_std
rho_validate_norm = (rho_validate_true - rho_mean) / rho_std
rho_test_norm  = (rho_test_true - rho_mean) / rho_std

# Normalize speed target
u_mean = np.mean(speed_train_true, axis=0)
u_std  = np.std(speed_train_true, axis=0) + 1e-6
u_train_norm = (speed_train_true - u_mean) / u_std
u_validate_norm = (speed_validate_true - u_mean) / u_std
u_test_norm  = (speed_test_true - u_mean) / u_std

# ---------------------------
# 4. Generate uniformly sampled collocation points for physics residual
# ---------------------------
def generate_uniform_collocation_points(X_norm, num_points):
    x_min, t_min = X_norm.min(axis=0)
    x_max, t_max = X_norm.max(axis=0)
    num_per_dim = int(np.sqrt(num_points))
    x_vals = np.linspace(x_min, x_max, num_per_dim)
    t_vals = np.linspace(t_min, t_max, num_per_dim)
    X1, T1 = np.meshgrid(x_vals, t_vals)
    return np.vstack([X1.flatten(), T1.flatten()]).T

num_collocation = int(0.8 * X_train.shape[0])
X_coll_np = generate_uniform_collocation_points(X_train_norm, num_collocation)
X_coll_tensor = torch.tensor(X_coll_np, dtype=torch.float32, device=device)
X_coll_tensor.requires_grad_()

# ---------------------------
# 5. Define network architecture
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
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
        self.net_rho = MLP(input_dim=2, hidden_dim=rho_hidden_dim, num_hidden=rho_num_hidden, output_dim=1)
        self.q_net   = MLP(input_dim=1, hidden_dim=q_hidden_dim, num_hidden=q_num_hidden, output_dim=1)
    def forward(self, X):
        # X: (batch, 2)
        rho_hat = self.net_rho(X)      # predict ρ (normalized space)
        Q_hat = self.q_net(rho_hat)    # compute Q
        eps = 1e-6
        u_hat = Q_hat / (rho_hat + eps)  # compute predicted speed u
        return rho_hat, Q_hat, u_hat

# ---------------------------
# 6. Define physics residual computation (using collocation points)
# ---------------------------
def compute_physics_residual(model, X_coll):
    """
    Compute physics residual f = (ρ_t + Q_x)
    where ρ_t = ∂ρ/∂t, Q_x = ∂Q/∂x.
    X_coll: collocation points used for residual computation, already Tensor with requires_grad=True
    """
    rho_hat, Q_hat, _ = model(X_coll)
    grads_rho = torch.autograd.grad(rho_hat, X_coll, grad_outputs=torch.ones_like(rho_hat), create_graph=True)[0]
    rho_t = grads_rho[:, 1:2]  # derivative w.r.t. t
    grads_Q = torch.autograd.grad(Q_hat, X_coll, grad_outputs=torch.ones_like(Q_hat), create_graph=True)[0]
    Q_x = grads_Q[:, 0:1]      # derivative w.r.t. x
    f = rho_t + Q_x
    return f

# ---------------------------
# 7. Convert training, validation, and test data to Tensor
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
# 8. Define multi-task training function with Early Stopping
# ---------------------------
def train_PINN_multitask(model, X_train, rho_train, u_train, X_val, rho_val, u_val, X_coll, num_epochs=20000, lr=1e-4, patience=200):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    tasks = ["rho", "phys", "speed"]
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        #X_train.requires_grad_()
        hat_rho, Q_hat, u_hat = model(X_train)
        loss_rho = mse(hat_rho, rho_train)
        f = compute_physics_residual(model, X_coll)
        loss_phys = mse(f, torch.zeros_like(f))
        loss_speed = mse(u_hat, u_train)

        # Compute gradients for each task (allow unused params)
        grads_rho = torch.autograd.grad(loss_rho, model.parameters(), retain_graph=True, allow_unused=True)
        grads_phys = torch.autograd.grad(loss_phys, model.parameters(), retain_graph=True, allow_unused=True)
        grads_speed = torch.autograd.grad(loss_speed, model.parameters(), retain_graph=True, allow_unused=True)
        grads_rho = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_rho, model.parameters())]
        grads_phys = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_phys, model.parameters())]
        grads_speed = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads_speed, model.parameters())]
        orig_grads = {"rho": grads_rho, "phys": grads_phys, "speed": grads_speed}
        grads = {t: [g.clone() for g in orig_grads[t]] for t in tasks}
        loss_data = {"rho": loss_rho.item(), "phys": loss_phys.item(), "speed": loss_speed.item()}

        gn = gradient_normalizers(grads, loss_data, 'l2')
        for t in tasks:
            for i in range(len(grads[t])):
                grads[t][i] = grads[t][i] / gn[t]

        vecs = []
        for t in tasks:
            vec = np.concatenate([g.detach().cpu().numpy().reshape(-1) for g in grads[t]])
            vecs.append(vec)

        sol, min_norm = MinNormSolverNumpy.find_min_norm_element(vecs)
        scale = {}
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])

        optimizer.zero_grad()
        X_train.requires_grad_()
        hat_rho, Q_hat, u_hat = model(X_train)
        loss_rho = mse(hat_rho, rho_train)
        f = compute_physics_residual(model, X_coll)
        loss_phys = mse(f, torch.zeros_like(f))
        loss_speed = mse(u_hat, u_train)
        total_loss = scale["rho"] * loss_rho + scale["phys"] * loss_phys + scale["speed"] * loss_speed
        total_loss.backward()
        optimizer.step()

        # Validation: compute sum of rho and speed loss
        model.eval()
        with torch.no_grad():
            hat_rho_val, _, u_hat_val = model(X_val)
            val_loss_rho = mse(hat_rho_val, rho_val)
            val_loss_speed = mse(u_hat_val, u_val)
            val_loss = val_loss_rho + val_loss_speed

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch+1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, loss_rho: {loss_rho.item():.6e}, loss_phys: {loss_phys.item():.6e}, loss_speed: {loss_speed.item():.6e}, val_loss: {val_loss.item():.6e}, scale: {scale}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# ---------------------------
# 9. Run multiple experiments and record L2 relative errors of density and speed on test set
# ---------------------------
num_experiments = 10
density_errors = []  # store L2 relative error for density
speed_errors = []    # store L2 relative error for speed

for exp in range(num_experiments):
    print(f"\nExperiment {exp+1}/{num_experiments}")
    pinn_model = PINN().to(device)
    pinn_model = train_PINN_multitask(pinn_model, X_train_tensor, rho_train_tensor, u_train_tensor,
                                      X_validate_tensor, rho_validate_tensor, u_validate_tensor,
                                      X_coll_tensor, num_epochs=50000, lr=1e-5, patience=500)
    pinn_model.eval()
    with torch.no_grad():
        rho_pred = pinn_model(X_test_tensor)[0]
        u_pred = pinn_model(X_test_tensor)[2]
    # Denormalize predictions
    rho_pred = rho_pred.cpu().numpy() * rho_std + rho_mean
    u_pred = u_pred.cpu().numpy() * u_std + u_mean
    rho_true_orig = rho_test_true
    u_true_orig = speed_test_true
    err_rho = np.linalg.norm(rho_pred - rho_true_orig) / np.linalg.norm(rho_true_orig)
    err_u = np.linalg.norm(u_pred - u_true_orig) / np.linalg.norm(u_true_orig)
    density_errors.append(err_rho)
    speed_errors.append(err_u)
    print(f"  Density L2 Relative Error: {err_rho:.3e}, Speed L2 Relative Error: {err_u:.3e}")

density_errors = np.array(density_errors)
speed_errors = np.array(speed_errors)

print("\n================ Final Results ================")
print(f"Density L2 Relative Error: Mean = {density_errors.mean():.3e}, Std = {density_errors.std():.3e}")
print(f"Speed L2 Relative Error:   Mean = {speed_errors.mean():.3e}, Std = {speed_errors.std():.3e}")
