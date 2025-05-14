# -*- coding: utf-8 -*-
"""
Joint training (data‑driven NN + physics‑based IDM) on the **real trajectory dataset** with
GA‑based calibration of IDM parameters.

Key changes w.r.t. the previous script
--------------------------------------
1. **Real dataset** : load ``real_data_lane3_f2l2.pickle``.
2. **GA calibration enabled** : ``USE_GA = True`` and GA hyper‑params identical to the original GA code.
3. **Evaluation metric** : Position & velocity **RMSE** (not relative error).
4. All data splits (train / val / ext / test) are **unchanged** and the GA sees exactly the same
   sequences as the learning algorithm (all samples except the last ``EVAL_NUM`` test sequences).
"""

import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from min_norm_solvers_numpy import MinNormSolverNumpy, gradient_normalizers
from GA.GA import GA
from model.nn import NN
from model.physics import IDM

# ---------------------------
# 1. Configuration
# ---------------------------
N_ITER = 500
EVAL_NUM = 10  # number of full sequences held out for final evaluation
NUM_TRAIN = 1000
INPUT_DIM = 3
N_HIDDEN = 3
HIDDEN_DIM = 60
OUTPUT_DIM = 1

USE_GA = True  # *** GA calibration switched ON ***

optimizer_kwargs = {"lr": 0.001}
optimizer_physics_kwargs = {"lr": 0.1}

SEED = 24
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# 2. Data helpers
# ---------------------------

def xvfl_to_feature(arr: np.ndarray) -> np.ndarray:
    """Convert raw XVFL sequence → [dx, dv, vf, a] per‑frame."""
    dx = (arr[:, 2] - arr[:, 0]).reshape(-1, 1)[:-1]
    dv = (arr[:, 3] - arr[:, 1]).reshape(-1, 1)[:-1]
    vf = arr[:, 1].reshape(-1, 1)[:-1]
    a  = (np.diff(arr[:, 1]) * 10).reshape(-1, 1)
    return np.hstack([dx, dv, vf, a])


def data_split(feature_a: np.ndarray, sizes: dict, *, seed: int = SEED):
    """Split into train / ext / val / test keeping row‑level randomness."""
    train_val_ext, test = train_test_split(feature_a, test_size=sizes["test"], random_state=seed)
    train_val, ext = train_test_split(train_val_ext, test_size=sizes["ext"], random_state=seed)
    train, val = train_test_split(train_val, test_size=sizes["val"], random_state=seed)

    def unpack(arr):
        X = arr[:, :3]
        a = arr[:, 3:].reshape(-1, 1)
        return X, a

    X_train, a_train = unpack(train)
    X_val, a_val = unpack(val)
    X_test, a_test = unpack(test)
    X_aux = np.vstack([X_train, ext[:, :3]])
    return X_train, a_train, X_val, a_val, X_test, a_test, X_aux


def make_datasets(num_train: int, feature_a: np.ndarray, *, seed: int = SEED):
    sizes = {"train": num_train, "ext": 300, "val": int(0.4 * num_train), "test": 300}
    sizes["total"] = sum(sizes.values())
    idx = np.random.choice(len(feature_a), sizes["total"], replace=False)
    return data_split(feature_a[idx], sizes, seed=seed)


# ---------------------------
# 3. Trajectory simulator + metrics
# ---------------------------

def simulate(init_state: np.ndarray, leader_traj: np.ndarray, nn_model: NN):
    """Forward simulate follower trajectory given initial follower state & leader trajectory."""
    xf, vf = init_state[0], init_state[1]
    XF, VF = [xf], [vf]
    for i in range(len(leader_traj) - 1):
        state = np.array([leader_traj[i, 0], leader_traj[i, 1], XF[-1], VF[-1]])
        feat = np.array([state[0] - state[2], state[1] - state[3], state[-1]])
        a_pred = nn_model(torch.tensor(feat, device=DEVICE).float().unsqueeze(0)).item()
        v_next = VF[-1] + 0.1 * a_pred
        x_next = XF[-1] + 0.5 * (VF[-1] + v_next) * 0.1
        XF.append(x_next)
        VF.append(v_next)
    return np.array(XF), np.array(VF)


def rmse_position_velocity(test_seqs: np.ndarray, nn_model: NN):
    """Compute trajectory‑level RMSE for position and velocity (absolute, not relative)."""
    preds_x, preds_v, trues_x, trues_v = [], [], [], []
    for seq in test_seqs:
        x_pred, v_pred = simulate(seq[0, :], seq[:, 2:], nn_model)
        preds_x.append(x_pred)
        preds_v.append(v_pred)
        trues_x.append(seq[:, 0])
        trues_v.append(seq[:, 1])

    preds_x = np.concatenate(preds_x)
    preds_v = np.concatenate(preds_v)
    trues_x = np.concatenate(trues_x)
    trues_v = np.concatenate(trues_v)

    pos_rmse = float(np.sqrt(np.mean((preds_x - trues_x) ** 2)))
    vel_rmse = float(np.sqrt(np.mean((preds_v - trues_v) ** 2)))
    return pos_rmse, vel_rmse


# ---------------------------
# 4. Main workflow
# ---------------------------
if __name__ == "__main__":
    n_runs = 30
    pos_rmses = []
    vel_rmses = []

    for run in range(n_runs):
        run_seed = SEED + run
        # reproducibility
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        random.seed(run_seed)

        print(f"\n=== Run {run+1}/{n_runs} (seed={run_seed}) ===")

        # -- 4.1 Load REAL dataset --
        DATA_FILE = os.path.join("data", "real_data_lane3_f2l2.pickle")
        with open(DATA_FILE, "rb") as f:
            xvfl = pickle.load(f)
        xvfl_test = xvfl[-EVAL_NUM:]
        xvfl_train_all = xvfl[:-EVAL_NUM]

        # -- 4.2 Build datasets --
        feature_a = np.vstack([xvfl_to_feature(seq) for seq in xvfl_train_all])
        X_train_np, a_train_np, X_val_np, a_val_np, X_test_np, a_test_np, X_aux_np = \
            make_datasets(NUM_TRAIN, feature_a)

        X_train = torch.tensor(X_train_np, device=DEVICE).float()
        a_train = torch.tensor(a_train_np, device=DEVICE).float()
        X_val   = torch.tensor(X_val_np,   device=DEVICE).float()
        a_val   = torch.tensor(a_val_np,   device=DEVICE).float()
        X_aux   = torch.tensor(X_aux_np,   device=DEVICE).float()

        # -- 4.3 GA calibration of IDM --
        args_GA = {
            "sol_per_pop": 10, "num_parents_mating": 5, "num_mutations": 1,
            "mutations_extend": 0.1, "num_generations": 10, "delta_t": 0.1,
            "mse": "position", "RMSPE_alpha_X": 0.5, "RMSPE_alpha_V": 0.5,
            "lb": [10, 0, 0, 0, 0], "ub": [40, 10, 10, 5, 5],
            "seed": run_seed,
        }
        ga = GA(args_GA)
        p_opt, _, _ = ga.executeGA(xvfl_train_all)
        para = dict(zip(["v0", "T", "s0", "a", "b"], p_opt))

        # -- 4.4 Build NN + physics --
        nn_kwargs = {
            "activation_type": "sigmoid", "last_activation_type": "none",
            "device": DEVICE, "mean": X_train.mean(0), "std": X_train.std(0),
        }
        net = NN((INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM), nn_kwargs).to(DEVICE)
        physics = IDM(para, {"v0": False, "T": False, "s0": False, "a": False, "b": False}, device=DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), **optimizer_kwargs)
        mse_loss = torch.nn.MSELoss()

        # -- 4.5 Multi‑task training (MGDA) --
        best_val, best_it = float("inf"), 0
        for it in range(N_ITER):
            net.train()
            a_pred      = net(X_train)
            a_aux_nn    = net(X_aux)
            a_aux_phys  = physics(X_aux)

            loss_obs = mse_loss(a_pred, a_train)
            loss_aux = mse_loss(a_aux_nn.view(-1), a_aux_phys.view(-1))

            grads_obs = torch.autograd.grad(loss_obs, net.parameters(), retain_graph=True)
            grads_aux = torch.autograd.grad(loss_aux, net.parameters(), retain_graph=True)
            grads_dict = {
                "obs": [g.clone() for g in grads_obs],
                "aux": [g.clone() for g in grads_aux],
            }
            loss_vals = {"obs": loss_obs.item(), "aux": loss_aux.item()}

            norms = gradient_normalizers(grads_dict, loss_vals, "l2")
            for t in grads_dict:
                grads_dict[t] = [g / norms[t] for g in grads_dict[t]]
            vecs = [np.concatenate([g.detach().cpu().ravel() for g in grads_dict[t]]) for t in ("obs", "aux")]
            w_obs, w_aux = MinNormSolverNumpy.find_min_norm_element(vecs)[0]

            optimizer.zero_grad()
            (w_obs * loss_obs + w_aux * loss_aux).backward()
            optimizer.step()

            val_loss = mse_loss(net(X_val), a_val).item()
            if val_loss < best_val:
                best_val, best_it = val_loss, it
            if it - best_it > 350:
                break

        # -- 4.6 Final evaluation --
        net.eval()
        pos_rmse, vel_rmse = rmse_position_velocity(xvfl_test, net)
        print(f"  → Test RMSE: Position={pos_rmse:.4f}, Velocity={vel_rmse:.4f}")

        pos_rmses.append(pos_rmse)
        vel_rmses.append(vel_rmse)
        # if pos_rmse > 60 or vel_rmse > 60:
        #     exit()

    # --- summary stats ---
    mean_px, std_px = np.mean(pos_rmses), np.std(pos_rmses, ddof=1)
    mean_vl, std_vl = np.mean(vel_rmses), np.std(vel_rmses, ddof=1)
    print("\n=== 30‑Run Summary ===")
    print(f"Position RMSE: mean = {mean_px:.4f}, std = {std_px:.4f}")
    print(f"Velocity RMSE: mean = {mean_vl:.4f}, std = {std_vl:.4f}")
    print(f"Average RMSE:   mean = {(mean_px+mean_vl)/2:.4f}, "
          f"std = {np.std([(x+v)/2 for x, v in zip(pos_rmses, vel_rmses)], ddof=1):.4f}")
    # # ---- 4.7 Plot predicted-vs-real trajectories ----
    # import matplotlib.pyplot as plt
    # plt.rcParams["font.size"] = 16
    # dt = 0.1  # same time step used in simulate()
    #
    # for idx, seq in enumerate(xvfl_test):
    #     XF_pred, _ = simulate(seq[0, :], seq[:, 2:], net)  # follower position prediction
    #     t = np.arange(len(XF_pred)) * dt
    #
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(t, XF_pred, label="Predicted", color="navy", linewidth=1)
    #     plt.plot(t, seq[:, 0], "--", label="Real", color="darkred", linewidth=1)
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Position (m)")
    #     plt.title(f"Test Sample {idx}")
    #     plt.legend()
    #     plt.grid(True, linestyle="--")
    #     plt.tight_layout()
    #     plt.savefig(f"Test_Sample_{idx}_TMGD.pdf")
    #     plt.close()
    #
    # print("\nSaved plots as Test_Sample_<idx>_TMGD.pdf")



