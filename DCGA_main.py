# -*- coding: utf-8 -*-
"""
DCGD joint training (NN + GA-calibrated IDM) on the real trajectory dataset.

For each algorithm in {center, avg, proj}:
  1. GA calibrates IDM parameters using the same training sequences
     that the NN later sees.
  2. Train the NN with the corresponding DCGD optimizer.
  3. Report position / velocity RMSE on 10 held-out sequences.
  4. Plot predicted vs. real follower positions and save to
     plots/<algorithm>/<ALG>_Test_Sample_<idx>.pdf
"""

import os, pickle, random, shutil
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from GA.GA         import GA
from dcgd          import DCGD
from model.nn      import NN
from model.physics import IDM

# ---------------------- 1. Configuration ----------------------
N_ITER      = 500
EVAL_NUM    = 10
NUM_TRAIN   = 1000
INPUT_DIM   = 3
N_HIDDEN    = 3
HIDDEN_DIM  = 60
OUTPUT_DIM  = 1
SEED        = 1234

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

plt.rcParams["font.size"] = 16

optimizer_kwargs = dict(lr=1e-3)
GA_ARGS = dict(
    sol_per_pop        = 10,
    num_parents_mating = 5,
    num_mutations      = 1,
    mutations_extend   = 0.1,
    num_generations    = 10,
    delta_t            = 0.1,
    mse                = "position",
    RMSPE_alpha_X      = 0.5,
    RMSPE_alpha_V      = 0.5,
    lb                 = [10, 0, 0, 0, 0],
    ub                 = [40, 10, 10, 5, 5],
    seed               = SEED,
)

# ---------------------- 2. Helper functions ----------------------
def xvfl_to_feature(arr: np.ndarray):
    dx = (arr[:, 2] - arr[:, 0]).reshape(-1, 1)[:-1]
    dv = (arr[:, 3] - arr[:, 1]).reshape(-1, 1)[:-1]
    vf = arr[:, 1].reshape(-1, 1)[:-1]
    a  = (np.diff(arr[:, 1]) * 10).reshape(-1, 1)
    return np.hstack([dx, dv, vf, a])

def split_feature_a(fa: np.ndarray, num_train: int):
    sizes = dict(train=num_train, ext=300, val=int(0.4*num_train))
    sizes["test"]  = 300
    sizes["total"] = sum(sizes.values())

    idx = np.random.choice(len(fa), sizes["total"], replace=False)
    data = fa[idx]

    trv_ext, _   = train_test_split(data, test_size=sizes["test"], random_state=SEED)
    trv, ext     = train_test_split(trv_ext, test_size=sizes["ext"], random_state=SEED)
    train, val   = train_test_split(trv,     test_size=sizes["val"], random_state=SEED)

    Xtr, atr = train[:, :3], train[:, 3:4]
    Xva, ava = val[:,   :3], val[:,   3:4]
    Xaux     = np.vstack([Xtr, ext[:, :3]])
    return Xtr, atr, Xva, ava, Xaux

def simulate(init_row: np.ndarray, leader: np.ndarray, net: NN):
    xf, vf = init_row[0], init_row[1]
    XF, VF = [xf], [vf]
    for k in range(len(leader) - 1):
        feat = np.array([leader[k,0] - XF[-1], leader[k,1] - VF[-1], VF[-1]],
                        dtype=np.float32).reshape(1, 3)
        a = net(torch.tensor(feat, device=DEVICE)).cpu().item()
        v_next = VF[-1] + 0.1 * a
        x_next = XF[-1] + 0.05 * (VF[-1] + v_next)
        VF.append(v_next); XF.append(x_next)
    return np.asarray(XF), np.asarray(VF)

def rmse(pred, true): return np.sqrt(np.mean((pred - true) ** 2))

def eval_rmse(test_seqs, net):
    px, pv, tx, tv = [], [], [], []
    for seq in test_seqs:
        Xpred, Vpred = simulate(seq[0], seq[:, 2:], net)
        px.append(Xpred); pv.append(Vpred)
        tx.append(seq[:, 0]); tv.append(seq[:, 1])
    return rmse(np.concatenate(px), np.concatenate(tx)), \
           rmse(np.concatenate(pv), np.concatenate(tv))

# ---------------------- 3. Load dataset ----------------------
with open(os.path.join("data", "real_data_lane3_f2l2.pickle"), "rb") as f:
    xvfl_all = pickle.load(f)

xvfl_test = xvfl_all[-EVAL_NUM:]      # 10 held-out sequences
train_seqs = xvfl_all[:-EVAL_NUM]

feature_a = np.vstack([xvfl_to_feature(seq) for seq in train_seqs])
Xtr_np, atr_np, Xva_np, ava_np, Xaux_np = split_feature_a(feature_a, NUM_TRAIN)

to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)
Xtr, atr = to_t(Xtr_np), to_t(atr_np)
Xva, ava = to_t(Xva_np), to_t(ava_np)
Xaux     = to_t(Xaux_np)

# ---------------------- 4. GA calibration ----------------------
print("Running GA to calibrate IDM parameters ...")
ga        = GA(GA_ARGS)
best_para, _, _ = ga.executeGA(train_seqs)
para_dict = dict(zip(["v0", "T", "s0", "a", "b"], best_para))
physics   = IDM(para_dict, {"v0": False, "T": False, "s0": False, "a": False, "b": False},
                device=DEVICE)
print("GA calibration finished:", para_dict)

# ---------------------- 5. Train DCGD variants ----------------------
loss_fn     = nn.MSELoss()
algorithms  = ["center", "avg", "proj"]
plots_root  = "plots"
if os.path.exists(plots_root): shutil.rmtree(plots_root)
os.makedirs(plots_root)

n_runs = 30

for alg in algorithms:
    print(f"\n=== DCGD-{alg.upper()} (over {n_runs} runs) ===")
    pos_rmses = []
    vel_rmses = []

    for run in range(n_runs):
        seed = SEED + run
        # reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # rebuild model & optimizer each run
        net = NN((INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM),
                 {"activation_type": "sigmoid",
                  "last_activation_type": "none",
                  "device": DEVICE,
                  "mean": Xtr.mean(0),
                  "std":  Xtr.std(0)})
        base_opt = torch.optim.Adam(net.parameters(), **optimizer_kwargs)
        dcgd_opt = DCGD(base_opt, num_pde=1, type=alg)

        # train
        best_val, best_it = float("inf"), 0
        for it in range(N_ITER):
            loss_obs = loss_fn(net(Xtr), atr)
            loss_aux = loss_fn(net(Xaux).flatten(), physics(Xaux))
            dcgd_opt.step([loss_aux, loss_obs])

            with torch.no_grad():
                val = loss_fn(net(Xva), ava).item()
            if val < best_val:
                best_val, best_it = val, it
            if it - best_it > 350:
                break

        # eval
        rx, rv = eval_rmse(xvfl_test, net)
        pos_rmses.append(rx)
        vel_rmses.append(rv)
        print(f" Run {run+1:2d}: RX={rx:.4f}, RV={rv:.4f}")

    # summary statistics
    mean_rx = np.mean(pos_rmses)
    std_rx  = np.std(pos_rmses,  ddof=1)
    mean_rv = np.mean(vel_rmses)
    std_rv  = np.std(vel_rmses,  ddof=1)
    avg_list = [(x+v)/2 for x, v in zip(pos_rmses, vel_rmses)]
    mean_avg = np.mean(avg_list)
    std_avg  = np.std(avg_list, ddof=1)

    print(f"\n--- Summary DCGD-{alg.upper()} ---")
    print(f"Position RMSE: mean = {mean_rx:.4f}, std = {std_rx:.4f}")
    print(f"Velocity RMSE: mean = {mean_rv:.4f}, std = {std_rv:.4f}")
    print(f"Average  RMSE: mean = {mean_avg:.4f}, std = {std_avg:.4f}")

    # # ---------------- Plot and save ----------------
    # out_dir = os.path.join(plots_root, alg)
    # os.makedirs(out_dir, exist_ok=True)
    # dt = 0.1
    # for idx, seq in enumerate(xvfl_test):
    #     Xpred, _ = simulate(seq[0], seq[:, 2:], net)
    #     t = np.arange(len(Xpred)) * dt
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(t, Xpred,          lw=1, color="navy",     label="Predicted")
    #     plt.plot(t, seq[:, 0], "--",lw=1, color="darkred",  label="Real")
    #     plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
    #     plt.title(f"{alg.upper()} – Test Sample {idx}")
    #     plt.grid(True, linestyle="--"); plt.legend(); plt.tight_layout()
    #     # ---------- changed filename ----------
    #     plt.savefig(os.path.join(out_dir, f"{alg.upper()}_Test_Sample_{idx}.pdf"))
    #     plt.close()

# print(f"\nAll plots saved under '{plots_root}/<center|avg|proj>/'")
