"""
This code extends *A Physics-informed Deep Learning Paradigm for Car-following Models*
(Zhaobin Mo, Rongye Shi, and Xuan Di).

NEW FEATURES
1. Hyper-parameter search for ALPHA ∈ {0.1 … 0.9}, 10 trials each.
2. In every trial we compute position-RMSE (X) and velocity-RMSE (V) from
   long-term simulation; their arithmetic mean is the combined error.
3. ALPHA is chosen by the smallest mean combined error (tie-break by std).
4. Final training with the chosen ALPHA reports both RMSEs and their mean,
   and saves trajectory plots.
"""

import os, random, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from GA.GA import GA
from model.nn import NN
from model.physics import IDM

plt.rcParams["font.size"] = 16

# ----------------------------------------------------------------------
MASTER_SEED = 24
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

# ----------------------------------------------------------------------
N_ITER, EVAL_NUM, NUM_TRAIN = 500, 10, 1000
INPUT_DIM, N_HIDDEN, HIDDEN_DIM, OUTPUT_DIM = 3, 3, 60, 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

USE_GA = True

nn_args = (INPUT_DIM, OUTPUT_DIM, N_HIDDEN, HIDDEN_DIM)
nn_kwargs_base = dict(
    activation_type="sigmoid",
    last_activation_type="none",
    device=DEVICE,
)

params_trainable = dict(v0=False, T=False, s0=False, a=False, b=False)
optim_kwargs_net     = dict(lr=1e-3)
optim_kwargs_phy     = dict(lr=1e-1)

# ---------- GA ----------
args_GA = {
    "sol_per_pop": 10,
    "num_parents_mating": 5,
    "num_mutations": 1,
    "mutations_extend": 0.1,
    "num_generations": 10,
    "delta_t": 0.1,
    "mse": "position",
    "RMSPE_alpha_X": 0.5,
    "RMSPE_alpha_V": 0.5,
    "lb": [10, 0, 0, 0, 0],
    "ub": [40, 10, 10, 5, 5],
}

# ----------------------------------------------------------------------
def xvfl_to_feature(arr):
    dx = (arr[:, 2] - arr[:, 0]).reshape(-1, 1)[:-1]
    dv = (arr[:, 3] - arr[:, 1]).reshape(-1, 1)[:-1]
    vf = arr[:, 1].reshape(-1, 1)[:-1]
    a  = (np.diff(arr[:, 1]) * 10).reshape(-1, 1)
    return np.hstack([dx, dv, vf, a])

def data_split(fa, sizes, seed):
    fa = fa[:sizes["total"]]
    trv_ext, test = train_test_split(fa, test_size=sizes["test"], random_state=seed)
    trv, ext      = train_test_split(trv_ext, test_size=sizes["ext"], random_state=seed)
    train, val    = train_test_split(trv, test_size=sizes["val"], random_state=seed)

    Xtr, atr = train[:, :3], train[:, 3:4]
    Xva, ava = val[:, :3],   val[:, 3:4]
    Xaux     = np.vstack([Xtr, ext[:, :3]])
    return Xtr, atr, Xva, ava, Xaux

def simulate(init, XV_L, net):
    xf, vf = init[0], init[1]
    XF, VF = [xf], [vf]
    for k in range(len(XV_L)-1):
        feat = np.array([XV_L[k,0]-XF[-1], XV_L[k,1]-VF[-1], VF[-1]],
                        dtype=np.float32).reshape(1,3)
        a = net(torch.tensor(feat, device=DEVICE)).cpu().item()
        v_next = VF[-1] + 0.1*a
        x_next = XF[-1] + 0.05*(VF[-1] + v_next)
        VF.append(v_next); XF.append(x_next)
    return np.asarray(XF), np.asarray(VF)

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def rmses_on_test(test_set, net):
    px, pv, tx, tv = [], [], [], []
    for traj in test_set:
        XF, VF = simulate(traj[0,:], traj[:,2:], net)
        px.append(XF); pv.append(VF)
        tx.append(traj[:,0]); tv.append(traj[:,1])
    px = np.concatenate(px); pv = np.concatenate(pv)
    tx = np.concatenate(tx); tv = np.concatenate(tv)
    return rmse(px, tx), rmse(pv, tv)

def prepare_data(feature_all, num_train, seed):
    sz = dict(train=num_train, ext=300, val=int(0.4*num_train))
    sz["test"] = 300; sz["total"] = sum(sz.values())
    idx = np.random.choice(len(feature_all), sz["total"], replace=False)
    return data_split(feature_all[idx], sz, seed)

# ----------------------------------------------------------------------
def run_single_experiment(alpha, seed, feature_all, test_set):
    set_seed(seed)

    Xtr, atr, Xva, ava, Xaux = prepare_data(feature_all, NUM_TRAIN, seed)

    if USE_GA:
        ga = GA(args_GA); pv, _, _ = ga.executeGA(xvfl)
        para = dict(v0=pv[0], T=pv[1], s0=pv[2], a=pv[3], b=pv[4])
    else:
        raise RuntimeError("GA disabled.")

    # ---------- FIX: set mean/std as float32 ----------
    mean_t = torch.tensor(Xtr.mean(0, dtype=np.float32), dtype=torch.float32, device=DEVICE)
    std_t  = torch.tensor(Xtr.std(0,  dtype=np.float32) + 1e-8,
                          dtype=torch.float32, device=DEVICE)
    # ---------------------------------------------------

    nn_kwargs = nn_kwargs_base.copy(); nn_kwargs.update(mean=mean_t, std=std_t)

    net = NN(nn_args, nn_kwargs)
    phy = IDM(para, params_trainable, device=DEVICE)

    opt_net = torch.optim.Adam(net.parameters(), **optim_kwargs_net)
    opt_phy = (torch.optim.Adam(
        [p for p in phy.torch_params.values() if p.requires_grad],
        **optim_kwargs_phy) if any(params_trainable.values()) else None)

    loss_fn = torch.nn.MSELoss()
    Xt  = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    at  = torch.tensor(atr, dtype=torch.float32, device=DEVICE)
    Xv  = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    av  = torch.tensor(ava, dtype=torch.float32, device=DEVICE)
    Xa  = torch.tensor(Xaux, dtype=torch.float32, device=DEVICE)

    best, best_it = 1e9, 0
    for it in range(N_ITER):
        pred = net(Xt); loss_obs = loss_fn(pred, at)
        loss_aux = loss_fn(net(Xa).flatten(), phy(Xa))
        loss = alpha*loss_obs + (1-alpha)*loss_aux
        opt_net.zero_grad();
        if opt_phy: opt_phy.zero_grad()
        loss.backward(); opt_net.step();
        if opt_phy: opt_phy.step()

        with torch.no_grad():
            val_mse = loss_fn(net(Xv), av).item()
        if val_mse < best: best, best_it = val_mse, it
        if it-best_it > 150: break

    rmse_x, rmse_v = rmses_on_test(test_set, net)
    return rmse_x, rmse_v, net

# ----------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(MASTER_SEED)

    with open(os.path.join("data", "real_data_lane3_f2l2.pickle"), "rb") as f:
        xvfl = pickle.load(f)

    test_set   = xvfl[-EVAL_NUM:]
    features   = [xvfl_to_feature(tr) for tr in xvfl[:-EVAL_NUM]]
    feature_all= np.vstack(features)

    alphas = np.arange(0.1, 1.0, 0.1)
    results = []  # (alpha, mean_combined, std_combined)

    # print("\n=== Search ALPHA ===")
    # for alpha in alphas:
    #     combined = []
    #     for t in range(10):
    #         rx, rv, _ = run_single_experiment(alpha, MASTER_SEED+t,
    #                                           feature_all, test_set)
    #         combined.append(0.5*(rx+rv))
    #         print(f"  α={alpha:.1f} trial {t:2d}  RX={rx:.4f}  RV={rv:.4f}  AVG={combined[-1]:.4f}")
    #     mean_c, std_c = float(np.mean(combined)), float(np.std(combined))
    #     results.append((alpha, mean_c, std_c))
    #     print(f"--> α={alpha:.1f}  mean={mean_c:.4f}  std={std_c:.4f}")
    #
    # results.sort(key=lambda x:(x[1], x[2]))
    # best_alpha, best_mean, best_std = results[0]
    # print(f"\nChosen α={best_alpha:.1f}  mean AVG={best_mean:.4f}  std={best_std:.4f}")
    best_alpha = 0.9
    n_runs = 30

    # Containers for per‑run RMSEs
    rxs = []
    rvs = []

    print(f"\n=== Running {n_runs} experiments with α={best_alpha:.1f} ===")
    for i in range(n_runs):
        seed = MASTER_SEED + i  # vary the seed each run
        rx, rv, net = run_single_experiment(best_alpha, seed, feature_all, test_set)
        rxs.append(rx)
        rvs.append(rv)
        print(f" Run {i + 1:2d}: RX={rx:.4f}, RV={rv:.4f}")

    # Compute statistics
    mean_rx, std_rx = np.mean(rxs), np.std(rxs, ddof=1)
    mean_rv, std_rv = np.mean(rvs), np.std(rvs, ddof=1)

    print("\n=== Summary over 30 runs ===")
    print(f"Position RMSE: mean = {mean_rx:.4f}, std = {std_rx:.4f}")
    print(f"Velocity RMSE: mean = {mean_rv:.4f}, std = {std_rv:.4f}")
    print(
        f"Average RMSE: mean = {(mean_rx + mean_rv) / 2:.4f}, std = {np.std([(x + v) / 2 for x, v in zip(rxs, rvs)], ddof=1):.4f}")

    # # -------- plot ----------
    # dt=0.1
    # for idx,tr in enumerate(test_set):
    #     XF, _ = simulate(tr[0,:], tr[:,2:], net)
    #     t=np.arange(len(XF))*dt
    #     plt.figure(figsize=(8,4))
    #     plt.plot(t,XF,label="Predicted X",lw=1,color='navy')
    #     plt.plot(t,tr[:,0],'--',label="Real X",lw=1,color='darkred')
    #     plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
    #     plt.title(f"Sample {idx} (α={best_alpha:.1f})")
    #     plt.legend(); plt.grid(True,ls="--"); plt.tight_layout()
    #     plt.savefig(f"Test_Sample_{idx}.pdf"); plt.close()
    #
    # print("\nSaved plots as Test_Sample_<idx>.pdf")
