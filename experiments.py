"""experiments.py — i quattro esperimenti C0-C3, nessuna dipendenza dai file precedenti."""
import numpy as np
from models import Curvature, LinearModel, LinearReg, MLP, make_linear
from attribution import get_attr
from dynamics import (ou_model, relaxation_times, _safe_tau, check_timescales, tau_max_of,
                      closed_form_sigma, ou_cov_at)

# ---- helper SGD: il weight decay viene applicato QUI, una sola volta (i modelli restituiscono il gradiente dei dati)
def sgd(model, th, X, y, lr, bs, wd, steps, rng, rec=None):
    g = np.empty(model.N)
    idx_all = rng.integers(0, len(y), size=(steps, bs))
    losses = []
    for t in range(steps):
        idx = idx_all[t]
        loss, _ = model.loss_grad(th, X[idx], y[idx], out=g)
        if wd: g += wd * th
        if not np.isfinite(g).all(): break
        old = th.copy() if rec else None
        th -= lr * g
        if rec: rec(th, old)
        losses.append(loss)
    return np.array(losses)

class Recorder:
    def __init__(self, N, U=None, center=None, n_steps=0, snap_every=0, attr_fn=None, attr_X=None):
        self.U, self.c = (U, center) if U is not None else (None, None)
        self.Z = np.full((n_steps, U.shape[1]), np.nan) if U is not None else None   # NaN se sgd si interrompe
        self.snaps, self.snap_every, self.ns = (np.empty((n_steps//snap_every, N)) if snap_every else None), snap_every, 0
        self.attr_fn, self.attr_X = attr_fn, attr_X
        self.attrs = [] if attr_fn is not None else None
        self.t = 0
    def __call__(self, th, old):
        if self.U is not None: self.Z[self.t] = (th - self.c) @ self.U
        if self.snaps is not None and self.t % self.snap_every == 0 and self.ns < len(self.snaps):
            self.snaps[self.ns] = th; self.ns += 1
        if self.attr_fn is not None and self.attr_X is not None and self.t % self.snap_every == 0:
            self.attrs.append(self.attr_fn(th))
        self.t += 1

class _MeanRec:
    """Media di theta passo-passo (centro theta_bar della finestra pilota)."""
    def __init__(self, N): self.s, self.n = np.zeros(N), 0
    def __call__(self, th, old): self.s += th; self.n += 1

def pilot_center(m, th, X, y, cfg, rng):
    """Centro theta_bar: media di theta su cfg.n_pilot passi dopo il burn-in (th avanza in place)."""
    r = _MeanRec(m.N)
    sgd(m, th, X, y, cfg.lr, cfg.bs, cfg.wd, cfg.n_pilot, rng, rec=r)
    return r.s / r.n

def _c0_data(cfg, data):
    """Dati del modello lineare di C0/C3: regressione (MSE, caso esatto) o i dati dati (softmax)."""
    return data["reg"] if (cfg.c0_model == "mse" and "reg" in data) else data

def _mlp(cfg, data):
    sizes = (cfg.mlp_sizes if cfg.dataset == "synthetic"
             else (data["Xtr"].shape[1], 64, 32, cfg.n_classes))
    return MLP(sizes, wd=cfg.wd, activation=cfg.activation)

# ---- C0: validazione sul lineare
def run_C0(cfg, data, seed, log=print):
    rng = np.random.default_rng(seed)
    dl = _c0_data(cfg, data)
    X, Y = dl["Xtr"], dl["ytr"]
    m = make_linear(cfg, dl)
    th = m.init(rng)
    sgd(m, th, X, Y, cfg.lr, cfg.bs, cfg.wd, cfg.burn, rng)
    theta_bar = pilot_center(m, th, X, Y, cfg, rng)              # H, U, D valutati al centro, non in un punto rumoroso

    # Hessiana e Dk
    idx = rng.choice(len(Y), min(cfg.hess_n, len(Y)), replace=False)
    cur = Curvature(m, X[idx], Y[idx])
    evals, U = cur.top_eigs(theta_bar, cfg.hess_k, rng)
    Hk = cur.project(theta_bar, U)
    gi = rng.choice(len(Y), min(cfg.grad_n, len(Y)), replace=False)
    gX, gy = X[gi], Y[gi]
    Dk = np.cov(m.proj_grads(theta_bar, gX, gy, U).T); Dk = 0.5 * (Dk + Dk.T)
    tau_max = check_timescales(cfg, evals, log)

    # finestra stazionaria
    rec = Recorder(m.N, U=U, center=theta_bar,
                   n_steps=cfg.stat_steps, snap_every=cfg.snap_every)
    sgd(m, th, X, Y, cfg.lr, cfg.bs, cfg.wd, cfg.stat_steps, rng, rec=rec)
    if not np.isfinite(rec.Z).all():
        raise RuntimeError("SGD divergente o interrotta nella finestra stazionaria")

    Z = rec.Z - rec.Z.mean(0)
    C0_emp = Z.T @ Z / len(Z)
    F, S, Q = ou_model(Hk, Dk, cfg.lr, cfg.wd, cfg.bs)

    # proiezione su span(U) in R^N, poi estrazione del blocco W
    snaps = np.stack([rec.snaps[k] for k in range(rec.ns)])      # (ns, N)
    snaps_c = snaps - snaps.mean(0)
    snaps_proj = (snaps_c @ U) @ U.T                              # proiezione ortonormale su span(U)
    W_slice = m.blocks[0][1]
    Ws_full = snaps_c[:, W_slice]
    Ws_proj = snaps_proj[:, W_slice]
    var_w_full  = Ws_full.var(0)                                  # dentro + fuori U
    var_w_in_U  = Ws_proj.var(0)                                  # dentro U
    U_W = U[W_slice, :]
    var_w_pred  = np.sum((U_W @ S) * U_W, axis=1)                 # Lyapunov in span(U)

    v_full_by_in = var_w_full.reshape(cfg.d, -1).mean(1)
    v_inU_by_in  = var_w_in_U.reshape(cfg.d, -1).mean(1)
    frac_in_U = np.divide(v_inU_by_in, v_full_by_in, out=np.ones(cfg.d), where=v_full_by_in > 0)

    # attribuzioni per-feature: φ_i = w_i x_i, mediata sulle classi
    n_attr = min(cfg.attr_points, len(dl["Xte"]))
    xs = dl["Xte"][:n_attr]
    x_sq = (xs ** 2).mean(0)
    var_attr_emp      = (x_sq[:, None] * var_w_in_U.reshape(cfg.d, -1)).mean(1)
    var_attr_pred     = (x_sq[:, None] * var_w_pred.reshape(cfg.d, -1)).mean(1)
    var_attr_emp_full = (x_sq[:, None] * var_w_full.reshape(cfg.d, -1)).mean(1)

    out = dict(seed=seed, evals=evals, F=F, S=S, C0=C0_emp, Hk=Hk, Dk=Dk, U=U, center=theta_bar,
               var_attr_emp=var_attr_emp, var_attr_pred=var_attr_pred,
               var_attr_emp_full=var_attr_emp_full, frac_in_U=frac_in_U,
               var_w_emp=var_w_in_U, var_w_pred=var_w_pred, var_w_full=var_w_full,
               trS=float(np.trace(S)), tau=relaxation_times(F, cfg.lr),
               tau_max=float(tau_max), burn_over_tau=cfg.burn / tau_max, stat_over_tau=cfg.stat_steps / tau_max,
               model=("mse" if dl.get("task") == "reg" else "softmax"))
    if isinstance(m, LinearReg):                                  # formula chiusa (5)/(6) con C empirica dei dati
        Cemp = X.T @ X / len(X)
        Sig = closed_form_sigma(Cemp, cfg.sigma ** 2, cfg.lr, cfg.wd, cfg.bs)
        SigN = np.zeros((m.N, m.N)); SigN[W_slice, W_slice] = Sig
        out["S_cf"] = U.T @ SigN @ U
    return out

# ---- C1: trasposizione alla MLP
def run_C1(cfg, data, seed, log=print):
    rng = np.random.default_rng(seed)
    m = _mlp(cfg, data)
    th = m.init(rng)
    sgd(m, th, data["Xtr"], data["ytr"], cfg.lr, cfg.bs, cfg.wd, cfg.burn, rng)
    theta_bar = pilot_center(m, th, data["Xtr"], data["ytr"], cfg, rng)

    # --- sottospazio Hessiano e statistiche del gradiente, al centro
    idx = rng.choice(len(data["ytr"]), cfg.hess_n, replace=False)
    cur = Curvature(m, data["Xtr"][idx], data["ytr"][idx])
    evals, U = cur.top_eigs(theta_bar, cfg.hess_k, rng)
    Hk = cur.project(theta_bar, U)
    gi = rng.choice(len(data["ytr"]), cfg.grad_n, replace=False)
    gX, gy = data["Xtr"][gi], data["ytr"][gi]
    Dk = np.cov(m.proj_grads(theta_bar, gX, gy, U).T)
    Dk = 0.5 * (Dk + Dk.T)
    tau_max = check_timescales(cfg, evals, log)

    # --- punti di test e classi predette, FISSATE al centro
    n_attr = min(cfg.attr_points, len(data["Xte"]))
    xs = data["Xte"][:n_attr]
    A, _ = m._forward(theta_bar, xs)
    c_pred = np.argmax(A[-1], axis=1)
    attr = get_attr(cfg)

    def attr_fn(th_t):
        return np.stack([attr(m, th_t, x, int(c)) for x, c in zip(xs, c_pred)])

    # --- finestra stazionaria: Z e attribuzioni registrate insieme
    rec = Recorder(m.N, U=U, center=theta_bar,
                   n_steps=cfg.stat_steps, snap_every=cfg.snap_every,
                   attr_fn=attr_fn, attr_X=xs)
    sgd(m, th, data["Xtr"], data["ytr"], cfg.lr, cfg.bs, cfg.wd,
        cfg.stat_steps, rng, rec=rec)
    if not np.isfinite(rec.Z).all():
        raise RuntimeError("SGD divergente o interrotta nella finestra stazionaria")

    # --- analisi OU
    Z = rec.Z - rec.Z.mean(0)
    C0_emp = Z.T @ Z / len(Z)
    a, b = Z[:-1], Z[1:]
    C1_emp = b.T @ a / len(a)
    F_emp = C1_emp @ np.linalg.pinv(C0_emp + 1e-10 * np.eye(len(C0_emp)))
    F, S, Q = ou_model(Hk, Dk, cfg.lr, cfg.wd, cfg.bs)
    tau_pred = relaxation_times(F)
    tau_emp = _safe_tau(F_emp)

    # --- attribuzioni nel tempo
    attrs = np.stack(rec.attrs) if rec.attrs else None   # (n_snap, n_attr, d)
    var_attr_time = attrs.var(0).mean(0) if attrs is not None else None  # (d,)

    # --- scaling eta/B: rilassamento e finestra ∝ tau_max(lr) = 1/(lr (lambda_min + wd)), cosi' la
    #     frazione di varianza lenta catturata non dipende da lr (niente pendenza spuria)
    scaling = []
    base = (cfg.lr, cfg.bs)
    for lr in cfg.scaling_lrs:
        for bs in cfg.scaling_bs:
            ratio = lr / bs
            if (lr, bs) == base:
                scaling.append(dict(lr=lr, bs=bs, ratio=ratio,
                                    tr_meas=float(np.trace(C0_emp)),
                                    tr_pred=float(np.trace(S)), n_rec=cfg.stat_steps))
                continue
            if lr * (evals[0] + cfg.wd) >= 2.0:        # instabilità lineare
                scaling.append(dict(lr=lr, bs=bs, ratio=ratio,
                                    tr_meas=np.nan, tr_pred=np.nan, n_rec=0))
                continue
            n_rec = int(min(cfg.scaling_cap, cfg.scaling_taus * tau_max_of(cfg, evals, lr)))
            n_rec = max(cfg.chunk, n_rec)
            th2 = th.copy()
            sgd(m, th2, data["Xtr"], data["ytr"], lr, bs, cfg.wd, n_rec, rng)       # rilassamento
            rec2 = Recorder(m.N, U=U, center=theta_bar, n_steps=n_rec)
            sgd(m, th2, data["Xtr"], data["ytr"], lr, bs, cfg.wd, n_rec, rng, rec=rec2)
            Z2 = rec2.Z - rec2.Z.mean(0)
            tr_meas = float(np.trace(Z2.T @ Z2 / len(Z2)))
            _, S2, _ = ou_model(Hk, Dk, lr, cfg.wd, bs)
            scaling.append(dict(lr=lr, bs=bs, ratio=ratio,
                                tr_meas=tr_meas, tr_pred=float(np.trace(S2)), n_rec=n_rec))

    return dict(
        seed=seed,
        evals=evals,
        F=F, S=S, C0=C0_emp, C1_emp=C1_emp, F_emp=F_emp,
        trS=float(np.trace(S)),
        tau_pred=tau_pred, tau_emp=tau_emp,
        tau_max=float(tau_max), burn_over_tau=cfg.burn / tau_max, stat_over_tau=cfg.stat_steps / tau_max,
        # materiale per C2 (niente riaddestramento): punto di espansione = centro theta_bar
        th=theta_bar.copy(), U=U.copy(),
        xs=xs, c_pred=c_pred,
        attrs=attrs, var_attr_time=var_attr_time,
        scaling=scaling,
        acc=float(m.accuracy(th, data["Xte"], data["yte"])),
    )

# ---- C2: varianza delle attribuzioni
def run_C2(cfg, data, seed, linear_res=None, mlp_res=None):
    if linear_res is None or mlp_res is None:
        return dict(seed=seed, note="C0/C1 mancanti")

    m = _mlp(cfg, data)
    th, U, S, C0 = mlp_res["th"], mlp_res["U"], mlp_res["S"], mlp_res["C0"]
    xs, c_pred = mlp_res["xs"], mlp_res["c_pred"]
    attr = get_attr(cfg)

    def jac_phi_U(x, c, eps=1e-4):
        cols = np.empty((len(x), U.shape[1]))
        for j in range(U.shape[1]):
            up = attr(m, th + eps*U[:, j], x, int(c))
            um = attr(m, th - eps*U[:, j], x, int(c))
            cols[:, j] = (up - um) / (2*eps)
        return cols

    vS, vC = [], []
    for x, c in zip(xs, c_pred):
        J = jac_phi_U(x, c)
        vS.append(np.diag(J @ S @ J.T))          # risposta lineare con la covarianza di Lyapunov (predetta)
        vC.append(np.diag(J @ C0 @ J.T))         # risposta lineare con la covarianza empirica dei modi (isola non-linearita' + leakage)
    var_loc, var_loc_C0 = np.mean(vS, axis=0), np.mean(vC, axis=0)

    var_time_mlp = mlp_res["attrs"].var(axis=0).mean(axis=0)  # (d,) totale
    safe = np.maximum(var_time_mlp, 1e-30)
    frac_S  = np.divide(var_loc,    safe, out=np.zeros_like(var_loc),    where=var_time_mlp > 0)
    frac_C0 = np.divide(var_loc_C0, safe, out=np.zeros_like(var_loc_C0), where=var_time_mlp > 0)

    var_emp_linear      = linear_res["var_attr_emp"]
    var_pred_linear     = linear_res["var_attr_pred"]
    var_emp_linear_full = linear_res["var_attr_emp_full"]
    frac_in_U_linear    = np.divide(var_pred_linear, np.maximum(var_emp_linear_full, 1e-30),
                                    out=np.zeros_like(var_pred_linear),
                                    where=var_emp_linear_full > 0)

    return dict(seed=seed,
                var_emp_linear=var_emp_linear,
                var_pred_linear=var_pred_linear,
                var_emp_linear_full=var_emp_linear_full,
                frac_in_U_linear=frac_in_U_linear,
                var_time_mlp=var_time_mlp,
                var_loc=var_loc, var_loc_C0=var_loc_C0,
                frac_in_U_mlp=frac_S, frac_C0_mlp=frac_C0)

# ---- C3: ergodicità per modo (ensemble di cfg.n_seeds_erg inizializzazioni indipendenti)
def run_C3(cfg, data, ref, log=print):
    """Per ogni T in erg_train_lengths: n_seeds_erg traiettorie da init indipendenti. Nella base Hessiana (U, centro)
    del riferimento `ref` (= risultato C0): Var_seed,k(T), Var_time,k (seconda meta' della traiettoria, mediata sui
    seed), predizione eq. (8) con la matrice F, S di Lyapunov e S0 = covarianza empirica degli z iniziali."""
    dl = _c0_data(cfg, data)
    X, Y = dl["Xtr"], dl["ytr"]
    m = make_linear(cfg, dl)
    U, center, F, S = ref["U"], ref["center"], ref["F"], ref["S"]
    Ts, k, n = list(cfg.erg_train_lengths), U.shape[1], cfg.n_seeds_erg
    var_seed, var_time = np.zeros((len(Ts), k)), np.zeros((len(Ts), k))
    z0_all, zT_all = [], []
    for it, T in enumerate(Ts):
        zT, vt = [], []
        for s in range(n):
            rng = np.random.default_rng(100000 + 1000 * it + s)
            th = m.init(rng)
            z0_all.append((th - center) @ U)
            rec = Recorder(m.N, U=U, center=center, n_steps=T)
            sgd(m, th, X, Y, cfg.lr, cfg.bs, cfg.wd, T, rng, rec=rec)
            zT.append(rec.Z[-1]); vt.append(rec.Z[T // 2:].var(0))
        zT = np.array(zT)
        var_seed[it], var_time[it] = zT.var(0, ddof=1), np.mean(vt, axis=0)
        zT_all.append(zT)
        log(f"C3: T={T} fatto")
    z0 = np.array(z0_all)
    S0 = np.cov(z0.T)
    pred = np.array([np.diag(ou_cov_at(F, S, S0, T)) for T in Ts])
    rho = np.clip(np.abs(np.diag(F)), 1e-12, 1 - 1e-12)
    tau_k = -1.0 / np.log(rho)
    s0, Sk = np.diag(S0), np.diag(S)
    t_star = np.where(s0 > Sk, 0.5 * tau_k * np.log(np.maximum(s0, 1e-300) / np.maximum(Sk, 1e-300)), 0.0)
    return dict(Ts=Ts, var_seed=var_seed, var_time=var_time, var_seed_pred=pred, S_diag=Sk, s0_diag=s0,
                tau=tau_k, t_star=t_star, evals=ref["evals"], n_seeds=n)

# ---- run_seed: orchestra C0-C2 per un seed (C3 e' un ensemble: vedi run.py)
def run_seed(cfg, data, seed, only=None):
    only = set(only or ["C0", "C1", "C2"])
    r = dict(seed=seed)
    if "C0" in only: r["C0"] = run_C0(cfg, data, seed)
    if "C1" in only: r["C1"] = run_C1(cfg, data, seed)
    if "C2" in only: r["C2"] = run_C2(cfg, data, seed, r.get("C0"), r.get("C1"))
    return r
