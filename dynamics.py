"""dynamics.py — OU, tempi di rilassamento, MSD teorico, formule chiuse del modello lineare, ergodicità."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

def ou_model(H, D, lr, wd, bs):
    """F = I - lr(H + wd I), Q = lr^2 D / bs, S = soluzione Lyapunov discreta.
    H e' la Hessiana dei SOLI dati (il weight decay entra qui, una volta sola)."""
    k = H.shape[0]
    F = np.eye(k) - lr * (H + wd * np.eye(k))
    Q = lr**2 * D / bs
    S = solve_discrete_lyapunov(F, Q)
    return F, 0.5*(S+S.T), Q

def relaxation_times(F, lr=None):
    ev = np.abs(np.linalg.eigvals(F))
    lam = np.clip(ev, 1e-12, 1.0 - 1e-12)
    taus = -1.0 / np.log(lam)
    return np.sort(taus)   # ordine canonico: modo 0 = più veloce

def _safe_tau(F, eps=1e-6):
    """tau_k = -1/log|λ_k|. I modi con |λ| ≥ 1-eps sono numericamente degeneri
    (finestra troppo corta per il modo k): NaN, così il plot li esclude."""
    ev = np.abs(np.linalg.eigvals(F))
    out = np.full_like(ev, np.nan)
    ok = (ev < 1 - eps) & np.isfinite(ev) & (ev > 0)
    out[ok] = -1.0 / np.log(np.clip(ev[ok], 1e-12, 1 - 1e-12))
    return np.sort(out)

def tau_max_of(cfg, evals, lr=None):
    """Tempo di rilassamento del modo piu' lento tra quelli modellati: 1/(lr (lambda_min + wd))."""
    lr = cfg.lr if lr is None else lr
    return 1.0 / (lr * (float(np.min(evals)) + cfg.wd))

def check_timescales(cfg, evals, log=print):
    """Avvisa se burn-in o finestra non coprono il modo piu' lento (le predizioni stazionarie non sarebbero valide)."""
    tau = tau_max_of(cfg, evals)
    if cfg.burn < 3 * tau:
        log(f"ATTENZIONE: burn-in {cfg.burn} < 3*tau_max = {3*tau:.0f}: i modi lenti non sono equilibrati")
    if cfg.stat_steps < 5 * tau:
        log(f"ATTENZIONE: finestra {cfg.stat_steps} < 5*tau_max = {5*tau:.0f}: varianza dei modi lenti sottostimata")
    return tau

def theoretical_msd(F, S, lags):
    """MSD totale teorico nel sottospazio: 2 [tr S - tr(F^tau S)] (F simmetrica)."""
    w, V = np.linalg.eigh(0.5 * (F + F.T))
    out = []
    for tau in lags:
        Ft = (V * np.power(w, int(tau))) @ V.T
        out.append(2.0 * (np.trace(S) - np.trace(Ft @ S)))
    return np.array(out)

def ou_cov_at(F, S, S0, t):
    """Covarianza tra seed al tempo t partendo da covarianza iniziale S0: F^t S0 F^t' + (S - F^t S F^t')  [eq. 8]."""
    w, V = np.linalg.eigh(0.5 * (F + F.T))
    Ft = (V * np.power(w, int(t))) @ V.T
    return Ft @ S0 @ Ft.T + S - Ft @ S @ Ft.T

def closed_form_sigma(C, sigma2, lr, wd, bs, exact_gamma0=True):
    """Covarianza stazionaria esatta del modello lineare MSE con dati gaussiani, nella base originale.
    wd > 0: eq. (5) (rumore additivo). wd = 0 e exact_gamma0: eq. (6) (rumore moltiplicativo)."""
    c, V = np.linalg.eigh(0.5 * (C + C.T))
    c = np.clip(c, 0.0, None)
    if wd == 0 and exact_gamma0:
        g = 2.0 - lr * c * (1.0 + 1.0 / bs)
        Phi = np.sum(c / g)
        Sk = (lr * sigma2 / bs) / (g * (1.0 - (lr / bs) * Phi))
    else:
        a = c + wd
        Sk = lr * sigma2 * c / (bs * np.maximum(a, 1e-300) * (2.0 - lr * a))
    return (V * Sk) @ V.T

def ergodicity_ratio(snaps_by_seed, active):
    """snaps_by_seed: lista di array (T, N) per seed. Var_seed / Var_time per parametro, con
    Var_seed = varianza tra seed al tempo finale e Var_time = varianza temporale (seconda meta' della
    traiettoria) MEDIATA SUI SEED (non la varianza della media d'insieme)."""
    X = np.stack([s[:, active].astype(np.float64) for s in snaps_by_seed])   # (S, T, d_act)
    S, T, _ = X.shape
    var_seed = X[:, -1, :].var(0, ddof=1) if S > 1 else np.full(X.shape[2], np.nan)
    var_time = X[:, T // 2:, :].var(1).mean(0)
    return var_seed / np.maximum(var_time, 1e-30)
