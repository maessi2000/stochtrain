"""plotting.py — figure, una per ogni messaggio della tesi."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pickle, glob, os

def fig_C0(R, path):
    R = [r for r in R if "C0" in r]
    if not R:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in R:
        c = r["C0"]
        ax[0].plot(np.diag(c["C0"]) / np.diag(c["S"]), "o-", ms=3, lw=0.8, label=f"seed {r['seed']}")
        ax[1].plot(c["tau"], "o-", ms=3, lw=0.8)
        if "S_cf" in c:
            ax[2].plot(np.diag(c["C0"]) / np.diag(c["S_cf"]), "o-", ms=3, lw=0.8)
    ax[0].axhline(1, c="k", ls="--")
    ax[0].set(xlabel="modo Hessiano (autovalore decrescente)", ylabel="var misurata / Lyapunov(Hk, Dk)",
              title="Chiusura OU per modo")
    ax[0].legend(fontsize=7)
    ax[1].set(yscale="log", xlabel="modo", ylabel="tau (passi)", title="Tempi di rilassamento")
    ax[2].axhline(1, c="k", ls="--")
    ax[2].set(xlabel="modo", ylabel="var misurata / formula chiusa (5)/(6)",
              title="Formula chiusa (solo modello MSE)")
    if not any("S_cf" in r["C0"] for r in R):
        ax[2].text(0.5, 0.5, "disponibile con c0_model=mse", ha="center", va="center", transform=ax[2].transAxes)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def fig_C1(R, path):
    R = [r for r in R if "C1" in r]
    if not R:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in R:
        ax[0].semilogy(r["C1"]["evals"], "o-", ms=3, lw=0.8, label=f"seed {r['seed']}")
    ax[0].set(xlabel="modo", ylabel="autovalore Hessiano", title="Spettro MLP (top-k)")
    if len(R) > 1:
        ax[0].legend(fontsize=7)
    for r in R:
        s = r["C1"]["scaling"]
        rr  = np.array([x["ratio"]   for x in s], float)
        tm  = np.array([x["tr_meas"] for x in s], float)
        tp  = np.array([x["tr_pred"] for x in s], float)
        ok  = np.isfinite(rr) & np.isfinite(tm) & (tm > 0)
        ax[1].loglog(rr[ok], tm[ok], "o", ms=5, alpha=0.8, label=f"seed {r['seed']} (misurata)")
        okp = np.isfinite(rr) & np.isfinite(tp) & (tp > 0)
        ax[1].loglog(rr[okp], tp[okp], "s", ms=5, mfc="none", alpha=0.6)
    if ax[1].has_data():
        xmin, xmax = ax[1].get_xlim(); ymin, ymax = ax[1].get_ylim()
        ax[1].loglog([xmin, xmax], [ymin, ymin*(xmax/xmin)], "k:", lw=0.8, label="pendenza 1")
    ax[1].set(xlabel="η/B", ylabel="tr Σ_k", title="Scaling (o = misurata, □ = Lyapunov)")
    ax[1].legend(fontsize=7)
    for r in R:
        ax[2].loglog(r["C1"]["tau_pred"], r["C1"]["tau_emp"], "o", ms=5, alpha=0.8, label=f"seed {r['seed']}")
    if ax[2].has_data():
        xmin, xmax = ax[2].get_xlim()
        ax[2].loglog([xmin, xmax], [xmin, xmax], "k--", lw=0.8)
    ax[2].set(xlabel="τ predetto  (1/(η(c_k+γ)))", ylabel="τ empirico  (−1/log|λ(F_emp)|)", title="Tempi di rilassamento")
    if len(R) > 1:
        ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def fig_C2(R, path):
    R = [r for r in R if "C2" in r and "var_pred_linear" in r["C2"]]
    if not R:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in R:
        ax[0].loglog(r["C2"]["var_pred_linear"], r["C2"]["var_emp_linear"], ".", ms=6, alpha=0.7)
    xs_, ys_ = ax[0].get_xlim(), ax[0].get_ylim()
    lo, hi = min(xs_[0], ys_[0]), max(xs_[1], ys_[1])
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    ax[0].set(xlim=(lo, hi), ylim=(lo, hi), xlabel="predetto (x² Σ, dentro U)", ylabel="misurato (dentro U)",
              title="Lineare: chiusura dentro span(U)")
    for r in R:
        ax[1].plot(r["C2"]["frac_in_U_mlp"], "o-", ms=4, lw=0.7, label=f"seed {r['seed']}")
    ax[1].axhline(1.0, c="k", ls="--", lw=1)
    ax[1].set(xlabel="feature", ylabel="J S Jᵀ / var_time", title="MLP: risposta lineare con S di Lyapunov")
    ax[1].legend(fontsize=7)
    for r in R:
        ax[2].plot(r["C2"]["frac_C0_mlp"], "o-", ms=4, lw=0.7, label=f"seed {r['seed']}")
    ax[2].axhline(1.0, c="k", ls="--", lw=1)
    ax[2].set(xlabel="feature", ylabel="J C0emp Jᵀ / var_time",
              title="MLP: risposta lineare con C0 empirica (isola non-linearità + leakage)")
    ax[2].title.set_fontsize(9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def fig_C3(erg, path):
    """Ergodicità per modo (eq. 8): Var_seed,k(T)/Σ_k con predizione, rapporto Var_seed/Var_time, t*_k contro tau_k."""
    Ts = np.array(erg["Ts"], float); k = erg["var_seed"].shape[1]
    col = plt.cm.viridis(np.linspace(0, 1, k))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j in range(k):
        ax[0].loglog(Ts, erg["var_seed"][:, j] / erg["S_diag"][j], "o-", ms=4, lw=0.8, color=col[j])
        ax[0].loglog(Ts, erg["var_seed_pred"][:, j] / erg["S_diag"][j], "--", lw=0.8, color=col[j])
        ax[1].semilogx(Ts, erg["var_seed"][:, j] / np.maximum(erg["var_time"][:, j], 1e-30), "o-", ms=4, lw=0.8, color=col[j])
    ax[0].axhline(1, c="k", lw=0.8)
    ax[0].set(xlabel="T_train (passi)", ylabel="Var_seed,k(T) / Σ_k",
              title=f"Var tra seed per modo (— misura, -- eq. 8), n={erg['n_seeds']} seed")
    ax[1].axhline(1, c="k", ls="--", lw=1)
    ax[1].set(xlabel="T_train (passi)", ylabel="Var_seed / Var_time (per modo)",
              title="Rottura di ergodicità: modi lenti = viola chiaro→scuro (veloci)")
    ax[1].title.set_fontsize(9)
    ax[2].loglog(erg["tau"], np.maximum(erg["t_star"], 1), "o")
    ax[2].axhline(Ts.max(), c="r", ls=":", label="T_train max")
    ax[2].set(xlabel="τ_k (passi)", ylabel="t*_k = (τ_k/2) ln(s0²/Σ_k)", title="Tempo di crossover per modo")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def make_all(out):
    R = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(f"{out}/seed_*.pkl"))]
    fig_C0(R, f"{out}/fig_C0.png")
    fig_C1(R, f"{out}/fig_C1.png")
    fig_C2(R, f"{out}/fig_C2.png")
    pe = f"{out}/erg.pkl"
    if os.path.exists(pe):
        fig_C3(pickle.load(open(pe, "rb")), f"{out}/fig_C3.png")

if __name__ == "__main__":
    import sys
    make_all(sys.argv[1])
