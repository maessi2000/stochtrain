"""run.py — esegue C0-C2 per seed, C3 come ensemble, aggrega scalari in summary.json."""
import numpy as np
import argparse, glob, json, os, pickle
from dataclasses import replace
from config import Config, PRESETS
from data import load
from experiments import run_seed, run_C3
from plotting import make_all

def _med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")

def scalars(r):
    """Metriche scalari per seed: i numeri citabili nella tesi."""
    S = {}
    if "C0" in r:
        c = r["C0"]; dC, dS = np.diag(c["C0"]), np.diag(c["S"])
        S["C0"] = dict(tr_ratio=float(np.trace(c["C0"]) / np.trace(c["S"])), mode_ratio_med=_med(dC / dS),
                       mode_ratio_min=float(np.min(dC / dS)), mode_ratio_max=float(np.max(dC / dS)),
                       attr_ratio_med=_med(c["var_attr_emp"] / np.maximum(c["var_attr_pred"], 1e-30)),
                       tau_max=c["tau_max"], burn_over_tau=c["burn_over_tau"], stat_over_tau=c["stat_over_tau"])
        if "S_cf" in c:
            S["C0"]["cf_ratio_med"] = _med(dC / np.diag(c["S_cf"]))
            S["C0"]["cf_ratio_max_dev"] = float(np.max(np.abs(dC / np.diag(c["S_cf"]) - 1)))
    if "C1" in r:
        c = r["C1"]; s = c["scaling"]
        rr, tm, tp = (np.array([x[k] for x in s], float) for k in ("ratio", "tr_meas", "tr_pred"))
        ok = np.isfinite(tm) & (tm > 0)
        slope = float(np.polyfit(np.log(rr[ok]), np.log(tm[ok]), 1)[0]) if ok.sum() > 2 else float("nan")
        slope_p = float(np.polyfit(np.log(rr[ok]), np.log(tp[ok]), 1)[0]) if ok.sum() > 2 else float("nan")
        coll = [np.std(tm[ok][np.isclose(rr[ok], v)] / v) / np.mean(tm[ok][np.isclose(rr[ok], v)] / v)
                for v in np.unique(np.round(rr[ok], 12)) if np.isclose(rr[ok], v).sum() > 1]
        S["C1"] = dict(tr_ratio=float(np.trace(c["C0"]) / np.trace(c["S"])), acc=c["acc"], scaling_slope=slope,
                       scaling_slope_pred=slope_p, collapse_err=float(np.mean(coll)) if coll else float("nan"),
                       tau_ratio_med=_med(c["tau_emp"] / c["tau_pred"]), tau_max=c["tau_max"],
                       burn_over_tau=c["burn_over_tau"], stat_over_tau=c["stat_over_tau"])
    if "C2" in r and "frac_in_U_mlp" in r["C2"]:
        c = r["C2"]
        S["C2"] = dict(frac_S_med=_med(c["frac_in_U_mlp"]), frac_C0_med=_med(c["frac_C0_mlp"]),
                       lin_closure_med=_med(c["var_emp_linear"] / np.maximum(c["var_pred_linear"], 1e-30)))
    return S

def summarize(R):
    per = [scalars(r) for r in R]
    out = {"meta": {"n_seeds": len(R), "seeds": [r["seed"] for r in R]}}
    for ex in sorted({e for p in per for e in p}):
        out[ex] = {}
        for k in sorted({k for p in per if ex in p for k in p[ex]}):
            v = np.array([p[ex][k] for p in per if ex in p and k in p[ex]], float)
            v = v[np.isfinite(v)]
            out[ex][k] = ([float("nan"), float("nan")] if v.size == 0 else
                          [float(v[0]), 0.0] if v.size == 1 else [float(v.mean()), float(v.std(ddof=1))])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="core", choices=list(PRESETS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--only", nargs="+", default=None, help="C0 C1 C2 C3")
    args = ap.parse_args()
    cfg = replace(Config(**PRESETS[args.preset]))
    for it in args.set or []:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, eval(v) if isinstance(cur, (tuple, list)) else type(cur)(v))
    os.makedirs(args.out, exist_ok=True)
    only = set(args.only or ["C0", "C1", "C2", "C3"])
    if not args.aggregate_only:
        data = load(cfg)
        for s in args.seeds:
            p = os.path.join(args.out, f"seed_{s}.pkl")
            if os.path.exists(p) and not args.force:
                print(f"seed {s} presente, salto"); continue
            r = run_seed(cfg, data, s, only=only & {"C0", "C1", "C2"})
            r["_cfg"] = cfg
            pickle.dump(r, open(p, "wb"))
            print(f"seed {s} fatto")
        pe = os.path.join(args.out, "erg.pkl")
        if "C3" in only and (args.force or not os.path.exists(pe)):
            ref_p = sorted(glob.glob(os.path.join(args.out, "seed_*.pkl")))
            ref = pickle.load(open(ref_p[0], "rb")).get("C0") if ref_p else None
            if ref is None:
                print("C3 saltato: serve C0 (base Hessiana di riferimento)")
            else:
                pickle.dump(run_C3(cfg, data, ref), open(pe, "wb"))
                print("C3 fatto")
    R = [pickle.load(open(p, "rb")) for p in sorted(glob.glob(os.path.join(args.out, "seed_*.pkl")))]
    summ = summarize(R)
    pe = os.path.join(args.out, "erg.pkl")
    if os.path.exists(pe):
        e = pickle.load(open(pe, "rb"))
        summ["C3"] = {"Ts": e["Ts"], "n_seeds": e["n_seeds"], "t_star": e["t_star"].tolist(), "tau": e["tau"].tolist(),
                      "var_seed_over_pred": (e["var_seed"] / np.maximum(e["var_seed_pred"], 1e-30)).tolist()}
    clean = lambda o: ({k: clean(v) for k, v in o.items()} if isinstance(o, dict) else
                       [clean(x) for x in o] if isinstance(o, (list, tuple)) else
                       None if (isinstance(o, float) and np.isnan(o)) else o)
    json.dump(clean(summ), open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)
    make_all(args.out)
    print(f"fatto: {len(R)} seed -> {args.out}")

if __name__ == "__main__":
    main()
