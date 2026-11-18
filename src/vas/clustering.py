"""Step 3 (optional, exploratory) — interaction analysis and pose clustering.

For one session: smooth trajectories, compute movement/proximity metrics
between the therapist and the child, detect therapist approaches and the
child's response, lagged speed correlation, activity-based session phases,
and unsupervised KMeans pose clusters per person.

Outputs go to ``outputs/<session>/pose_clustering/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import project_config, resolve_paths
from .utils import read_table, write_json, write_table

logger = logging.getLogger(__name__)
COLORS = {"Therapist": "#440154", "Child": "#31688e"}


@dataclass
class ClusteringConfig:
    movement_threshold: float = 10.0
    proximity_close: float = 200.0
    proximity_medium: float = 400.0
    response_window: float = 5.0
    confidence_threshold: float = 0.5
    smooth_window: int = 11
    smooth_poly: int = 3
    k_range: tuple = (2, 8)
    phase_window: float = 30.0
    n_phases: int = 4
    max_frames_for_k_search: int = 5000
    therapist_role: str = "Therapist"
    child_role: str = "Child"
    random_state: int = 42

    @classmethod
    def from_dict(cls, d: dict | None) -> ClusteringConfig:
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (d or {}).items() if k in names}
        if "k_range" in kw:
            kw["k_range"] = tuple(kw["k_range"])
        return cls(**kw)


# =============================================================================
# Preprocessing
# =============================================================================
def preprocess(df: pd.DataFrame, cfg: ClusteringConfig, roles: dict) -> pd.DataFrame:
    df = df.copy()
    df["role"] = df["person_label"].map(lambda p: roles.get(p, p))
    df = df[df["role"].isin([cfg.therapist_role, cfg.child_role])]
    df = df[df["avg_pose_conf"] > cfg.confidence_threshold] if "avg_pose_conf" in df else df
    if df.empty:
        return df
    coords = [c for c in df.columns if c.endswith(("_x", "_y"))]
    df = df.sort_values(["role", "time_s"]).reset_index(drop=True)
    df[coords] = df.groupby("role")[coords].transform(lambda s: s.interpolate(limit_direction="both"))

    def smooth(s: pd.Series) -> pd.Series:
        if len(s) < cfg.smooth_window or s.isna().any():
            return s
        return pd.Series(savgol_filter(s.to_numpy(), cfg.smooth_window, cfg.smooth_poly), index=s.index)

    df[coords] = df.groupby("role")[coords].transform(smooth)
    xs, ys = [c for c in coords if c.endswith("_x")], [c for c in coords if c.endswith("_y")]
    df["com_x"], df["com_y"] = df[xs].mean(axis=1), df[ys].mean(axis=1)
    df["hands_x"] = df[["l_wrist_x", "r_wrist_x"]].mean(axis=1)
    df["hands_y"] = df[["l_wrist_y", "r_wrist_y"]].mean(axis=1)
    dt = df.groupby("role")["time_s"].diff()
    for part, (x, y) in {"com": ("com_x", "com_y"), "head": ("nose_x", "nose_y"),
                         "hands": ("hands_x", "hands_y")}.items():
        vx, vy = df.groupby("role")[x].diff() / dt, df.groupby("role")[y].diff() / dt
        df[f"{part}_speed"] = np.hypot(vx, vy)
        if part == "com":
            df["com_vx"], df["com_vy"] = vx, vy
    return df


# =============================================================================
# Interaction analysis
# =============================================================================
def movement_profiles(df: pd.DataFrame, cfg: ClusteringConfig) -> dict:
    out = {}
    for role, p in df.groupby("role"):
        moving = p["com_speed"] > cfg.movement_threshold
        out[role] = {
            "frames": len(p), "activity_ratio": float(moving.mean()),
            "mean_speed": float(p["com_speed"].mean()), "max_speed": float(p["com_speed"].max()),
            "mean_head_speed": float(p["head_speed"].mean()), "mean_hand_speed": float(p["hands_speed"].mean()),
            "coverage_x": float(np.ptp(p["com_x"].dropna())) if p["com_x"].notna().any() else np.nan,
            "coverage_y": float(np.ptp(p["com_y"].dropna())) if p["com_y"].notna().any() else np.nan,
        }
    return out


def proximity(df: pd.DataFrame, cfg: ClusteringConfig) -> pd.DataFrame:
    cols = ["time_s", "com_x", "com_y", "com_speed"]
    t = df[df["role"] == cfg.therapist_role][cols]
    c = df[df["role"] == cfg.child_role][cols]
    m = t.merge(c, on="time_s", suffixes=("_therapist", "_child")).sort_values("time_s")
    if m.empty:
        return m
    m["distance"] = np.hypot(m["com_x_therapist"] - m["com_x_child"], m["com_y_therapist"] - m["com_y_child"])
    m["zone"] = np.select([m["distance"] < cfg.proximity_close, m["distance"] < cfg.proximity_medium],
                          ["close", "medium"], "far")
    m["distance_vel"] = m["distance"].diff() / m["time_s"].diff()
    return m.reset_index(drop=True)


def approach_events(df: pd.DataFrame, prox: pd.DataFrame, cfg: ClusteringConfig) -> pd.DataFrame:
    if prox.empty:
        return pd.DataFrame()
    approaching = (prox["distance_vel"] < -5) & (prox["com_speed_therapist"] > cfg.movement_threshold)
    starts = prox[approaching & ~approaching.shift(1, fill_value=False)]
    child = df[df["role"] == cfg.child_role]
    events = []
    for _, row in starts.iterrows():
        t0 = row["time_s"]
        w = child[(child["time_s"] >= t0) & (child["time_s"] <= t0 + cfg.response_window)]
        if w.empty:
            continue
        moved = w[w["com_speed"] > cfg.movement_threshold]
        events.append({
            "approach_time": t0, "initial_distance": row["distance"],
            "child_responded": not moved.empty,
            "child_head_response": bool((w["head_speed"] > cfg.movement_threshold).any()),
            "child_hand_response": bool((w["hands_speed"] > cfg.movement_threshold).any()),
            "response_latency": float(moved["time_s"].iloc[0] - t0) if not moved.empty else np.nan,
        })
    return pd.DataFrame(events)


def lagged_correlation(prox: pd.DataFrame, max_lag: int = 30) -> pd.DataFrame:
    m = prox[["time_s", "com_speed_therapist", "com_speed_child"]].dropna()
    if len(m) < 50:
        return pd.DataFrame()
    dt = float(m["time_s"].diff().median())
    a, b = m["com_speed_therapist"].to_numpy(), m["com_speed_child"].to_numpy()
    rows = []
    for lag in range(-max_lag, max_lag + 1):  # lag > 0: child follows therapist
        x, y = (a[:-lag], b[lag:]) if lag > 0 else (a[-lag:], b[:lag or None]) if lag < 0 else (a, b)
        if len(x) > 10 and x.std() > 0 and y.std() > 0:
            rows.append({"lag_frames": lag, "lag_s": lag * dt, "r": pearsonr(x, y)[0]})
    return pd.DataFrame(rows)


def session_phases(prox: pd.DataFrame, cfg: ClusteringConfig) -> pd.DataFrame:
    if prox.empty:
        return pd.DataFrame()
    p = prox.copy()
    p["bin"] = ((p["time_s"] - p["time_s"].min()) // cfg.phase_window).astype(int)
    agg = p.groupby("bin").agg(
        time_start=("time_s", "min"), time_end=("time_s", "max"),
        therapist_activity=("com_speed_therapist", lambda s: float((s > cfg.movement_threshold).mean())),
        child_activity=("com_speed_child", lambda s: float((s > cfg.movement_threshold).mean())),
        therapist_speed=("com_speed_therapist", "mean"), child_speed=("com_speed_child", "mean"),
        mean_distance=("distance", "mean"),
    ).reset_index(drop=True)
    if len(agg) >= cfg.n_phases:
        X = StandardScaler().fit_transform(agg.drop(columns=["time_start", "time_end"]).fillna(0))
        agg["phase"] = KMeans(cfg.n_phases, random_state=cfg.random_state, n_init=10).fit_predict(X)
    return agg


# =============================================================================
# Pose clustering
# =============================================================================
def pose_matrix(person: pd.DataFrame) -> pd.DataFrame:
    """Hip-centred, torso-scaled coordinates (so clusters reflect posture, not position)."""
    xs = [c for c in person.columns if c.endswith("_x") and not c.startswith(("com_", "hands_"))]
    hip_x, hip_y = person["mid_hip_x"], person["mid_hip_y"]
    torso = np.hypot(person["neck_x"] - hip_x, person["neck_y"] - hip_y).replace(0, np.nan)
    torso = torso.fillna(torso.median())
    feats = {}
    for x in xs:
        j = x[:-2]
        if f"{j}_y" in person:
            feats[f"{j}_x"] = (person[x] - hip_x) / torso
            feats[f"{j}_y"] = (person[f"{j}_y"] - hip_y) / torso
    out = pd.DataFrame(feats, index=person.index)
    out = out.loc[:, out.notna().mean() > 0.5]
    return out.fillna(out.median())


def choose_k(X: np.ndarray, cfg: ClusteringConfig) -> dict:
    rng = np.random.default_rng(cfg.random_state)
    idx = rng.choice(len(X), min(len(X), cfg.max_frames_for_k_search), replace=False)
    Xs = X[idx]
    ks = [k for k in range(*cfg.k_range) if k < len(Xs)]
    sil, inertia = [], []
    for k in ks:
        km = KMeans(k, random_state=cfg.random_state, n_init=10).fit(Xs)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Xs, km.labels_)))
    k_sil = ks[int(np.argmax(sil))]
    if len(ks) >= 3:
        k_elbow = ks[int(np.argmax(np.diff(np.asarray(inertia) / max(inertia), n=2))) + 1]
    else:
        k_elbow = k_sil
    # prefer the elbow when its silhouette is within 0.05 of the best
    k = k_elbow if max(sil) - sil[ks.index(k_elbow)] <= 0.05 else k_sil
    return {"k": k, "k_silhouette": k_sil, "k_elbow": k_elbow, "ks": ks, "silhouette": sil, "inertia": inertia}


# =============================================================================
# Plots
# =============================================================================
def _save(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_overview(df, prox, events, phases, cfg: ClusteringConfig, out: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for role, d in df.groupby("role"):
        ax.plot(d["time_s"], d["com_speed"].rolling(50, center=True, min_periods=1).mean(),
                label=role, color=COLORS.get(role), lw=1.5)
    ax.set(title="Movement speed (rolling mean)", xlabel="time (s)", ylabel="speed (px/s)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out / "movement_speed.png", dpi)

    if not prox.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(prox["time_s"], prox["distance"].rolling(30, center=True, min_periods=1).mean(), color="#31688e")
        ax.axhline(cfg.proximity_close, color="red", ls="--", label="close")
        ax.axhline(cfg.proximity_medium, color="orange", ls="--", label="medium")
        ax.set(title="Therapist–child distance", xlabel="time (s)", ylabel="distance (px)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, out / "proximity_distance.png", dpi)

    if not events.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        resp = events["child_responded"].astype(bool)
        ax.scatter(events.loc[resp, "approach_time"], events.loc[resp, "initial_distance"],
                   c="#35b779", label="child responded", s=40)
        ax.scatter(events.loc[~resp, "approach_time"], events.loc[~resp, "initial_distance"],
                   c="#fde725", edgecolors="k", label="no response", s=40)
        ax.set(title=f"Therapist approaches ({resp.mean():.0%} with child response)",
               xlabel="time (s)", ylabel="distance at approach (px)")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, out / "approach_events.png", dpi)

    if not phases.empty and "phase" in phases:
        fig, ax = plt.subplots(figsize=(14, 3))
        cmap = plt.get_cmap("viridis", phases["phase"].nunique())
        for _, r in phases.iterrows():
            ax.axvspan(r["time_start"], r["time_end"], color=cmap(int(r["phase"])), alpha=0.8)
        ax.set(title="Session phases (KMeans on activity windows)", xlabel="time (s)", yticks=[])
        _save(fig, out / "session_phases.png", dpi)


def plot_clusters(X: np.ndarray, labels: np.ndarray, kinfo: dict, role: str, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].plot(kinfo["ks"], kinfo["inertia"], "o-")
    axes[0].axvline(kinfo["k_elbow"], color="red", ls="--")
    axes[0].set(title="Elbow", xlabel="k", ylabel="inertia")
    axes[1].plot(kinfo["ks"], kinfo["silhouette"], "o-", color="green")
    axes[1].axvline(kinfo["k_silhouette"], color="red", ls="--")
    axes[1].set(title="Silhouette", xlabel="k")
    pcs = PCA(2).fit(X)
    Z = pcs.transform(X)
    axes[2].scatter(Z[:, 0], Z[:, 1], c=labels, s=2, cmap="viridis")
    axes[2].set(title=f"PCA (k={kinfo['k']})",
                xlabel=f"PC1 {pcs.explained_variance_ratio_[0]:.0%}",
                ylabel=f"PC2 {pcs.explained_variance_ratio_[1]:.0%}")
    pd.Series(labels).value_counts().sort_index().plot(kind="bar", ax=axes[3], color="steelblue", rot=0)
    axes[3].set(title="Cluster sizes", xlabel="cluster", ylabel="frames")
    fig.suptitle(f"Pose clusters — {role}")
    _save(fig, out / f"clusters_{role}.png", dpi)


# =============================================================================
# Runner
# =============================================================================
def run_pose_clustering(project_dir: Path, cfg: dict, force: bool = False) -> Path | None:
    project_dir = Path(project_dir)
    cfg = project_config(cfg, project_dir)
    session_out = resolve_paths(cfg).outputs / project_dir.name
    out = session_out / "pose_clustering"
    result_path = out / "pose_clusters.parquet"
    if result_path.exists() and not force:
        logger.info("Clustering results exist — %s", result_path)
        return result_path
    src = session_out / "processed_data.parquet"
    if not src.exists():
        logger.error("Pose table missing: %s", src)
        return None
    out.mkdir(parents=True, exist_ok=True)

    ccfg = ClusteringConfig.from_dict(cfg.get("pose_clustering"))
    dpi = cfg.get("plotting", {}).get("dpi", 150)
    df = preprocess(read_table(src), ccfg, cfg.get("roles") or {})
    if df.empty:
        logger.error("No therapist/child frames after preprocessing — check `roles`.")
        return None

    prox = proximity(df, ccfg)
    events = approach_events(df, prox, ccfg)
    lag = lagged_correlation(prox)
    phases = session_phases(prox, ccfg)
    plot_overview(df, prox, events, phases, ccfg, out, dpi)

    df["cluster"] = -1
    summary = {"movement": movement_profiles(df, ccfg), "clusters": {}}
    for role, person in df.groupby("role"):
        M = pose_matrix(person)
        if len(M) < max(ccfg.k_range) * 5:
            logger.info("Too few frames for clustering %s (%d)", role, len(M))
            continue
        X = StandardScaler().fit_transform(M)
        kinfo = choose_k(X, ccfg)
        labels = KMeans(kinfo["k"], random_state=ccfg.random_state, n_init=10).fit_predict(X)
        df.loc[M.index, "cluster"] = labels
        summary["clusters"][role] = {k: v for k, v in kinfo.items()}
        plot_clusters(X, labels, kinfo, role, out, dpi)
        logger.info("%s: k=%d (silhouette %.3f)", role, kinfo["k"], max(kinfo["silhouette"]))

    if not lag.empty:
        best = lag.loc[lag["r"].abs().idxmax()]
        summary["lagged_correlation_peak"] = {"lag_s": float(best["lag_s"]), "r": float(best["r"])}
    if not events.empty:
        summary["approach_events"] = {"n": len(events), "response_rate": float(events["child_responded"].mean())}

    write_table(df[["session", "person_label", "role", "time_s", "com_speed", "cluster"]]
                if "session" in df else df[["person_label", "role", "time_s", "com_speed", "cluster"]], result_path)
    for name, t in {"proximity.csv": prox, "approach_events.csv": events,
                    "lagged_correlation.csv": lag, "session_phases.csv": phases}.items():
        if not t.empty:
            write_table(t, out / name)
    write_json(summary, out / "summary.json")
    return result_path
