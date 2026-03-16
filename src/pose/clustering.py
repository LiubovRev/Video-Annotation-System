#!/usr/bin/env python3
# coding: utf-8
"""
clustering.py
-------------
Optional Step 3 of the Video Annotation Pipeline.
Pose clustering and therapist-child interaction analysis.

Modules:
  DataPreprocessor  — load, filter, interpolate, smooth pose data
  ClinicalAnalyzer  — movement, proximity, approach-response, lagged correlation, session phases
  PoseClusterer     — optimal-k selection (silhouette + elbow), KMeans clustering
  Visualizer        — plot and save all figures to output_dir

Can be run standalone or imported by full_pipeline.py via
run_pose_clustering(project_dir, output_dir, cfg).

Configuration loaded from config.yaml at project root.
Optional: skip_pose_clustering = True in flags to skip.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class ClusteringConfig:
    """Parameters for pose clustering and clinical analysis."""

    # Movement detection
    movement_threshold = 10  # px/frame

    # Proximity zones (pixels)
    proximity_close = 200
    proximity_medium = 400

    # Approach-response
    response_window = 5  # seconds

    # Data preprocessing
    confidence_threshold = 0.5
    smooth_window = 11
    smooth_poly = 3

    # Clustering
    k_range = range(2, 8)

    # Session phase detection
    phase_window = 30  # seconds per window
    n_phases = 4

    # Labels
    therapist_label = "Therapist"
    child_label = "Child"

    @classmethod
    def from_dict(cls, d: dict):
        """Populate config from dict (e.g., cfg['pose_clustering'])."""
        obj = cls()
        for key, val in d.items():
            if hasattr(obj, key):
                if key == "k_range" and isinstance(val, list):
                    val = range(*val)
                setattr(obj, key, val)
        return obj


# =============================================================================
# Part 1 — Data Preprocessing
# =============================================================================

class DataPreprocessor:
    """Load, filter, interpolate, and smooth pose data."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.df: pd.DataFrame = pd.DataFrame()

    def load_data(self, filepath: Path) -> pd.DataFrame:
        self.df = pd.read_csv(filepath)
        self.df["person_label"] = self.df["person_label"].replace({
            "Patient1": "Child",
            "Therapist1": "Therapist",
        })
        self.df = self.df[self.df["person_label"].isin(["Child", "Therapist"])].reset_index(drop=True)
        print(f"  Loaded {len(self.df):,} rows — persons: {list(self.df['person_label'].unique())}")
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """Filter low-confidence rows, interpolate gaps, smooth coordinates."""
        # Filter by confidence
        before = len(self.df)
        self.df = self.df[self.df["avg_pose_conf"] > self.config.confidence_threshold].copy()
        print(f"  Removed {before - len(self.df):,} low-confidence rows")

        # Interpolate per person
        self.df = (
            self.df.groupby("person_label", group_keys=False)
            .apply(lambda x: (
                x.drop(columns="person_label")
                 .set_index("time_s")
                 .interpolate(method="linear")
                 .reset_index()
                 .assign(person_label=x["person_label"].iloc[0])
            ))
            .reset_index(drop=True)
        )

        # Smooth coordinates
        coord_cols = [c for c in self.df.columns if c.endswith("_x") or c.endswith("_y")]
        for col in coord_cols:
            self.df[f"{col}_smooth"] = self.df.groupby("person_label")[col].transform(
                lambda x: savgol_filter(x, self.config.smooth_window, self.config.smooth_poly)
                if len(x) >= self.config.smooth_window else x
            )
        print(f"  Smoothed {len(coord_cols)} coordinate columns")
        return self.df

    def compute_body_metrics(self) -> pd.DataFrame:
        """Compute center of mass, velocities, head, and hand positions."""
        x_cols = [c for c in self.df.columns if c.endswith("_x_smooth")]
        y_cols = [c for c in self.df.columns if c.endswith("_y_smooth")]

        if not x_cols:
            x_cols = [c for c in self.df.columns if c.endswith("_x") and "_smooth" not in c and c != "com_x"]
            y_cols = [c for c in self.df.columns if c.endswith("_y") and "_smooth" not in c and c != "com_y"]

        # Centre of mass
        self.df["com_x"] = self.df[x_cols].mean(axis=1)
        self.df["com_y"] = self.df[y_cols].mean(axis=1)

        # Velocities
        def _velocity(pos_col):
            return self.df.groupby("person_label")[pos_col].diff() / self.df.groupby("person_label")["time_s"].diff()

        self.df["com_x_vel"] = _velocity("com_x")
        self.df["com_y_vel"] = _velocity("com_y")
        self.df["com_speed"] = np.sqrt(self.df["com_x_vel"]**2 + self.df["com_y_vel"]**2)

        # Head (nose)
        hx = "nose_x_smooth" if "nose_x_smooth" in self.df else "nose_x"
        hy = "nose_y_smooth" if "nose_y_smooth" in self.df else "nose_y"
        self.df["head_x"], self.df["head_y"] = self.df[hx], self.df[hy]
        self.df["head_x_vel"], self.df["head_y_vel"] = _velocity("head_x"), _velocity("head_y")
        self.df["head_speed"] = np.sqrt(self.df["head_x_vel"]**2 + self.df["head_y_vel"]**2)

        # Hands (average of wrists)
        rwx = "r_wrist_x_smooth" if "r_wrist_x_smooth" in self.df else "r_wrist_x"
        lwx = "l_wrist_x_smooth" if "l_wrist_x_smooth" in self.df else "l_wrist_x"
        rwy = "r_wrist_y_smooth" if "r_wrist_y_smooth" in self.df else "r_wrist_y"
        lwy = "l_wrist_y_smooth" if "l_wrist_y_smooth" in self.df else "l_wrist_y"

        self.df["hands_x"] = (self.df[rwx] + self.df[lwx]) / 2
        self.df["hands_y"] = (self.df[rwy] + self.df[lwy]) / 2
        self.df["hands_x_vel"], self.df["hands_y_vel"] = _velocity("hands_x"), _velocity("hands_y")
        self.df["hands_speed"] = np.sqrt(self.df["hands_x_vel"]**2 + self.df["hands_y_vel"]**2)

        print("  Body metrics computed (COM, velocities, head, hands)")
        return self.df


# =============================================================================
# Part 2 — Clinical Interaction Analysis
# =============================================================================

class ClinicalAnalyzer:
    """Analyze therapist-child interactions: movement, proximity, approach-response, lagged correlation, session phases."""

    def __init__(self, df: pd.DataFrame, config: ClusteringConfig):
        self.df = df
        self.config = config
        self.movement_profiles = {}
        self.proximity_df = pd.DataFrame()
        self.approach_events_df = pd.DataFrame()
        self.lagged_corr_df = pd.DataFrame()
        self.phases_df = pd.DataFrame()

    def analyze_movement_profiles(self) -> dict:
        """Compute activity ratios, mean/max speed, and spatial coverage per person."""
        for label in [self.config.therapist_label, self.config.child_label]:
            person = self.df[self.df["person_label"] == label].copy()
            if person.empty:
                continue
            person["is_moving"] = person["com_speed"] > self.config.movement_threshold
            self.movement_profiles[label] = {
                "total_frames": len(person),
                "moving_frames": int(person["is_moving"].sum()),
                "activity_ratio": float(person["is_moving"].mean()),
                "mean_speed": float(person["com_speed"].mean()),
                "max_speed": float(person["com_speed"].max()),
                "std_speed": float(person["com_speed"].std()),
                "mean_head_speed": float(person["head_speed"].mean()),
                "mean_hand_speed": float(person["hands_speed"].mean()),
                "spatial_coverage_x": float(person["com_x"].max() - person["com_x"].min()),
                "spatial_coverage_y": float(person["com_y"].max() - person["com_y"].min()),
            }
        return self.movement_profiles

    def analyze_proximity(self) -> pd.DataFrame:
        """Compute inter-person distances, proximity zones, and approach/withdrawal flags."""
        t = self.df[self.df["person_label"] == self.config.therapist_label][["time_s", "com_x", "com_y", "com_speed"]].copy()
        c = self.df[self.df["person_label"] == self.config.child_label][["time_s", "com_x", "com_y", "com_speed"]].copy()
        merged = pd.merge(t, c, on="time_s", suffixes=("_therapist", "_child"))
        if merged.empty:
            return pd.DataFrame()

        merged["distance"] = np.sqrt(
            (merged["com_x_therapist"] - merged["com_x_child"])**2 +
            (merged["com_y_therapist"] - merged["com_y_child"])**2
        )
        merged["proximity_zone"] = "far"
        merged.loc[merged["distance"] < self.config.proximity_medium, "proximity_zone"] = "medium"
        merged.loc[merged["distance"] < self.config.proximity_close, "proximity_zone"] = "close"
        merged["distance_vel"] = merged["distance"].diff() / merged["time_s"].diff()
        merged["therapist_approaching"] = merged["distance_vel"] < -5
        merged["therapist_withdrawing"] = merged["distance_vel"] > 5

        self.proximity_df = merged
        return self.proximity_df

    def detect_approach_response(self) -> pd.DataFrame:
        """Detect child responses to therapist approaches within response_window."""
        if self.proximity_df.empty:
            return pd.DataFrame()
        prox = self.proximity_df.copy()
        prox["approaching"] = (prox["distance_vel"] < -5) & (prox["com_speed_therapist"] > self.config.movement_threshold)
        approach_starts = prox[prox["approaching"] & ~prox["approaching"].shift(1).fillna(False)]

        events = []
        for _, row in approach_starts.iterrows():
            t0 = row["time_s"]
            child_window = self.df[
                (self.df["person_label"] == self.config.child_label) &
                (self.df["time_s"] >= t0) &
                (self.df["time_s"] <= t0 + self.config.response_window)
            ]
            if child_window.empty:
                continue
            mt = self.config.movement_threshold
            moved = (child_window["com_speed"] > mt).any()
            head_moved = (child_window["head_speed"] > mt).any()
            hands_moved = (child_window["hands_speed"] > mt).any()
            latency = None
            responders = child_window[child_window["com_speed"] > mt]
            if not responders.empty:
                latency = float(responders["time_s"].iloc[0] - t0)
            events.append({
                "approach_time": t0,
                "initial_distance": row["distance"],
                "child_responded": moved,
                "child_head_response": head_moved,
                "child_hand_response": hands_moved,
                "response_latency": latency,
                "therapist_speed": row["com_speed_therapist"],
            })

        self.approach_events_df = pd.DataFrame(events)
        return self.approach_events_df

    def compute_lagged_correlation(self, max_lag: int = 10) -> pd.DataFrame:
        """Compute lagged correlations between therapist and child speed."""
        t = self.df[self.df["person_label"] == self.config.therapist_label].sort_values("time_s")
        c = self.df[self.df["person_label"] == self.config.child_label].sort_values("time_s")
        merged = pd.merge(
            t[["time_s", "com_speed"]],
            c[["time_s", "com_speed"]],
            on="time_s",
            suffixes=("_therapist", "_child"),
        ).dropna()
        if len(merged) < 50:
            return pd.DataFrame()

        dt = merged["time_s"].diff().mean() if len(merged) > 1 else 0.033
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                ts, cs = merged["com_speed_therapist"].iloc[-lag:].values, merged["com_speed_child"].iloc[:lag].values
            elif lag > 0:
                ts, cs = merged["com_speed_therapist"].iloc[:-lag].values, merged["com_speed_child"].iloc[lag:].values
            else:
                ts, cs = merged["com_speed_therapist"].values, merged["com_speed_child"].values
            if len(ts) > 10 and len(ts) == len(cs) and ts.std() > 0 and cs.std() > 0:
                corr, _ = pearsonr(ts, cs)
                correlations.append({"lag": lag, "lag_seconds": lag * dt, "correlation": corr})
        self.lagged_corr_df = pd.DataFrame(correlations)
        return self.lagged_corr_df

    def detect_session_phases(self) -> pd.DataFrame:
        """Segment session into phases based on activity features and KMeans clustering."""
        t_data = self.df[self.df["person_label"] == self.config.therapist_label].copy()
        c_data = self.df[self.df["person_label"] == self.config.child_label].copy()
        if t_data.empty or c_data.empty:
            return pd.DataFrame()

        time_bins = np.arange(self.df["time_s"].min(), self.df["time_s"].max(), self.config.phase_window)
        phases = []
        mt = self.config.movement_threshold
        for t_start, t_end in zip(time_bins, time_bins[1:]):
            tw = t_data[(t_data["time_s"] >= t_start) & (t_data["time_s"] < t_end)]
            cw = c_data[(c_data["time_s"] >= t_start) & (c_data["time_s"] < t_end)]
            if tw.empty or cw.empty:
                continue
            phases.append({
                "time_start": t_start,
                "time_end": t_end,
                "time_mid": (t_start + t_end) / 2,
                "therapist_activity": float((tw["com_speed"] > mt).mean()),
                "child_activity": float((cw["com_speed"] > mt).mean()),
                "therapist_mean_speed": float(tw["com_speed"].mean()),
                "child_mean_speed": float(cw["com_speed"].mean()),
            })
        self.phases_df = pd.DataFrame(phases)

        # Cluster phases
        if len(self.phases_df) >= self.config.n_phases:
            feats = self.phases_df[["therapist_activity", "child_activity", "therapist_mean_speed", "child_mean_speed"]].values
            km = KMeans(n_clusters=self.config.n_phases, random_state=42, n_init=10)
            self.phases_df["phase"] = km.fit_predict(feats)
        return self.phases_df


# =============================================================================
# Part 3 — Pose Clustering
# =============================================================================

class PoseClusterer:
    """KMeans clustering on pose features with automatic optimal-k selection."""

    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.kmeans = None
        self.optimal_k = None
        self.optimal_k_sil = None
        self.optimal_k_elbow = None

    def extract_features(self, df: pd.DataFrame, person_label: str):
        """Extract pose features and optional velocities for clustering."""
        person_df = df[df["person_label"] == person_label].copy()
        x_cols = [c for c in person_df.columns if "_x_smooth" in c or (c.endswith("_x") and "_smooth" not in c and c != "com_x")]
        y_cols = [c for c in person_df.columns if "_y_smooth" in c or (c.endswith("_y") and "_smooth" not in c and c != "com_y")]
        features = person_df[x_cols + y_cols].copy()
        if "com_speed" in person_df.columns:
            features["com_speed"] = person_df["com_speed"].values
            features["com_x_vel"] = person_df["com_x_vel"].values
            features["com_y_vel"] = person_df["com_y_vel"].values
        return features.ffill().bfill(), person_df["time_s"].values

    def find_optimal_k(self, X: np.ndarray):
        """Select optimal k using silhouette and elbow methods."""
        X_scaled = self.scaler.fit_transform(X)
        k_values = list(self.config.k_range)
        sil_scores, inertia_scores = [], []

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertia_scores.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels) if k > 1 else -1)

        # Silhouette-based optimal k
        valid = [(i, s) for i, s in enumerate(sil_scores) if s > -1]
        best_sil_idx, best_sil_score = max(valid, key=lambda x: x[1])
        self.optimal_k_sil = k_values[best_sil_idx]

        # Elbow-based optimal k
        if len(k_values) >= 3:
            inertia_norm = np.array(inertia_scores) / max(inertia_scores)
            second_deriv = np.diff(inertia_norm, n=2)
            elbow_idx = int(np.argmax(second_deriv)) + 2
            self.optimal_k_elbow = k_values[elbow_idx]
        else:
            self.optimal_k_elbow = k_values[1] if len(k_values) > 1 else k_values[0]

        # Decide final k
        threshold = 0.05
        if self.optimal_k_sil != self.optimal_k_elbow:
            elbow_sil = sil_scores[k_values.index(self.optimal_k_elbow)]
            self.optimal_k = self.optimal_k_elbow if best_sil_score - elbow_sil <= threshold else self.optimal_k_sil
        else:
            self.optimal_k = self.optimal_k_sil

        return self.optimal_k, sil_scores, inertia_scores, k_values

    def fit_predict(self, X: np.ndarray, n_clusters: int = None) -> np.ndarray:
        if n_clusters is None:
            n_clusters = self.optimal_k or 3
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        print(f"    KMeans k={n_clusters}, silhouette={sil:.3f}")
        return labels


# =============================================================================
# Part 4 — Visualizer
# =============================================================================

class Visualizer:
    """Plot and save clinical and clustering results."""

    COLORS = {"Therapist": "#440154", "Child": "#31688e", "Response": "#35b779", "No Response": "#fde725"}
    TS, LS, TKS, LS2 = 36, 32, 28, 26  # font sizes

    @classmethod
    def _save(cls, fig, path: Path, name: str, dpi: int = 300) -> None:
        if path:
            fig.savefig(path / name, dpi=dpi, bbox_inches="tight", facecolor="white")
            print(f"    Saved: {name}")
        plt.close(fig)




# =============================================================================
# Pipeline Runner
# =============================================================================

def run_pose_clustering(project_dir: Path, output_dir: Path, cfg: dict = None) -> dict:
    """Full pose clustering + clinical analysis pipeline for one project."""
    cluster_cfg = ClusteringConfig()
    if cfg and "pose_clustering" in cfg:
        cluster_cfg = ClusteringConfig.from_dict(cfg["pose_clustering"])
    dpi = cfg.get("plotting", {}).get("dpi", 300) if cfg else 300

    project_dir = Path(project_dir)
    csv_path = project_dir / "processed_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"processed_data.csv not found: {csv_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nPose clustering: {project_dir.name}\n{'='*70}")

    # Step 1: Preprocess
    print("\n[1/4] Loading and preprocessing data...")
    prep = DataPreprocessor(cluster_cfg)
    df = prep.load_data(csv_path)
    df = prep.preprocess()
    df = prep.compute_body_metrics()

    # Step 2: Clinical analysis
    print("\n[2/4] Clinical interaction analysis...")
    clinical = ClinicalAnalyzer(df, cluster_cfg)
    clinical.analyze_movement_profiles()
    clinical.analyze_proximity()
    clinical.detect_approach_response()
    clinical.compute_lagged_correlation()
    clinical.detect_session_phases()
    Visualizer.plot_clinical_overview(clinical, output_dir=output_dir, dpi=dpi)

    # Step 3: Pose clustering
    print("\n[3/4] Clustering pose patterns...")
    clusterer = PoseClusterer(cluster_cfg)
    cluster_results = {}
    for person_label in df["person_label"].unique():
        print(f"  → {person_label}...")
        try:
            features, times = clusterer.extract_features(df, person_label)
            if len(features) < 10:
                print(f"    Skipping — only {len(features)} frames.")
                continue
            optimal_k, sil_scores, inertia_scores, k_values = clusterer.find_optimal_k(features.values)
            labels = clusterer.fit_predict(features.values, n_clusters=optimal_k)
            cluster_results[person_label] = {
                "features": features, "labels": labels, "times": times,
                "optimal_k": optimal_k, "silhouette_scores": sil_scores, "k_values": k_values,
            }
            Visualizer.plot_clustering_results(
                features.values, labels, person_label, optimal_k,
                sil_scores, inertia_scores, k_values,
                output_dir=output_dir, dpi=dpi
            )
        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Step 4: Combine results
    print("\n[4/4] Assembling results...")
    parts = [
        pd.DataFrame({
            "time_s": r["times"], "person_label": lbl,
            "cluster": r["labels"], "optimal_k": r["optimal_k"]
        }) for lbl, r in cluster_results.items()
    ]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    final = df.merge(combined, on=["time_s", "person_label"], how="left")
    results_path = output_dir / "pose_clusters_with_clinical.csv"
    final.to_csv(results_path, index=False)
    print(f"  Results saved: {results_path.name}")

    # Summary
    print(f"\n{'='*70}\nCLUSTERING COMPLETE\n{'='*70}")
    print(f"  {len(df):,} frames analysed")
    for lbl, m in clinical.movement_profiles.items():
        print(f"  {lbl}: {m['activity_ratio']*100:.1f}% active, mean speed {m['mean_speed']:.1f} px/frame")
    for lbl, r in cluster_results.items():
        print(f"  {lbl}: {r['optimal_k']} pose clusters (best silhouette {max(r['silhouette_scores']):.3f})")

    return {"data": final, "clinical_analyzer": clinical, "cluster_results": cluster_results, "config": cluster_cfg}


# =============================================================================
# Standalone Entry Point
# =============================================================================

def main():
    CONFIG_FILE = Path(__file__).parent.parent.parent / "config.yaml"
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    base_data_dir = Path(cfg["directories"]["base_data_dir"])
    output_base_dir = Path(cfg["directories"]["output_base_dir"])
    project_dirs = [d for d in base_data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(project_dirs)} project directories.")

    for project_dir in sorted(project_dirs):
        output_dir = output_base_dir / project_dir.name / "pose_clustering"
        run_pose_clustering(project_dir, output_dir, cfg)


if __name__ == "__main__":
    main()

