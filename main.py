from __future__ import annotations

import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def bootstrap_runtime_env() -> None:
    """Point caches to writable paths before importing matplotlib or Gradio."""
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")

    for key in ("XDG_CACHE_HOME", "MPLCONFIGDIR", "GRADIO_TEMP_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


bootstrap_runtime_env()

import gradio as gr

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional preflight helper
    pq = None

try:
    from histoseg.contour import (
        Pattern1IsolineConfig,
        compute_segmentation_confidence_score,
        run_pattern1_isoline,
    )

    HISTOSEG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    Pattern1IsolineConfig = None  # type: ignore[assignment]
    compute_segmentation_confidence_score = None  # type: ignore[assignment]
    run_pattern1_isoline = None  # type: ignore[assignment]
    HISTOSEG_IMPORT_ERROR = str(exc)


APP_NAME = "Fine Structure Pattern1 Explorer"
APP_DESCRIPTION = (
    "A SciLifeLab Serve Gradio app for the original HistoSeg Pattern1 contour workflow: "
    "upload Xenium cells.parquet and clusters.csv, choose Pattern1 cluster IDs, and tune "
    "the KNN and Gaussian sigma parameters before generating isoline contours."
)
DEFAULT_PATTERN1 = "10,23,19"
DEFAULT_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")


@dataclass(frozen=True)
class RuntimeProfile:
    grid_n: int
    bg_max_points: int
    syn_bg_density: float
    syn_bg_min: int
    syn_bg_max: int
    scale_label: str
    notes: tuple[str, ...]


def resolve_work_dir() -> Path:
    for candidate in (DEFAULT_WORK_DIR, FALLBACK_WORK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"Could not find a writable work directory. Tried: {DEFAULT_WORK_DIR} and {FALLBACK_WORK_DIR}"
    )


WORK_DIR = resolve_work_dir()
RUNS_DIR = WORK_DIR / "runs"


def ensure_workdirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_pattern1_clusters(raw: str) -> list[int | str]:
    normalized = str(raw or "").replace("\n", ",").replace(";", ",")
    values: list[int | str] = []
    for item in normalized.split(","):
        token = item.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            values.append(int(token))
        else:
            values.append(token)
    if not values:
        raise ValueError("Pattern1 clusters cannot be empty. Example: 10,23,19")
    return values


def stage_uploaded_file(uploaded: object | None, target_dir: Path) -> Path | None:
    if uploaded is None:
        return None
    source = Path(str(uploaded))
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    destination = target_dir / source.name
    shutil.copy2(source, destination)
    return destination


def resolve_inputs(
    *,
    cells_upload: object | None,
    clusters_upload: object | None,
    tissue_upload: object | None,
    target_dir: Path,
) -> tuple[Path, Path, Path | None]:
    cells_path = stage_uploaded_file(cells_upload, target_dir)
    clusters_path = stage_uploaded_file(clusters_upload, target_dir)
    tissue_path = stage_uploaded_file(tissue_upload, target_dir)

    if cells_path is None:
        raise ValueError("Missing cells.parquet. Please upload the Xenium cell coordinate file.")
    if clusters_path is None:
        raise ValueError("Missing clusters.csv. Please upload the GraphClust cluster assignment file.")
    return cells_path, clusters_path, tissue_path


def build_run_dir() -> Path:
    ensure_workdirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"run-{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = RUNS_DIR / f"run-{stamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def cleanup_old_runs(max_keep: int = 3) -> list[str]:
    ensure_workdirs()
    runs = sorted(
        [path for path in RUNS_DIR.glob("run-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for stale in runs[max_keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def safe_count_parquet_rows(parquet_path: Path) -> int | None:
    if pq is None:
        return None
    try:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def safe_count_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            count = sum(1 for _ in handle) - 1
        return max(count, 0)
    except Exception:
        return None


def choose_runtime_profile(
    *,
    requested_grid_n: int,
    requested_syn_bg_density: float,
    use_synth_bg: bool,
    estimated_rows: int | None,
) -> RuntimeProfile:
    effective_grid_n = int(requested_grid_n)
    bg_max_points = 60000
    syn_bg_density = float(requested_syn_bg_density)
    syn_bg_min = 20000
    syn_bg_max = 120000
    notes: list[str] = []

    ref_rows = estimated_rows or 0
    if ref_rows >= 80000:
        scale_label = "large"
        effective_grid_n = min(effective_grid_n, 450)
        bg_max_points = 12000
        syn_bg_density = min(syn_bg_density, 0.0015)
        syn_bg_min = 4000
        syn_bg_max = 12000
    elif ref_rows >= 40000:
        scale_label = "medium-large"
        effective_grid_n = min(effective_grid_n, 550)
        bg_max_points = 18000
        syn_bg_density = min(syn_bg_density, 0.0025)
        syn_bg_min = 5000
        syn_bg_max = 18000
    elif ref_rows >= 20000:
        scale_label = "medium"
        effective_grid_n = min(effective_grid_n, 650)
        bg_max_points = 25000
        syn_bg_density = min(syn_bg_density, 0.0035)
        syn_bg_min = 8000
        syn_bg_max = 25000
    elif ref_rows >= 10000:
        scale_label = "small-medium"
        effective_grid_n = min(effective_grid_n, 800)
        bg_max_points = 35000
        syn_bg_density = min(syn_bg_density, 0.0050)
        syn_bg_min = 12000
        syn_bg_max = 35000
    else:
        scale_label = "small"

    if effective_grid_n != int(requested_grid_n):
        notes.append(
            f"Auto-reduced grid_n from {requested_grid_n} to {effective_grid_n} for Serve runtime stability."
        )
    if use_synth_bg and syn_bg_density != float(requested_syn_bg_density):
        notes.append(
            f"Auto-reduced synthetic background density from {requested_syn_bg_density:.4f} to {syn_bg_density:.4f}."
        )

    return RuntimeProfile(
        grid_n=effective_grid_n,
        bg_max_points=bg_max_points,
        syn_bg_density=syn_bg_density,
        syn_bg_min=syn_bg_min,
        syn_bg_max=syn_bg_max,
        scale_label=scale_label,
        notes=tuple(notes),
    )


def directory_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def zip_outputs(output_dir: Path, archive_dir: Path) -> tuple[Path | None, str | None]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_base = archive_dir / "pattern1_outputs"
    archive_path = Path(f"{archive_base}.zip")
    output_bytes = directory_size_bytes(output_dir)
    free_bytes = shutil.disk_usage(archive_dir).free
    required_free = max(output_bytes * 2, 256 * 1024 * 1024)
    if free_bytes < required_free:
        return None, (
            "Skipped ZIP creation because disk space is low on the Serve instance. "
            "The raw output files are still available below."
        )

    try:
        archive_path_str = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
        return Path(archive_path_str), None
    except OSError as exc:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass
        if getattr(exc, "errno", None) == 28:
            return None, (
                "Skipped ZIP creation because the Serve instance ran out of disk space. "
                "The raw output files are still available below."
            )
        raise


def label_scheme_description(label_scheme: str) -> str:
    if label_scheme == "p1_is_zero":
        return "Selected Pattern1 clusters are treated as background"
    return "Selected Pattern1 clusters are treated as the signal of interest"


def emit_status(
    *,
    phase: str,
    run_dir: Path,
    lines: list[str],
    summary: dict[str, object],
    preview_path: str | None = None,
    archive_path: str | None = None,
    output_files: list[str] | None = None,
) -> tuple[str, str | None, dict[str, object], str | None, list[str]]:
    status_lines = [f"Phase: {phase}", f"Run directory: {run_dir}"]
    status_lines.extend(lines)
    return "\n".join(status_lines), preview_path, summary, archive_path, output_files or []


def run_analysis(
    cells_parquet: object | None,
    clusters_csv: object | None,
    tissue_boundary_csv: object | None,
    pattern1_clusters: str,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    min_cells_inside: int,
    margin_um: float,
    max_dist_threshold: float,
    bg_d_min: float,
    bg_d_max: float,
    bg_max_points: int,
    label_scheme: str,
    use_synth_bg: bool,
    bbox_expand_um: float,
    syn_bg_density: float,
    syn_bg_min: int,
    syn_bg_max: int,
    compute_confidence_score: bool,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )

    removed_runs = cleanup_old_runs(max_keep=3)
    start_time = time.perf_counter()
    run_dir = build_run_dir()
    upload_dir = run_dir / "inputs"
    output_dir = run_dir / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"work_dir": str(run_dir)}

    try:
        progress(0.05, desc="Staging uploaded files")
        yield emit_status(
            phase="staging-inputs",
            run_dir=run_dir,
            lines=["Copying uploaded files into the app workspace."]
            + ([f"Cleaned old run directories: {', '.join(removed_runs)}"] if removed_runs else []),
            summary=summary,
        )

        cells_path, clusters_path, tissue_path = resolve_inputs(
            cells_upload=cells_parquet,
            clusters_upload=clusters_csv,
            tissue_upload=tissue_boundary_csv,
            target_dir=upload_dir,
        )
        selected_clusters = parse_pattern1_clusters(pattern1_clusters)

        estimated_cells_rows = safe_count_parquet_rows(cells_path)
        estimated_cluster_rows = safe_count_csv_rows(clusters_path)
        synth_enabled = bool(use_synth_bg and tissue_path is not None)
        runtime_profile = choose_runtime_profile(
            requested_grid_n=grid_n,
            requested_syn_bg_density=syn_bg_density,
            use_synth_bg=synth_enabled,
            estimated_rows=estimated_cells_rows,
        )

        runtime_notes = list(runtime_profile.notes)
        if use_synth_bg and tissue_path is None:
            runtime_notes.append(
                "Synthetic background was disabled because tissue_boundary.csv was not uploaded."
            )

        effective_bg_max_points = min(int(bg_max_points), int(runtime_profile.bg_max_points))
        effective_syn_bg_min = min(int(syn_bg_min), int(runtime_profile.syn_bg_min))
        effective_syn_bg_max = min(int(syn_bg_max), int(runtime_profile.syn_bg_max))

        summary.update(
            {
                "estimated_cells_rows": estimated_cells_rows,
                "estimated_cluster_rows": estimated_cluster_rows,
                "selected_clusters": [str(item) for item in selected_clusters],
                "label_scheme": label_scheme_description(label_scheme),
                "requested_parameters": {
                    "grid_n": int(grid_n),
                    "knn_k": int(knn_k),
                    "smooth_sigma": float(smooth_sigma),
                    "min_cells_inside": int(min_cells_inside),
                    "margin_um": float(margin_um),
                    "max_dist_threshold": float(max_dist_threshold),
                    "bg_d_min": float(bg_d_min),
                    "bg_d_max": float(bg_d_max),
                    "bg_max_points": int(bg_max_points),
                    "use_synth_bg": bool(use_synth_bg),
                    "bbox_expand_um": float(bbox_expand_um),
                    "syn_bg_density": float(syn_bg_density),
                    "syn_bg_min": int(syn_bg_min),
                    "syn_bg_max": int(syn_bg_max),
                },
                "effective_parameters": {
                    "grid_n": int(runtime_profile.grid_n),
                    "knn_k": int(knn_k),
                    "smooth_sigma": float(smooth_sigma),
                    "min_cells_inside": int(min_cells_inside),
                    "margin_um": float(margin_um),
                    "max_dist_threshold": float(max_dist_threshold),
                    "bg_d_min": float(bg_d_min),
                    "bg_d_max": float(bg_d_max),
                    "bg_max_points": int(effective_bg_max_points),
                    "use_synth_bg": bool(synth_enabled),
                    "bbox_expand_um": float(bbox_expand_um),
                    "syn_bg_density": float(runtime_profile.syn_bg_density),
                    "syn_bg_min": int(effective_syn_bg_min),
                    "syn_bg_max": int(effective_syn_bg_max),
                },
                "runtime_scale_label": runtime_profile.scale_label,
                "runtime_notes": runtime_notes,
            }
        )

        preflight_lines = [
            f"Pattern1 clusters: {', '.join(str(item) for item in selected_clusters)}",
            f"Estimated cells.parquet rows: {estimated_cells_rows if estimated_cells_rows is not None else 'unknown'}",
            f"Estimated clusters.csv rows: {estimated_cluster_rows if estimated_cluster_rows is not None else 'unknown'}",
            f"grid_n: requested {grid_n}, effective {runtime_profile.grid_n}",
            f"knn_k: {knn_k}",
            f"smooth_sigma: {smooth_sigma:.2f}",
            f"min_cells_inside: {min_cells_inside}",
            f"label scheme: {label_scheme_description(label_scheme)}",
            f"Synthetic background: {'enabled' if synth_enabled else 'disabled'}",
        ]
        preflight_lines.extend(runtime_notes)

        progress(0.18, desc="Inputs ready")
        yield emit_status(
            phase="preflight",
            run_dir=run_dir,
            lines=preflight_lines,
            summary=summary,
        )

        cfg = Pattern1IsolineConfig(
            clusters_csv=clusters_path,
            cells_parquet=cells_path,
            tissue_boundary_csv=tissue_path if synth_enabled else None,
            out_dir=output_dir,
            pattern1_clusters=selected_clusters,
            grid_n=int(runtime_profile.grid_n),
            knn_k=int(knn_k),
            smooth_sigma=float(smooth_sigma),
            margin_um=float(margin_um),
            max_dist_threshold=float(max_dist_threshold),
            bg_d_min=float(bg_d_min),
            bg_d_max=float(bg_d_max),
            bg_max_points=int(effective_bg_max_points),
            min_cells_inside=int(min_cells_inside),
            use_synth_bg=bool(synth_enabled),
            bbox_expand_um=float(bbox_expand_um),
            syn_bg_density=float(runtime_profile.syn_bg_density),
            syn_bg_min=int(effective_syn_bg_min),
            syn_bg_max=int(effective_syn_bg_max),
            label_scheme=label_scheme,
            compute_confidence_score=bool(compute_confidence_score),
            save_params_json=True,
            save_contours_npy=True,
            save_preview_png=True,
        )

        progress(0.42, desc="Running Pattern1 isoline")
        log_event(
            "Running Pattern1 isoline | "
            f"clusters={summary['selected_clusters']} | grid_n={cfg.grid_n} | knn_k={cfg.knn_k} | "
            f"smooth_sigma={cfg.smooth_sigma}"
        )
        result = run_pattern1_isoline(cfg)

        if compute_confidence_score:
            try:
                conf_res = compute_segmentation_confidence_score(
                    clusters_csv=clusters_path,
                    cells_parquet=cells_path,
                    pattern1_clusters=selected_clusters,
                )
                summary["segmentation_confidence_score"] = float(conf_res.score_mean)
                summary["segmentation_confidence_stats"] = dict(conf_res.stats)
            except Exception as exc:
                runtime_notes.append(f"Confidence score could not be computed: {exc}")

        summary.update(
            {
                "id_col_used": result.id_col_used,
                "x_col": result.x_col,
                "y_col": result.y_col,
                "n_target_cells": int(result.n_target_cells),
                "n_bg0_points": int(result.n_bg0_points),
                "n_contours": int(len(result.contours)),
                "preview_png": str(result.preview_png) if result.preview_png is not None else None,
                "params_json": str(result.params_json) if result.params_json is not None else None,
            }
        )

        progress(0.82, desc="Collecting outputs")
        archive_path, archive_note = zip_outputs(output_dir, archive_dir=run_dir)
        output_files = sorted(str(path) for path in output_dir.iterdir() if path.is_file())
        summary["output_files"] = output_files
        summary["zip_archive"] = str(archive_path) if archive_path is not None else None
        summary["run_seconds"] = round(time.perf_counter() - start_time, 2)
        summary["runtime_notes"] = runtime_notes

        final_lines = [
            f"Contours found: {len(result.contours)}",
            f"Target cells: {result.n_target_cells}",
            f"Background points: {result.n_bg0_points}",
            f"Preview PNG: {'yes' if result.preview_png is not None else 'no'}",
            f"Output files: {len(output_files)}",
            f"Elapsed seconds: {summary['run_seconds']}",
        ]
        if "segmentation_confidence_score" in summary:
            final_lines.append(
                f"Segmentation confidence score: {float(summary['segmentation_confidence_score']):.4f}"
            )
        if archive_note:
            final_lines.append(archive_note)
        elif archive_path is not None:
            final_lines.append("A ZIP archive of the output directory is available below.")

        progress(1.0, desc="Done")
        yield emit_status(
            phase="completed",
            run_dir=run_dir,
            lines=final_lines,
            summary=summary,
            preview_path=str(result.preview_png) if result.preview_png is not None else None,
            archive_path=str(archive_path) if archive_path is not None else None,
            output_files=output_files,
        )
    except Exception as exc:
        log_event(f"Pattern1 run failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


CUSTOM_CSS = """
:root {
  --app-bg: #08121d;
  --app-bg-soft: #0d1826;
  --panel-bg: rgba(13, 24, 38, 0.96);
  --panel-border: #224260;
  --panel-border-soft: rgba(122, 184, 255, 0.16);
  --text-main: #f4f8ff;
  --text-muted: #9eb2ca;
  --accent: #6ef0d4;
  --accent-cool: #77b8ff;
  --accent-warm: #ffbe72;
  --shadow-strong: 0 26px 72px rgba(0, 0, 0, 0.34);
}

html,
body,
gradio-app,
.gradio-container {
  min-height: 100%;
  color: var(--text-main) !important;
  background:
    radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 26%),
    radial-gradient(circle at top right, rgba(119, 184, 255, 0.10), transparent 30%),
    linear-gradient(180deg, #050d16 0%, #08121d 100%) !important;
  font-family: "Aptos", "Bahnschrift", "Segoe UI Variable", "Segoe UI", sans-serif;
}

.gradio-container {
  width: min(100%, 1440px) !important;
  max-width: 1440px !important;
  margin: 0 auto !important;
}

.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .panel,
.gradio-container .gr-accordion,
.gradio-container .gr-dataframe,
.gradio-container .gr-form {
  background: var(--panel-bg) !important;
  border: 1px solid var(--panel-border-soft) !important;
  box-shadow: none !important;
}

.hero-shell {
  background:
    linear-gradient(135deg, rgba(17, 30, 46, 0.98) 0%, rgba(11, 19, 31, 0.96) 48%, rgba(16, 32, 47, 0.98) 100%);
  border: 1px solid rgba(110, 240, 212, 0.18);
  border-radius: 28px;
  padding: 30px 32px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-strong);
}

.hero-kicker {
  display: inline-flex;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(110, 240, 212, 0.10);
  border: 1px solid rgba(110, 240, 212, 0.18);
  color: var(--accent);
  font-size: 0.86rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 16px;
}

.hero-shell h1 {
  margin: 0;
  font-size: 2.9rem;
  line-height: 1.04;
  letter-spacing: -0.03em;
  color: #f7fbff;
}

.hero-shell p {
  margin: 14px 0 0 0;
  max-width: 980px;
  color: #d4e2f2;
  font-size: 1.06rem;
  line-height: 1.7;
}

.hero-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 20px;
}

.hero-metrics span {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #eaf3ff;
  font-size: 0.92rem;
}

.guide-shell {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.guide-card {
  background: rgba(15, 27, 42, 0.98);
  border: 1px solid rgba(122, 184, 255, 0.16);
  border-radius: 22px;
  padding: 18px;
  min-height: 180px;
}

.guide-step {
  color: var(--accent-warm);
  font-size: 0.84rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}

.guide-card h3 {
  margin: 0 0 10px 0;
  color: #f7fbff;
  font-size: 1.08rem;
}

.guide-card p {
  margin: 0;
  color: #cad8e8;
  line-height: 1.65;
  font-size: 0.96rem;
}

.app-note {
  margin-bottom: 18px;
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(13, 24, 38, 0.96);
  border: 1px solid rgba(255, 190, 114, 0.18);
  color: #eef5ff;
  line-height: 1.7;
}

.app-note strong {
  color: var(--accent-warm);
}

.micro-guide {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(13, 24, 38, 0.98);
  border: 1px solid rgba(122, 184, 255, 0.18);
  color: #dbe8f7;
  line-height: 1.64;
  margin-bottom: 14px;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #0a1523 !important;
  color: var(--text-main) !important;
  border: 1px solid #28445f !important;
}

.gradio-container .gr-button {
  border-radius: 14px !important;
  font-weight: 700 !important;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, #6ef0d4 0%, #3cc8ff 100%) !important;
  color: #04111b !important;
  border: none !important;
}

.gradio-container .gr-button-secondary {
  background: #14263a !important;
  color: var(--text-main) !important;
  border: 1px solid #28445f !important;
}

#left-rail,
#right-rail {
  gap: 14px;
}

#pattern1-clusters textarea,
#status-text textarea {
  min-height: 180px !important;
}

@media (max-width: 1120px) {
  .guide-shell {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 700px) {
  .hero-shell {
    padding: 24px 20px;
  }
  .hero-shell h1 {
    font-size: 2.2rem;
  }
}
"""


GLOBAL_HEAD = """
<style>
  html,
  body,
  gradio-app {
    min-height: 100%;
    background:
      radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 26%),
      radial-gradient(circle at top right, rgba(119, 184, 255, 0.10), transparent 30%),
      linear-gradient(180deg, #050d16 0%, #08121d 100%) !important;
  }
</style>
"""


with gr.Blocks(
    title=APP_NAME,
    css=CUSTOM_CSS,
    head=GLOBAL_HEAD,
    fill_width=True,
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
) as demo:
    ensure_workdirs()

    gr.HTML(
        f"""
        <div class="hero-shell">
          <div class="hero-kicker">SciLifeLab Serve app | HistoSeg Pattern1</div>
          <h1>{APP_NAME}</h1>
          <p>{APP_DESCRIPTION}</p>
          <div class="hero-metrics">
            <span>Original Pattern1 KNN contour workflow</span>
            <span>Direct control of grid_n, knn_k, sigma</span>
            <span>Preview PNG + params.json + contour .npy downloads</span>
          </div>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="guide-shell">
          <div class="guide-card">
            <div class="guide-step">Step 1</div>
            <h3>Upload Xenium inputs</h3>
            <p>
              Provide <code>cells.parquet</code> and <code>clusters.csv</code>. Upload
              <code>tissue_boundary.csv</code> only if you want synthetic background points.
            </p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 2</div>
            <h3>Choose Pattern1 clusters</h3>
            <p>
              Enter the cluster IDs that define the Pattern1 signal of interest, for example
              <code>10,23,19</code>.
            </p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 3</div>
            <h3>Tune KNN and sigma</h3>
            <p>
              Adjust <code>grid_n</code>, <code>knn_k</code>, <code>smooth_sigma</code>, and
              <code>min_cells_inside</code>, then run the original contour workflow.
            </p>
          </div>
        </div>
        <div class="app-note">
          <strong>What this app is for.</strong> This version is intentionally not the multi-structure workflow.
          It is a single Pattern1 isoline tool for the original KNN-plus-Gaussian-smoothing contour algorithm.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-rail"):
            gr.HTML(
                """
                <div class="micro-guide">
                  In a standard Xenium <code>outs</code> directory, <code>cells.parquet</code> is usually in
                  the root of <code>outs</code>, while one common <code>clusters.csv</code> path is
                  <code>outs\\analysis\\clustering\\gene_expression_graphclust\\clusters.csv</code>.
                </div>
                """
            )
            cells_parquet = gr.File(
                label="Cell coordinates (cells.parquet)",
                file_types=[".parquet"],
                type="filepath",
            )
            clusters_csv = gr.File(
                label="Cluster assignments (clusters.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            tissue_boundary_csv = gr.File(
                label="Tissue boundary (optional: tissue_boundary.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            pattern1_clusters = gr.Textbox(
                label="Pattern1 cluster IDs",
                value=DEFAULT_PATTERN1,
                lines=4,
                placeholder="10,23,19",
                info="Comma-separated cluster IDs. Newlines and semicolons are also accepted.",
                elem_id="pattern1-clusters",
            )
            grid_n = gr.Slider(
                label="grid_n (mesh resolution)",
                minimum=300,
                maximum=1600,
                step=50,
                value=900,
            )
            knn_k = gr.Slider(
                label="knn_k (nearest neighbors)",
                minimum=5,
                maximum=120,
                step=1,
                value=30,
            )
            smooth_sigma = gr.Slider(
                label="smooth_sigma (Gaussian sigma)",
                minimum=0.5,
                maximum=12.0,
                step=0.25,
                value=5.0,
            )
            min_cells_inside = gr.Slider(
                label="min_cells_inside",
                minimum=1,
                maximum=300,
                step=1,
                value=10,
            )

            with gr.Accordion("Advanced parameters", open=False):
                margin_um = gr.Slider(label="margin_um", minimum=0, maximum=500, step=10, value=50)
                max_dist_threshold = gr.Slider(
                    label="max_dist_threshold",
                    minimum=20,
                    maximum=600,
                    step=10,
                    value=200,
                )
                bg_d_min = gr.Slider(label="bg_d_min", minimum=0, maximum=100, step=1, value=20)
                bg_d_max = gr.Slider(label="bg_d_max", minimum=20, maximum=600, step=5, value=250)
                bg_max_points = gr.Slider(
                    label="bg_max_points",
                    minimum=5000,
                    maximum=100000,
                    step=1000,
                    value=60000,
                )
                label_scheme = gr.Radio(
                    label="Label scheme",
                    choices=[
                        ("Selected Pattern1 clusters are inside / positive", "p1_is_one"),
                        ("Selected Pattern1 clusters are outside / inverted", "p1_is_zero"),
                    ],
                    value="p1_is_one",
                )
                use_synth_bg = gr.Checkbox(
                    label="Use synthetic background points",
                    value=True,
                )
                bbox_expand_um = gr.Slider(label="bbox_expand_um", minimum=0, maximum=400, step=10, value=100)
                syn_bg_density = gr.Slider(
                    label="syn_bg_density",
                    minimum=0.0005,
                    maximum=0.0200,
                    step=0.0005,
                    value=0.0100,
                )
                syn_bg_min = gr.Slider(label="syn_bg_min", minimum=1000, maximum=50000, step=1000, value=20000)
                syn_bg_max = gr.Slider(label="syn_bg_max", minimum=5000, maximum=150000, step=5000, value=120000)
                compute_confidence_score = gr.Checkbox(
                    label="Compute segmentation confidence score",
                    value=False,
                )

            run_button = gr.Button("Run Pattern1 contour analysis", variant="primary")

        with gr.Column(scale=1, elem_id="right-rail"):
            status_text = gr.Textbox(label="Run status", lines=10, elem_id="status-text")
            preview_image = gr.Image(
                label="Pattern1 preview PNG",
                type="filepath",
                interactive=False,
            )
            summary_json = gr.JSON(label="Run summary")
            output_archive = gr.File(label="Download all outputs as ZIP", file_count="single")
            output_files = gr.File(label="Download raw output files", file_count="multiple")

    run_button.click(
        fn=run_analysis,
        inputs=[
            cells_parquet,
            clusters_csv,
            tissue_boundary_csv,
            pattern1_clusters,
            grid_n,
            knn_k,
            smooth_sigma,
            min_cells_inside,
            margin_um,
            max_dist_threshold,
            bg_d_min,
            bg_d_max,
            bg_max_points,
            label_scheme,
            use_synth_bg,
            bbox_expand_um,
            syn_bg_density,
            syn_bg_min,
            syn_bg_max,
            compute_confidence_score,
        ],
        outputs=[status_text, preview_image, summary_json, output_archive, output_files],
    )


def main() -> None:
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
    )


if __name__ == "__main__":
    main()
