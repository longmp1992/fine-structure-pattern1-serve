from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.neighbors import KNeighborsRegressor


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
        Pattern1IsolineResult,
        compute_segmentation_confidence_score,
    )
    from histoseg.contour.pattern1_isoline import (
        _normalize_cluster_label,
        _validate_label_scheme,
        align_clusters_with_cells,
        extract_contour_paths,
        filter_loops_by_cell_count,
        generate_synthetic_bg_in_bbox,
        make_mesh_from_xy,
        sample_background_from_other_cells_plus_synth,
        tissue_mask_from_xy,
    )

    HISTOSEG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    Pattern1IsolineConfig = None  # type: ignore[assignment]
    Pattern1IsolineResult = None  # type: ignore[assignment]
    compute_segmentation_confidence_score = None  # type: ignore[assignment]
    _normalize_cluster_label = None  # type: ignore[assignment]
    _validate_label_scheme = None  # type: ignore[assignment]
    align_clusters_with_cells = None  # type: ignore[assignment]
    extract_contour_paths = None  # type: ignore[assignment]
    filter_loops_by_cell_count = None  # type: ignore[assignment]
    generate_synthetic_bg_in_bbox = None  # type: ignore[assignment]
    make_mesh_from_xy = None  # type: ignore[assignment]
    sample_background_from_other_cells_plus_synth = None  # type: ignore[assignment]
    tissue_mask_from_xy = None  # type: ignore[assignment]
    HISTOSEG_IMPORT_ERROR = str(exc)


APP_NAME = "Fine Structure Pattern1 Explorer"
APP_DESCRIPTION = (
    "A SciLifeLab Serve Gradio app for the original HistoSeg Pattern1 contour workflow: "
    "use browser uploads or mounted project-storage paths for Xenium cells.parquet together "
    "with either clusters.csv or transcript.parquet, choose Pattern1 cluster IDs or a single "
    "gene, and tune the KNN, Gaussian sigma, and isoline level parameters before generating "
    "isoline contours. "
    "Blank grid cells inside the tissue mask are also modeled as background."
)
DEFAULT_PATTERN1 = "10,23,19"
DEFAULT_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")
SOURCE_MODE_CLUSTER = "cluster_ids"
SOURCE_MODE_TRANSCRIPT = "transcript_gene"
INPUT_MODE_UPLOAD = "upload_files"
INPUT_MODE_STORAGE = "mounted_storage"


@dataclass(frozen=True)
class RuntimeProfile:
    grid_n: int
    bg_max_points: int
    syn_bg_density: float
    syn_bg_min: int
    syn_bg_max: int
    scale_label: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BlankGridBackgroundStats:
    traditional_bg_points: int
    empty_grid_bg_points: int
    empty_grid_candidate_points: int
    occupied_grid_count: int


@dataclass(frozen=True)
class TranscriptInputSchema:
    gene_col: str
    x_col: str
    y_col: str
    cell_id_col: str | None


@dataclass(frozen=True)
class ResolvedInputFile:
    path: Path
    source: str
    original: str


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


def configured_input_roots() -> tuple[Path, ...]:
    raw_env = os.environ.get("APP_ALLOWED_INPUT_ROOTS", "")
    candidates = [item.strip() for item in raw_env.split(os.pathsep) if item.strip()]
    if not candidates:
        candidates = ["/home/data", "/srv/shiny-server/data"]

    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        roots.append(path)
    return tuple(roots)


ALLOWED_INPUT_ROOTS = configured_input_roots()
PRIMARY_INPUT_ROOT = ALLOWED_INPUT_ROOTS[0] if ALLOWED_INPUT_ROOTS else Path("/home/data")
INPUT_ROOTS_LABEL = ", ".join(str(path) for path in ALLOWED_INPUT_ROOTS)


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


def parse_transcript_gene(raw: str | None) -> str:
    gene = str(raw or "").strip()
    if not gene:
        raise ValueError("Transcript gene cannot be empty. Please choose or type a gene from transcript.parquet.")
    return gene


def safe_filename_component(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(raw).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "pattern1"


def path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_storage_file(storage_path: str | None, *, label: str, suffixes: tuple[str, ...]) -> Path | None:
    raw = str(storage_path or "").strip()
    if not raw:
        return None

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PRIMARY_INPUT_ROOT / candidate

    try:
        resolved = candidate.resolve()
    except OSError as exc:
        raise ValueError(f"Could not resolve the mounted-storage path for {label}: {exc}") from exc

    resolved_roots = tuple(root.expanduser().resolve(strict=False) for root in ALLOWED_INPUT_ROOTS)
    if not any(path_is_within(resolved, root) for root in resolved_roots):
        raise ValueError(
            f"{label} must stay inside the configured mounted storage roots: {INPUT_ROOTS_LABEL}"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"{label} was not found at mounted-storage path: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"{label} must point to a file, not a directory: {resolved}")
    if suffixes and resolved.suffix.lower() not in suffixes:
        suffix_label = ", ".join(suffixes)
        raise ValueError(f"{label} must use one of these suffixes: {suffix_label}")
    return resolved


def parquet_column_names(parquet_path: Path) -> list[str]:
    if pq is not None:
        return [str(name) for name in pq.ParquetFile(parquet_path).schema.names]
    return [str(col) for col in pd.read_parquet(parquet_path).columns]


def first_present_column(columns: list[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {label} column. Available columns: {columns}")


def detect_transcript_input_schema(transcript_parquet: Path) -> TranscriptInputSchema:
    columns = parquet_column_names(transcript_parquet)
    gene_col = first_present_column(
        columns,
        ("feature_name", "gene", "gene_name", "feature_id"),
        "transcript gene",
    )
    x_col = first_present_column(
        columns,
        ("x_location", "x", "X", "x_centroid"),
        "transcript x-coordinate",
    )
    y_col = first_present_column(
        columns,
        ("y_location", "y", "Y", "y_centroid"),
        "transcript y-coordinate",
    )
    cell_id_col = next((candidate for candidate in ("cell_id", "CellID", "cell", "cell_barcode") if candidate in columns), None)
    return TranscriptInputSchema(
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        cell_id_col=cell_id_col,
    )


def detect_cell_coordinate_columns(cells_parquet: Path) -> tuple[str, str, str]:
    columns = parquet_column_names(cells_parquet)
    id_col = first_present_column(columns, ("cell_id", "CellID", "Barcode"), "cell ID")
    x_col = first_present_column(columns, ("x_centroid", "x", "X", "x_location"), "cell x-coordinate")
    y_col = first_present_column(columns, ("y_centroid", "y", "Y", "y_location"), "cell y-coordinate")
    return id_col, x_col, y_col


def read_parquet_subset(parquet_path: Path, columns: list[str]) -> pd.DataFrame:
    unique_columns = list(dict.fromkeys(columns))
    if pq is not None:
        return pq.read_table(parquet_path, columns=unique_columns).to_pandas()
    return pd.read_parquet(parquet_path, columns=unique_columns)


def list_transcript_genes(transcript_parquet: Path) -> list[str]:
    schema = detect_transcript_input_schema(transcript_parquet)
    if pq is not None:
        import pyarrow.compute as pc

        gene_array = pq.read_table(transcript_parquet, columns=[schema.gene_col]).column(schema.gene_col)
        raw_values = pc.unique(gene_array).to_pylist()
    else:
        gene_df = read_parquet_subset(transcript_parquet, [schema.gene_col])
        raw_values = gene_df[schema.gene_col].dropna().tolist()
    genes = sorted({str(value).strip() for value in raw_values if value is not None and str(value).strip()})
    return genes


def pick_transcript_source(
    input_mode: str,
    transcript_upload: object | None,
    transcript_storage_path: str | None,
) -> Path | None:
    if input_mode == INPUT_MODE_STORAGE:
        return resolve_storage_file(
            transcript_storage_path,
            label="transcript.parquet",
            suffixes=(".parquet",),
        )

    if transcript_upload is None:
        return None

    transcript_path = Path(str(transcript_upload))
    if not transcript_path.exists():
        return None
    return transcript_path


def load_transcript_gene_options(
    input_mode: str,
    transcript_upload: object | None,
    transcript_storage_path: str | None,
):
    try:
        transcript_path = pick_transcript_source(input_mode, transcript_upload, transcript_storage_path)
    except Exception:
        return gr.update(choices=[], value=None, interactive=True)
    if transcript_path is None:
        return gr.update(choices=[], value=None, interactive=True)

    try:
        genes = list_transcript_genes(transcript_path)
    except Exception:
        return gr.update(choices=[], value=None, interactive=True)

    if not genes:
        return gr.update(choices=[], value=None, interactive=True)
    return gr.update(choices=genes, value=genes[0], interactive=True)


def update_visibility(input_mode: str, source_mode: str):
    use_cluster_mode = source_mode != SOURCE_MODE_TRANSCRIPT
    use_upload_mode = input_mode != INPUT_MODE_STORAGE
    return (
        gr.update(visible=use_upload_mode),
        gr.update(visible=use_upload_mode and use_cluster_mode),
        gr.update(visible=use_upload_mode and not use_cluster_mode),
        gr.update(visible=use_upload_mode),
        gr.update(visible=not use_upload_mode),
        gr.update(visible=not use_upload_mode and use_cluster_mode),
        gr.update(visible=not use_upload_mode and not use_cluster_mode),
        gr.update(visible=not use_upload_mode),
        gr.update(visible=use_cluster_mode),
        gr.update(visible=not use_cluster_mode),
        gr.update(visible=use_cluster_mode),
        gr.update(visible=use_cluster_mode),
    )


def stage_uploaded_file(uploaded: object | None, target_dir: Path) -> Path | None:
    if uploaded is None:
        return None
    source = Path(str(uploaded))
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    destination = target_dir / source.name
    shutil.copy2(source, destination)
    return destination


def resolve_input_file(
    *,
    input_mode: str,
    uploaded: object | None,
    storage_path: str | None,
    target_dir: Path,
    label: str,
    suffixes: tuple[str, ...],
) -> ResolvedInputFile | None:
    if input_mode == INPUT_MODE_STORAGE:
        resolved = resolve_storage_file(storage_path, label=label, suffixes=suffixes)
        if resolved is None:
            return None
        return ResolvedInputFile(path=resolved, source=INPUT_MODE_STORAGE, original=str(resolved))

    staged = stage_uploaded_file(uploaded, target_dir)
    if staged is None:
        return None
    return ResolvedInputFile(path=staged, source=INPUT_MODE_UPLOAD, original=str(uploaded))


def resolve_inputs(
    *,
    input_mode: str,
    cells_upload: object | None,
    cells_storage_path: str | None,
    clusters_upload: object | None,
    clusters_storage_path: str | None,
    tissue_upload: object | None,
    tissue_storage_path: str | None,
    transcript_upload: object | None,
    transcript_storage_path: str | None,
    target_dir: Path,
) -> tuple[ResolvedInputFile | None, ResolvedInputFile | None, ResolvedInputFile | None, ResolvedInputFile | None]:
    cells_path = resolve_input_file(
        input_mode=input_mode,
        uploaded=cells_upload,
        storage_path=cells_storage_path,
        target_dir=target_dir,
        label="cells.parquet",
        suffixes=(".parquet",),
    )
    clusters_path = resolve_input_file(
        input_mode=input_mode,
        uploaded=clusters_upload,
        storage_path=clusters_storage_path,
        target_dir=target_dir,
        label="clusters.csv",
        suffixes=(".csv",),
    )
    tissue_path = resolve_input_file(
        input_mode=input_mode,
        uploaded=tissue_upload,
        storage_path=tissue_storage_path,
        target_dir=target_dir,
        label="tissue_boundary.csv",
        suffixes=(".csv",),
    )
    transcript_path = resolve_input_file(
        input_mode=input_mode,
        uploaded=transcript_upload,
        storage_path=transcript_storage_path,
        target_dir=target_dir,
        label="transcript.parquet",
        suffixes=(".parquet",),
    )
    return cells_path, clusters_path, tissue_path, transcript_path


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


def grid_step_um(xx: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
    step_x = float(abs(xx[0, 1] - xx[0, 0])) if xx.shape[1] > 1 else 1.0
    step_y = float(abs(yy[1, 0] - yy[0, 0])) if yy.shape[0] > 1 else 1.0
    return step_x, step_y


def build_cell_occupancy_mask(all_xy: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    occupied = np.zeros_like(xx, dtype=bool)
    if all_xy.size == 0:
        return occupied

    x0 = float(xx[0, 0])
    y0 = float(yy[0, 0])
    step_x, step_y = grid_step_um(xx, yy)

    x_idx = np.rint((all_xy[:, 0] - x0) / step_x).astype(int)
    y_idx = np.rint((all_xy[:, 1] - y0) / step_y).astype(int)
    x_idx = np.clip(x_idx, 0, xx.shape[1] - 1)
    y_idx = np.clip(y_idx, 0, yy.shape[0] - 1)
    occupied[y_idx, x_idx] = True
    return occupied


def sample_empty_grid_background(
    *,
    all_xy: np.ndarray,
    target_xy: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    tissue_mask: np.ndarray,
    d_max: float,
    margin_um: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Use blank grid cells inside the tissue mask as explicit background candidates."""
    occupied = build_cell_occupancy_mask(all_xy, xx, yy)
    empty_mask = tissue_mask & ~occupied

    xmin, ymin = target_xy.min(axis=0)
    xmax, ymax = target_xy.max(axis=0)
    pad = float(d_max) + float(margin_um)
    bbox_mask = (
        (xx >= xmin - pad)
        & (xx <= xmax + pad)
        & (yy >= ymin - pad)
        & (yy <= ymax + pad)
    )
    empty_mask &= bbox_mask

    empty_xy = np.c_[xx[empty_mask], yy[empty_mask]].astype(float)
    candidate_count = int(len(empty_xy))
    occupied_count = int(occupied.sum())
    if candidate_count == 0:
        return np.empty((0, 2), dtype=float), candidate_count, occupied_count

    target_tree = cKDTree(np.asarray(target_xy, dtype=float))
    target_dist, _ = target_tree.query(empty_xy, k=1)
    empty_xy = empty_xy[target_dist <= float(d_max)]
    if len(empty_xy) == 0:
        return np.empty((0, 2), dtype=float), candidate_count, occupied_count

    rng = np.random.default_rng(seed)
    if len(empty_xy) > max_points:
        idx = rng.choice(len(empty_xy), size=max_points, replace=False)
        empty_xy = empty_xy[idx]
    return empty_xy, candidate_count, occupied_count


def combine_background_sources(
    *,
    empty_grid_bg_xy: np.ndarray,
    traditional_bg_xy: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    """Prioritize blank grid background points, then fill the remaining budget."""
    rng = np.random.default_rng(seed)
    if len(empty_grid_bg_xy) >= max_points:
        idx = rng.choice(len(empty_grid_bg_xy), size=max_points, replace=False)
        return empty_grid_bg_xy[idx]

    if len(traditional_bg_xy) == 0:
        return empty_grid_bg_xy

    remaining = max_points - len(empty_grid_bg_xy)
    if len(traditional_bg_xy) > remaining:
        idx = rng.choice(len(traditional_bg_xy), size=remaining, replace=False)
        traditional_bg_xy = traditional_bg_xy[idx]

    if len(empty_grid_bg_xy) == 0:
        return traditional_bg_xy
    return np.vstack([empty_grid_bg_xy, traditional_bg_xy])


def sample_cell_background_plus_synth(
    *,
    all_xy: np.ndarray,
    synthetic_bg_xy: np.ndarray | None,
    max_points: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cell_bg_xy = np.asarray(all_xy, dtype=float)
    if len(cell_bg_xy) > max_points:
        idx = rng.choice(len(cell_bg_xy), size=max_points, replace=False)
        return cell_bg_xy[idx]

    if synthetic_bg_xy is None or len(synthetic_bg_xy) == 0:
        return cell_bg_xy

    remaining = max_points - len(cell_bg_xy)
    if remaining <= 0:
        return cell_bg_xy

    synth_xy = np.asarray(synthetic_bg_xy, dtype=float)
    if len(synth_xy) > remaining:
        idx = rng.choice(len(synth_xy), size=remaining, replace=False)
        synth_xy = synth_xy[idx]

    if len(cell_bg_xy) == 0:
        return synth_xy
    return np.vstack([cell_bg_xy, synth_xy])


def run_pattern1_isoline_blank_grid(
    cfg: Pattern1IsolineConfig,
) -> tuple[Pattern1IsolineResult, BlankGridBackgroundStats]:
    """Run Pattern1 isoline with blank grid cells treated as explicit background."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_scheme = _validate_label_scheme(cfg.label_scheme)

    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        cfg.clusters_csv,
        cfg.cells_parquet,
        barcode_col=cfg.barcode_col,
        cluster_col=cfg.cluster_col,
    )

    merged = merged.copy()
    merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
    merged = merged.loc[merged["cluster"] != ""].copy()

    p1 = {_normalize_cluster_label(x) for x in cfg.pattern1_clusters}
    p1 = {x for x in p1 if x != ""}
    if len(p1) == 0:
        raise ValueError("pattern1_clusters is empty after normalization.")

    merged["_is_p1"] = merged["cluster"].isin(p1)
    p1_df = merged.loc[merged["_is_p1"], [id_col_used, x_col, y_col]].copy()
    if len(p1_df) < 10:
        raise RuntimeError(f"pattern1 cells too few after merge: {len(p1_df)}")

    target_ids = set(p1_df[id_col_used].astype(str))
    target_xy = p1_df[[x_col, y_col]].to_numpy(float)
    all_xy = merged[[x_col, y_col]].to_numpy(float)

    syn_bg_xy: np.ndarray | None = None
    if cfg.use_synth_bg:
        if cfg.tissue_boundary_csv is None:
            raise ValueError("use_synth_bg=True but tissue_boundary_csv was not provided.")
        boundary_xy = pd.read_csv(cfg.tissue_boundary_csv)
        if {"x", "y"}.issubset(boundary_xy.columns):
            boundary_xy_np = boundary_xy[["x", "y"]].to_numpy(float)
        elif {"X", "Y"}.issubset(boundary_xy.columns):
            boundary_xy_np = boundary_xy[["X", "Y"]].to_numpy(float)
        else:
            raise ValueError(
                f"tissue_boundary.csv must contain x,y or X,Y columns, got {list(boundary_xy.columns)}"
            )
        syn_bg_xy = generate_synthetic_bg_in_bbox(
            boundary_xy_np,
            expand_um=cfg.bbox_expand_um,
            density=cfg.syn_bg_density,
            min_n=cfg.syn_bg_min,
            max_n=cfg.syn_bg_max,
            seed=cfg.random_state,
        )

    xx, yy, grid = make_mesh_from_xy(
        target_xy,
        grid_n=cfg.grid_n,
        pad_fraction=cfg.pad_fraction,
        margin_um=cfg.margin_um,
    )
    tissue_mask = tissue_mask_from_xy(all_xy, xx, yy, max_dist_threshold=cfg.max_dist_threshold)

    traditional_bg_xy = sample_background_from_other_cells_plus_synth(
        cells_df=merged.rename(columns={id_col_used: "tmp_id"}),
        synthetic_bg_xy=syn_bg_xy,
        target_ids={str(x) for x in target_ids},
        target_xy=target_xy,
        cell_id_col="tmp_id",
        x_col=x_col,
        y_col=y_col,
        d_min=cfg.bg_d_min,
        d_max=cfg.bg_d_max,
        max_points=cfg.bg_max_points,
        seed=cfg.random_state,
        margin_um=cfg.margin_um,
    )

    empty_grid_bg_xy, empty_grid_candidates, occupied_grid_count = sample_empty_grid_background(
        all_xy=all_xy,
        target_xy=target_xy,
        xx=xx,
        yy=yy,
        tissue_mask=tissue_mask,
        d_max=cfg.bg_d_max,
        margin_um=cfg.margin_um,
        max_points=cfg.bg_max_points,
        seed=cfg.random_state,
    )
    bg0_xy = combine_background_sources(
        empty_grid_bg_xy=empty_grid_bg_xy,
        traditional_bg_xy=traditional_bg_xy,
        max_points=cfg.bg_max_points,
        seed=cfg.random_state,
    )
    if len(bg0_xy) == 0:
        raise RuntimeError(
            "No background points were sampled. Try relaxing bg_d_max, lowering grid_n, or uploading tissue_boundary.csv."
        )

    X_train = np.vstack([bg0_xy, target_xy])
    if label_scheme == "p1_is_one":
        y_train = np.hstack([np.zeros(len(bg0_xy)), np.ones(len(target_xy))])
    else:
        y_train = np.hstack([np.ones(len(bg0_xy)), np.zeros(len(target_xy))])

    reg = KNeighborsRegressor(n_neighbors=cfg.knn_k, weights="distance")
    reg.fit(X_train, y_train)

    prob = reg.predict(grid).reshape(xx.shape)
    prob_smooth = gaussian_filter(prob, sigma=cfg.smooth_sigma)
    prob_smooth_masked = prob_smooth.copy()
    prob_smooth_masked[~tissue_mask] = np.nan

    verts_list = extract_contour_paths(xx, yy, prob_smooth_masked, level=cfg.isoline_level)
    verts_list = filter_loops_by_cell_count(verts_list, target_xy, min_cells_inside=cfg.min_cells_inside)
    if len(verts_list) == 0:
        raise RuntimeError(
            "No isoline found.\n"
            "Suggestions: lower min_cells_inside, raise knn_k, lower smooth_sigma for finer structures, or raise grid_n."
        )

    conf_score: float | None = None
    conf_stats: dict[str, object] | None = None
    if cfg.compute_confidence_score:
        conf_res = compute_segmentation_confidence_score(
            clusters_csv=cfg.clusters_csv,
            cells_parquet=cfg.cells_parquet,
            pattern1_clusters=cfg.pattern1_clusters,
        )
        conf_score = float(conf_res.score_mean)
        conf_stats = dict(conf_res.stats)

    params_path: Path | None = None
    if cfg.save_params_json:
        params = {
            **cfg.__dict__,
            "id_col_used": id_col_used,
            "x_col": x_col,
            "y_col": y_col,
            "n_target_cells": int(len(target_xy)),
            "n_bg0": int(len(bg0_xy)),
            "n_contours": int(len(verts_list)),
            "label_scheme": label_scheme,
            "segmentation_confidence_score": conf_score,
            "segmentation_confidence_stats": conf_stats,
            "traditional_bg_points": int(len(traditional_bg_xy)),
            "empty_grid_bg_points": int(len(empty_grid_bg_xy)),
            "empty_grid_candidate_points": int(empty_grid_candidates),
            "occupied_grid_count": int(occupied_grid_count),
            "blank_grid_background_enabled": True,
        }
        params_path = out_dir / "params.json"
        params_path.write_text(
            json.dumps(params, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    if cfg.save_contours_npy:
        for i, verts in enumerate(verts_list):
            np.save(out_dir / f"pattern1_isoline_{cfg.isoline_level:g}_{i}.npy", verts)

    preview_path: Path | None = None
    if cfg.save_preview_png:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        if len(traditional_bg_xy) > 0:
            plt.scatter(
                traditional_bg_xy[:, 0],
                traditional_bg_xy[:, 1],
                s=1,
                alpha=0.03,
                label="traditional bg",
            )
        if len(empty_grid_bg_xy) > 0:
            plt.scatter(
                empty_grid_bg_xy[:, 0],
                empty_grid_bg_xy[:, 1],
                s=1,
                alpha=0.05,
                label="blank grid bg",
            )
        plt.scatter(target_xy[:, 0], target_xy[:, 1], s=3, alpha=0.85, label="pattern1 cells")
        for verts in verts_list:
            plt.plot(verts[:, 0], verts[:, 1], linewidth=2)
        plt.gca().set_aspect("equal")

        title = (
            f"Pattern1 segmentation | isoline={cfg.isoline_level:g} | contours={len(verts_list)} "
            f"| blank_grid_bg={len(empty_grid_bg_xy)}"
        )
        if conf_score is not None:
            title += f" | confidence(mean)={conf_score:.4f}"
        plt.title(title)
        plt.legend(frameon=False)
        plt.tight_layout()
        preview_path = out_dir / f"pattern1_isoline_{cfg.isoline_level:g}.png"
        plt.savefig(preview_path, dpi=300)
        plt.close()

    result = Pattern1IsolineResult(
        out_dir=out_dir,
        id_col_used=id_col_used,
        x_col=x_col,
        y_col=y_col,
        n_target_cells=int(len(target_xy)),
        n_bg0_points=int(len(bg0_xy)),
        contours=list(verts_list),
        label_scheme=label_scheme,
        segmentation_confidence_score=conf_score,
        segmentation_confidence_stats=conf_stats,
        params_json=params_path,
        preview_png=preview_path,
    )
    stats = BlankGridBackgroundStats(
        traditional_bg_points=int(len(traditional_bg_xy)),
        empty_grid_bg_points=int(len(empty_grid_bg_xy)),
        empty_grid_candidate_points=int(empty_grid_candidates),
        occupied_grid_count=int(occupied_grid_count),
    )
    return result, stats


def run_transcript_gene_isoline_blank_grid(
    *,
    cells_parquet: Path,
    transcript_parquet: Path,
    transcript_gene: str,
    tissue_boundary_csv: Path | None,
    out_dir: Path,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    isoline_level: float,
    margin_um: float,
    max_dist_threshold: float,
    bg_max_points: int,
    bg_d_max: float,
    min_cells_inside: int,
    use_synth_bg: bool,
    bbox_expand_um: float,
    syn_bg_density: float,
    syn_bg_min: int,
    syn_bg_max: int,
    random_state: int = 0,
    pad_fraction: float = 0.02,
) -> tuple[Pattern1IsolineResult, BlankGridBackgroundStats, dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_id_col, cell_x_col, cell_y_col = detect_cell_coordinate_columns(cells_parquet)
    cells_df = read_parquet_subset(cells_parquet, [cell_id_col, cell_x_col, cell_y_col]).dropna(
        subset=[cell_x_col, cell_y_col]
    )
    all_xy = cells_df[[cell_x_col, cell_y_col]].to_numpy(float)
    if len(all_xy) < 10:
        raise RuntimeError(f"Too few cells found in cells.parquet after filtering: {len(all_xy)}")

    transcript_schema = detect_transcript_input_schema(transcript_parquet)
    transcript_columns = [transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]
    if transcript_schema.cell_id_col is not None:
        transcript_columns.append(transcript_schema.cell_id_col)
    selected_gene = parse_transcript_gene(transcript_gene)
    transcript_rows_total = safe_count_parquet_rows(transcript_parquet)
    if pq is not None:
        target_df = pq.read_table(
            transcript_parquet,
            columns=transcript_columns,
            filters=[(transcript_schema.gene_col, "=", selected_gene)],
        ).to_pandas()
    else:
        transcripts_df = read_parquet_subset(transcript_parquet, transcript_columns).dropna(
            subset=[transcript_schema.gene_col, transcript_schema.x_col, transcript_schema.y_col]
        )
        gene_series = transcripts_df[transcript_schema.gene_col].astype(str).str.strip()
        target_mask = gene_series.str.casefold() == selected_gene.casefold()
        target_df = transcripts_df.loc[target_mask].copy()
    target_df = target_df.dropna(subset=[transcript_schema.x_col, transcript_schema.y_col])
    target_xy = target_df[[transcript_schema.x_col, transcript_schema.y_col]].to_numpy(float)
    if len(target_xy) < 10:
        raise RuntimeError(
            f"Too few transcripts found for gene '{selected_gene}' after filtering: {len(target_xy)}"
        )

    syn_bg_xy: np.ndarray | None = None
    if use_synth_bg:
        if tissue_boundary_csv is None:
            raise ValueError("use_synth_bg=True but tissue_boundary_csv was not provided.")
        boundary_xy = pd.read_csv(tissue_boundary_csv)
        if {"x", "y"}.issubset(boundary_xy.columns):
            boundary_xy_np = boundary_xy[["x", "y"]].to_numpy(float)
        elif {"X", "Y"}.issubset(boundary_xy.columns):
            boundary_xy_np = boundary_xy[["X", "Y"]].to_numpy(float)
        else:
            raise ValueError(
                f"tissue_boundary.csv must contain x,y or X,Y columns, got {list(boundary_xy.columns)}"
            )
        syn_bg_xy = generate_synthetic_bg_in_bbox(
            boundary_xy_np,
            expand_um=bbox_expand_um,
            density=syn_bg_density,
            min_n=syn_bg_min,
            max_n=syn_bg_max,
            seed=random_state,
        )

    xx, yy, grid = make_mesh_from_xy(
        target_xy,
        grid_n=grid_n,
        pad_fraction=pad_fraction,
        margin_um=margin_um,
    )
    tissue_mask = tissue_mask_from_xy(all_xy, xx, yy, max_dist_threshold=max_dist_threshold)

    traditional_bg_xy = sample_cell_background_plus_synth(
        all_xy=all_xy,
        synthetic_bg_xy=syn_bg_xy,
        max_points=bg_max_points,
        seed=random_state,
    )

    empty_grid_bg_xy, empty_grid_candidates, occupied_grid_count = sample_empty_grid_background(
        all_xy=all_xy,
        target_xy=target_xy,
        xx=xx,
        yy=yy,
        tissue_mask=tissue_mask,
        d_max=bg_d_max,
        margin_um=margin_um,
        max_points=bg_max_points,
        seed=random_state,
    )
    bg0_xy = combine_background_sources(
        empty_grid_bg_xy=empty_grid_bg_xy,
        traditional_bg_xy=traditional_bg_xy,
        max_points=bg_max_points,
        seed=random_state,
    )
    if len(bg0_xy) == 0:
        raise RuntimeError("No background points were generated for transcript mode.")

    y_train = np.hstack([np.zeros(len(bg0_xy)), np.ones(len(target_xy))])
    X_train = np.vstack([bg0_xy, target_xy])

    reg = KNeighborsRegressor(n_neighbors=knn_k, weights="distance")
    reg.fit(X_train, y_train)

    prob = reg.predict(grid).reshape(xx.shape)
    prob_smooth = gaussian_filter(prob, sigma=smooth_sigma)
    prob_smooth_masked = prob_smooth.copy()
    prob_smooth_masked[~tissue_mask] = np.nan

    verts_list = extract_contour_paths(xx, yy, prob_smooth_masked, level=isoline_level)
    verts_list = filter_loops_by_cell_count(verts_list, target_xy, min_cells_inside=min_cells_inside)
    if len(verts_list) == 0:
        raise RuntimeError(
            "No transcript-driven isoline found.\n"
            "Suggestions: lower min_cells_inside, lower isoline_level, lower smooth_sigma for finer structures, or raise grid_n."
        )

    safe_gene = safe_filename_component(selected_gene)
    params_path = out_dir / "params.json"
    transcript_unique_cells = None
    if transcript_schema.cell_id_col is not None and transcript_schema.cell_id_col in target_df.columns:
        transcript_unique_cells = int(target_df[transcript_schema.cell_id_col].nunique())

    params = {
        "analysis_mode": SOURCE_MODE_TRANSCRIPT,
        "cells_parquet": str(cells_parquet),
        "transcript_parquet": str(transcript_parquet),
        "transcript_gene": selected_gene,
        "transcript_gene_col": transcript_schema.gene_col,
        "transcript_x_col": transcript_schema.x_col,
        "transcript_y_col": transcript_schema.y_col,
        "cell_id_col": cell_id_col,
        "cell_x_col": cell_x_col,
        "cell_y_col": cell_y_col,
        "grid_n": int(grid_n),
        "knn_k": int(knn_k),
        "smooth_sigma": float(smooth_sigma),
        "isoline_level": float(isoline_level),
        "margin_um": float(margin_um),
        "max_dist_threshold": float(max_dist_threshold),
        "bg_max_points": int(bg_max_points),
        "bg_d_max": float(bg_d_max),
        "min_cells_inside": int(min_cells_inside),
        "use_synth_bg": bool(use_synth_bg),
        "bbox_expand_um": float(bbox_expand_um),
        "syn_bg_density": float(syn_bg_density),
        "syn_bg_min": int(syn_bg_min),
        "syn_bg_max": int(syn_bg_max),
        "pad_fraction": float(pad_fraction),
        "n_target_transcripts": int(len(target_xy)),
        "n_background_points": int(len(bg0_xy)),
        "n_contours": int(len(verts_list)),
        "traditional_bg_points": int(len(traditional_bg_xy)),
        "empty_grid_bg_points": int(len(empty_grid_bg_xy)),
        "empty_grid_candidate_points": int(empty_grid_candidates),
        "occupied_grid_count": int(occupied_grid_count),
        "transcript_unique_cells": transcript_unique_cells,
        "blank_grid_background_enabled": True,
    }
    params_path.write_text(
        json.dumps(params, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    for i, verts in enumerate(verts_list):
        np.save(out_dir / f"pattern1_transcript_{safe_gene}_isoline_{isoline_level:g}_{i}.npy", verts)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    if len(traditional_bg_xy) > 0:
        plt.scatter(
            traditional_bg_xy[:, 0],
            traditional_bg_xy[:, 1],
            s=1,
            alpha=0.03,
            label="cell bg",
        )
    if len(empty_grid_bg_xy) > 0:
        plt.scatter(
            empty_grid_bg_xy[:, 0],
            empty_grid_bg_xy[:, 1],
            s=1,
            alpha=0.05,
            label="blank grid bg",
        )
    plt.scatter(target_xy[:, 0], target_xy[:, 1], s=2, alpha=0.75, label=f"{selected_gene} transcripts")
    for verts in verts_list:
        plt.plot(verts[:, 0], verts[:, 1], linewidth=2)
    plt.gca().set_aspect("equal")
    plt.title(
        f"Transcript Pattern1 | gene={selected_gene} | isoline={isoline_level:g} | contours={len(verts_list)}"
    )
    plt.legend(frameon=False)
    plt.tight_layout()
    preview_path = out_dir / f"pattern1_transcript_{safe_gene}_isoline_{isoline_level:g}.png"
    plt.savefig(preview_path, dpi=300)
    plt.close()

    result = Pattern1IsolineResult(
        out_dir=out_dir,
        id_col_used=transcript_schema.cell_id_col or "transcript_id",
        x_col=transcript_schema.x_col,
        y_col=transcript_schema.y_col,
        n_target_cells=int(len(target_xy)),
        n_bg0_points=int(len(bg0_xy)),
        contours=list(verts_list),
        label_scheme="transcript_gene_is_signal",
        segmentation_confidence_score=None,
        segmentation_confidence_stats=None,
        params_json=params_path,
        preview_png=preview_path,
    )
    stats = BlankGridBackgroundStats(
        traditional_bg_points=int(len(traditional_bg_xy)),
        empty_grid_bg_points=int(len(empty_grid_bg_xy)),
        empty_grid_candidate_points=int(empty_grid_candidates),
        occupied_grid_count=int(occupied_grid_count),
    )
    metadata = {
        "analysis_mode": SOURCE_MODE_TRANSCRIPT,
        "transcript_gene": selected_gene,
        "transcript_gene_col": transcript_schema.gene_col,
        "transcript_x_col": transcript_schema.x_col,
        "transcript_y_col": transcript_schema.y_col,
        "transcript_rows": int(transcript_rows_total) if transcript_rows_total is not None else None,
        "n_target_transcripts": int(len(target_xy)),
        "transcript_unique_cells": transcript_unique_cells,
        "cell_id_col": cell_id_col,
        "cell_x_col": cell_x_col,
        "cell_y_col": cell_y_col,
    }
    return result, stats, metadata


def run_analysis(
    input_mode: str,
    source_mode: str,
    cells_parquet: object | None,
    cells_storage_path: str | None,
    clusters_csv: object | None,
    clusters_storage_path: str | None,
    transcript_parquet: object | None,
    transcript_storage_path: str | None,
    tissue_boundary_csv: object | None,
    tissue_storage_path: str | None,
    pattern1_clusters: str,
    transcript_gene: str | None,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    isoline_level: float,
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
        use_storage_mode = input_mode == INPUT_MODE_STORAGE
        progress(0.05, desc="Resolving input files")
        yield emit_status(
            phase="staging-inputs",
            run_dir=run_dir,
            lines=[
                "Reading files from mounted project storage."
                if use_storage_mode
                else "Copying uploaded files into the app workspace."
            ]
            + ([f"Cleaned old run directories: {', '.join(removed_runs)}"] if removed_runs else []),
            summary=summary,
        )

        cells_input, clusters_input, tissue_input, transcript_input = resolve_inputs(
            input_mode=input_mode,
            cells_upload=cells_parquet,
            cells_storage_path=cells_storage_path,
            clusters_upload=clusters_csv,
            clusters_storage_path=clusters_storage_path,
            tissue_upload=tissue_boundary_csv,
            tissue_storage_path=tissue_storage_path,
            transcript_upload=transcript_parquet,
            transcript_storage_path=transcript_storage_path,
            target_dir=upload_dir,
        )
        if cells_input is None:
            if use_storage_mode:
                raise ValueError("Missing cells.parquet. Enter a mounted-storage path for the Xenium cell coordinate file.")
            raise ValueError("Missing cells.parquet. Please upload the Xenium cell coordinate file.")

        cells_path = cells_input.path
        clusters_path = clusters_input.path if clusters_input is not None else None
        tissue_path = tissue_input.path if tissue_input is not None else None
        transcript_path = transcript_input.path if transcript_input is not None else None

        use_cluster_mode = source_mode != SOURCE_MODE_TRANSCRIPT
        analysis_mode = SOURCE_MODE_CLUSTER if use_cluster_mode else SOURCE_MODE_TRANSCRIPT

        estimated_cells_rows = safe_count_parquet_rows(cells_path)
        estimated_cluster_rows = safe_count_csv_rows(clusters_path) if clusters_path is not None else None
        estimated_transcript_rows = safe_count_parquet_rows(transcript_path) if transcript_path is not None else None
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
        if not use_cluster_mode:
            runtime_notes.append(
                "Transcript mode treats the selected gene as the signal of interest and all cells as background."
            )
            runtime_notes.append(
                "In transcript mode, min_cells_inside is applied to target transcript points inside each contour."
            )
            if compute_confidence_score:
                runtime_notes.append(
                    "Segmentation confidence score is not available in transcript mode and was skipped."
                )

        effective_bg_max_points = min(int(bg_max_points), int(runtime_profile.bg_max_points))
        effective_syn_bg_min = min(int(syn_bg_min), int(runtime_profile.syn_bg_min))
        effective_syn_bg_max = min(int(syn_bg_max), int(runtime_profile.syn_bg_max))

        summary.update(
            {
                "estimated_cells_rows": estimated_cells_rows,
                "estimated_cluster_rows": estimated_cluster_rows,
                "estimated_transcript_rows": estimated_transcript_rows,
                "analysis_mode": analysis_mode,
                "input_mode": input_mode,
                "input_sources": {
                    "cells_parquet": {
                        "path": str(cells_path),
                        "source": cells_input.source,
                        "original": cells_input.original,
                    },
                    "clusters_csv": (
                        {
                            "path": str(clusters_path),
                            "source": clusters_input.source,
                            "original": clusters_input.original,
                        }
                        if clusters_input is not None and clusters_path is not None
                        else None
                    ),
                    "transcript_parquet": (
                        {
                            "path": str(transcript_path),
                            "source": transcript_input.source,
                            "original": transcript_input.original,
                        }
                        if transcript_input is not None and transcript_path is not None
                        else None
                    ),
                    "tissue_boundary_csv": (
                        {
                            "path": str(tissue_path),
                            "source": tissue_input.source,
                            "original": tissue_input.original,
                        }
                        if tissue_input is not None and tissue_path is not None
                        else None
                    ),
                },
                "requested_parameters": {
                    "input_mode": input_mode,
                    "source_mode": analysis_mode,
                    "grid_n": int(grid_n),
                    "knn_k": int(knn_k),
                    "smooth_sigma": float(smooth_sigma),
                    "isoline_level": float(isoline_level),
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
                    "compute_confidence_score": bool(compute_confidence_score),
                },
                "effective_parameters": {
                    "input_mode": input_mode,
                    "source_mode": analysis_mode,
                    "grid_n": int(runtime_profile.grid_n),
                    "knn_k": int(knn_k),
                    "smooth_sigma": float(smooth_sigma),
                    "isoline_level": float(isoline_level),
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
            f"Input mode: {'mounted project storage' if use_storage_mode else 'browser upload'}",
            f"Pattern1 source mode: {analysis_mode}",
            f"cells.parquet path: {cells_path}",
        ]
        if use_cluster_mode:
            if clusters_path is None:
                if use_storage_mode:
                    raise ValueError(
                        "Missing clusters.csv. Enter a mounted-storage path for the GraphClust cluster assignment file."
                    )
                raise ValueError("Missing clusters.csv. Please upload the GraphClust cluster assignment file.")
            selected_clusters = parse_pattern1_clusters(pattern1_clusters)
            summary["selected_clusters"] = [str(item) for item in selected_clusters]
            summary["label_scheme"] = label_scheme_description(label_scheme)
            preflight_lines.extend(
                [
                    f"clusters.csv path: {clusters_path}",
                    f"Pattern1 clusters: {', '.join(str(item) for item in selected_clusters)}",
                    f"Estimated cells.parquet rows: {estimated_cells_rows if estimated_cells_rows is not None else 'unknown'}",
                    f"Estimated clusters.csv rows: {estimated_cluster_rows if estimated_cluster_rows is not None else 'unknown'}",
                    f"grid_n: requested {grid_n}, effective {runtime_profile.grid_n}",
                    f"knn_k: {knn_k}",
                    f"smooth_sigma: {smooth_sigma:.2f}",
                    f"isoline_level: {isoline_level:.2f}",
                    f"min_cells_inside: {min_cells_inside}",
                    f"label scheme: {label_scheme_description(label_scheme)}",
                    f"Synthetic background: {'enabled' if synth_enabled else 'disabled'}",
                    "Blank grid background: enabled (empty tissue-mask grid cells are treated as background)",
                ]
            )
        else:
            if transcript_path is None:
                if use_storage_mode:
                    raise ValueError("Missing transcript.parquet. Enter a mounted-storage path for the Xenium transcript file.")
                raise ValueError("Missing transcript.parquet. Please upload the Xenium transcript file.")
            selected_gene = parse_transcript_gene(transcript_gene)
            summary["transcript_gene"] = selected_gene
            summary["label_scheme"] = "Selected transcript gene is treated as the signal of interest"
            preflight_lines.extend(
                [
                    f"transcript.parquet path: {transcript_path}",
                    f"Transcript gene: {selected_gene}",
                    f"Estimated cells.parquet rows: {estimated_cells_rows if estimated_cells_rows is not None else 'unknown'}",
                    f"Estimated transcript.parquet rows: {estimated_transcript_rows if estimated_transcript_rows is not None else 'unknown'}",
                    f"grid_n: requested {grid_n}, effective {runtime_profile.grid_n}",
                    f"knn_k: {knn_k}",
                    f"smooth_sigma: {smooth_sigma:.2f}",
                    f"isoline_level: {isoline_level:.2f}",
                    f"min_cells_inside: {min_cells_inside}",
                    f"Synthetic background: {'enabled' if synth_enabled else 'disabled'}",
                    "Background mode: all cells are treated as background in addition to blank grid cells.",
                ]
            )
        preflight_lines.extend(runtime_notes)

        progress(0.18, desc="Inputs ready")
        yield emit_status(
            phase="preflight",
            run_dir=run_dir,
            lines=preflight_lines,
            summary=summary,
        )

        progress(0.42, desc="Running Pattern1 isoline")
        if use_cluster_mode:
            cfg = Pattern1IsolineConfig(
                clusters_csv=clusters_path,
                cells_parquet=cells_path,
                tissue_boundary_csv=tissue_path if synth_enabled else None,
                out_dir=output_dir,
                pattern1_clusters=selected_clusters,
                grid_n=int(runtime_profile.grid_n),
                knn_k=int(knn_k),
                smooth_sigma=float(smooth_sigma),
                isoline_level=float(isoline_level),
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
            log_event(
                "Running Pattern1 isoline | "
                f"mode={analysis_mode} | clusters={summary['selected_clusters']} | grid_n={cfg.grid_n} | "
                f"knn_k={cfg.knn_k} | smooth_sigma={cfg.smooth_sigma} | isoline_level={cfg.isoline_level}"
            )
            result, blank_grid_stats = run_pattern1_isoline_blank_grid(cfg)
            target_label = "Target cells"

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
        else:
            log_event(
                "Running Pattern1 isoline | "
                f"mode={analysis_mode} | transcript_gene={summary['transcript_gene']} | "
                f"grid_n={runtime_profile.grid_n} | knn_k={knn_k} | smooth_sigma={smooth_sigma} | "
                f"isoline_level={isoline_level}"
            )
            result, blank_grid_stats, transcript_meta = run_transcript_gene_isoline_blank_grid(
                cells_parquet=cells_path,
                transcript_parquet=transcript_path,
                transcript_gene=selected_gene,
                tissue_boundary_csv=tissue_path if synth_enabled else None,
                out_dir=output_dir,
                grid_n=int(runtime_profile.grid_n),
                knn_k=int(knn_k),
                smooth_sigma=float(smooth_sigma),
                isoline_level=float(isoline_level),
                margin_um=float(margin_um),
                max_dist_threshold=float(max_dist_threshold),
                bg_max_points=int(effective_bg_max_points),
                bg_d_max=float(bg_d_max),
                min_cells_inside=int(min_cells_inside),
                use_synth_bg=bool(synth_enabled),
                bbox_expand_um=float(bbox_expand_um),
                syn_bg_density=float(runtime_profile.syn_bg_density),
                syn_bg_min=int(effective_syn_bg_min),
                syn_bg_max=int(effective_syn_bg_max),
            )
            summary.update(transcript_meta)
            target_label = "Target transcripts"

        summary.update(
            {
                "id_col_used": result.id_col_used,
                "x_col": result.x_col,
                "y_col": result.y_col,
                "n_target_cells": int(result.n_target_cells),
                "n_bg0_points": int(result.n_bg0_points),
                "n_contours": int(len(result.contours)),
                "blank_grid_background": {
                    "enabled": True,
                    "traditional_bg_points": int(blank_grid_stats.traditional_bg_points),
                    "empty_grid_bg_points": int(blank_grid_stats.empty_grid_bg_points),
                    "empty_grid_candidate_points": int(blank_grid_stats.empty_grid_candidate_points),
                    "occupied_grid_count": int(blank_grid_stats.occupied_grid_count),
                },
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
            f"{target_label}: {result.n_target_cells}",
            f"Background points: {result.n_bg0_points}",
            f"Isoline level: {isoline_level:.2f}",
            f"Blank grid background points used: {blank_grid_stats.empty_grid_bg_points}",
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
            <span>Direct control of grid_n, knn_k, sigma, isoline</span>
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
            <h3>Choose upload or mounted storage</h3>
            <p>
              Either upload Xenium files in the browser or point the app at mounted project-storage
              paths such as <code>/home/data/...</code>. Then provide <code>cells.parquet</code>
              together with either <code>clusters.csv</code> or <code>transcript.parquet</code>.
            </p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 2</div>
            <h3>Choose the Pattern1 signal</h3>
            <p>
              Either enter cluster IDs such as <code>10,23,19</code>, or upload
              <code>transcript.parquet</code> and choose a single gene to use as Pattern1.
            </p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 3</div>
            <h3>Tune KNN, sigma, and isoline</h3>
            <p>
              Adjust <code>grid_n</code>, <code>knn_k</code>, <code>smooth_sigma</code>,
              <code>isoline_level</code>, and <code>min_cells_inside</code>, then run the original
              contour workflow.
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
                f"""
                <div class="micro-guide">
                  In a standard Xenium <code>outs</code> directory, <code>cells.parquet</code> is usually in
                  the root of <code>outs</code>, while one common <code>clusters.csv</code> path is
                  <code>outs\\analysis\\clustering\\gene_expression_graphclust\\clusters.csv</code>.
                  A matching <code>transcripts.parquet</code> file is usually also in the root of <code>outs</code>.
                  Mounted-storage mode accepts files only inside: <code>{INPUT_ROOTS_LABEL}</code>.
                </div>
                """
            )
            input_mode = gr.Radio(
                label="Input source",
                choices=[
                    ("Upload files in browser", INPUT_MODE_UPLOAD),
                    ("Use mounted project storage paths", INPUT_MODE_STORAGE),
                ],
                value=INPUT_MODE_UPLOAD,
                info="Mounted-storage mode reads files directly from the attached Serve volume and avoids the 100 MB browser upload limit.",
            )
            source_mode = gr.Radio(
                label="Pattern1 source",
                choices=[
                    ("Cluster IDs from clusters.csv", SOURCE_MODE_CLUSTER),
                    ("Single gene from transcript.parquet", SOURCE_MODE_TRANSCRIPT),
                ],
                value=SOURCE_MODE_CLUSTER,
                info="Choose whether Pattern1 comes from clustered cells or directly from transcript coordinates.",
            )
            cells_parquet = gr.File(
                label="Cell coordinates (cells.parquet)",
                file_types=[".parquet"],
                type="filepath",
            )
            cells_storage_path = gr.Textbox(
                label="Mounted path: cells.parquet",
                value="",
                placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/outs/cells.parquet",
                visible=False,
            )
            clusters_csv = gr.File(
                label="Cluster assignments (clusters.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            clusters_storage_path = gr.Textbox(
                label="Mounted path: clusters.csv",
                value="",
                placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/outs/analysis/clustering/gene_expression_graphclust/clusters.csv",
                visible=False,
            )
            transcript_parquet = gr.File(
                label="Transcripts (transcript.parquet)",
                file_types=[".parquet"],
                type="filepath",
                visible=False,
            )
            transcript_storage_path = gr.Textbox(
                label="Mounted path: transcript.parquet",
                value="",
                placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/outs/transcripts.parquet",
                visible=False,
            )
            tissue_boundary_csv = gr.File(
                label="Tissue boundary (optional: tissue_boundary.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            tissue_storage_path = gr.Textbox(
                label="Mounted path: tissue_boundary.csv (optional)",
                value="",
                placeholder=f"{PRIMARY_INPUT_ROOT}/my-run/tissue_boundary.csv",
                visible=False,
            )
            pattern1_clusters = gr.Textbox(
                label="Pattern1 cluster IDs",
                value=DEFAULT_PATTERN1,
                lines=4,
                placeholder="10,23,19",
                info="Comma-separated cluster IDs. Newlines and semicolons are also accepted.",
                elem_id="pattern1-clusters",
            )
            transcript_gene = gr.Dropdown(
                label="Transcript gene as Pattern1",
                choices=[],
                value=None,
                allow_custom_value=True,
                filterable=True,
                interactive=True,
                visible=False,
                info="Upload transcript.parquet or enter a mounted-storage path to auto-load genes, or type one manually.",
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
            isoline_level = gr.Slider(
                label="isoline_level (contour threshold)",
                minimum=0.05,
                maximum=0.95,
                step=0.01,
                value=0.50,
                info="Lower values expand the contour; higher values make the contour stricter.",
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

    input_mode.change(
        fn=update_visibility,
        inputs=[input_mode, source_mode],
        outputs=[
            cells_parquet,
            clusters_csv,
            transcript_parquet,
            tissue_boundary_csv,
            cells_storage_path,
            clusters_storage_path,
            transcript_storage_path,
            tissue_storage_path,
            pattern1_clusters,
            transcript_gene,
            label_scheme,
            compute_confidence_score,
        ],
    )
    source_mode.change(
        fn=update_visibility,
        inputs=[input_mode, source_mode],
        outputs=[
            cells_parquet,
            clusters_csv,
            transcript_parquet,
            tissue_boundary_csv,
            cells_storage_path,
            clusters_storage_path,
            transcript_storage_path,
            tissue_storage_path,
            pattern1_clusters,
            transcript_gene,
            label_scheme,
            compute_confidence_score,
        ],
    )
    source_mode.change(
        fn=load_transcript_gene_options,
        inputs=[input_mode, transcript_parquet, transcript_storage_path],
        outputs=[transcript_gene],
    )
    transcript_parquet.change(
        fn=load_transcript_gene_options,
        inputs=[input_mode, transcript_parquet, transcript_storage_path],
        outputs=[transcript_gene],
    )
    transcript_storage_path.change(
        fn=load_transcript_gene_options,
        inputs=[input_mode, transcript_parquet, transcript_storage_path],
        outputs=[transcript_gene],
    )
    input_mode.change(
        fn=load_transcript_gene_options,
        inputs=[input_mode, transcript_parquet, transcript_storage_path],
        outputs=[transcript_gene],
    )

    run_button.click(
        fn=run_analysis,
        inputs=[
            input_mode,
            source_mode,
            cells_parquet,
            cells_storage_path,
            clusters_csv,
            clusters_storage_path,
            transcript_parquet,
            transcript_storage_path,
            tissue_boundary_csv,
            tissue_storage_path,
            pattern1_clusters,
            transcript_gene,
            grid_n,
            knn_k,
            smooth_sigma,
            isoline_level,
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
