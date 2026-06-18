# Fine Structure Pattern1 Serve App

This directory is a deployment-ready wrapper for the original HistoSeg `pattern1` contour workflow.

It is intentionally not the multi-structure app. The UI exposes two single-structure Pattern1 entry modes:

- cluster mode: use `cells.parquet` + `clusters.csv` and choose Pattern1 cluster IDs
- transcript mode: use `cells.parquet` + `transcripts.parquet` and choose a single gene as Pattern1

The shared contour parameters are:

- `pattern1_clusters`
- `transcript_gene`
- `grid_n`
- `knn_k`
- `smooth_sigma`
- `isoline_level`
- `min_cells_inside`

It also keeps the optional synthetic background controls used by the upstream Pattern1 implementation.
In addition, this wrapper now treats blank grid cells inside the tissue mask as explicit background points, which is useful for finer structures such as gland lumens.

## Inputs

Upload these files in the app:

- `cells.parquet`
- `clusters.csv` for cluster mode
- `transcripts.parquet` for transcript mode
- `tissue_boundary.csv` only if you want synthetic background points

Typical Xenium locations:

- `outs/cells.parquet`
- `outs/analysis/clustering/gene_expression_graphclust/clusters.csv`
- `outs/transcripts.parquet` or `outs/transcript.parquet` depending on the export

## What the app writes

For each run, the app creates a new run directory and writes:

- `params.json`
- `pattern1_isoline_*.npy`
- `pattern1_isoline_<isoline_level>.png`
- `pattern1_transcript_<gene>_isoline_*.npy` for transcript mode
- `pattern1_transcript_<gene>_isoline_<isoline_level>.png` for transcript mode
- a ZIP archive when disk space allows

## Local Docker test

```bash
docker build --platform linux/amd64 -t fine-structure-pattern1-serve:local .
docker run --rm -it -p 7860:7860 fine-structure-pattern1-serve:local
```

Then open `http://localhost:7860`.

## GitHub Container Registry

The workflow in `.github/workflows/docker-image-ghcr.yml` publishes the image to:

```text
ghcr.io/<your-github-owner>/fine-structure-pattern1-serve
```

It publishes:

- `sha-<commit>`
- `latest`

For SciLifeLab Serve, use a unique tag such as `sha-<commit>` instead of `latest`.

## SciLifeLab Serve setup

Create a **Gradio app** in your `fine structure` project and use:

- `Port`: `7860`
- `Image`: `ghcr.io/<your-github-owner>/fine-structure-pattern1-serve:sha-<commit>`
- `Source code URL`: your public GitHub repository URL
- `Permissions`: `Link` or `Public`

Serve documentation currently notes a Gradio deployment bug with `Private` and `Project` permissions, so `Link` or `Public` is the safer choice.

If you want run outputs to persist across restarts, configure a project storage mount path in Serve and attach it to the app.

## Upstream HistoSeg reference

This wrapper installs HistoSeg from:

- Repository: `https://github.com/hutaobo/HistoSeg`
- Pinned commit: `7e0526013f2d36200e464a070a359dc12a982c19`

The upstream project README and license should stay visible in your public source repository because the container depends on that code.

## License note

The upstream HistoSeg repository declares the PolyForm Noncommercial 1.0.0 license. Make sure your intended use on Serve is consistent with that license.
