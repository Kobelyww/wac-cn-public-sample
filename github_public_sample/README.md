# `github_public_sample` layout

## Audio (`wac_cn_audio_sample/`)

- **Bilibili-only:** `source_dataset == bilibili` in the HuBERT–Chinese-Base manifest.
- Up to **3 clips per label**; six regional labels appear in that manifest slice.
- Henan / Shanxi / read-Mandarin register clips are not part of this Bilibili-only bundle.

## Metadata (`metadata/wac_cn_sample_manifest.csv`)

- `relpath` is relative to `github_public_sample/`.
- `source_dataset` is `bilibili` for all listed clips.

## Results (`results/`)

- `experiment1_results/`, `experiment1_ablation_results/` — coarse binary probe.
- `experiment2_results/` — nine-way geometry and t-SNE exports.
- `experiment3_results/` — seven-way linear probe, confusion matrices, CSV summaries.
- `experiment3_supplement_mlp/` — MLP head comparison supplement.

## Figures (`figures/`)

Pipeline / ingest overview PNG used in the paper.
