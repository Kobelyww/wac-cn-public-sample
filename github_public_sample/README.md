# `github_public_sample` bundle

## Audio (`wac_cn_audio_sample/`)

- Layout: one folder per macro label (English slug).
- Drawn from the paper's HuBERT-ZH-B feature manifest: **3 clips per class**, 9 classes → **27** WAV files.
- Not a statistical sample of the corpus; intended for listening tests, format checks, and teaching.

## Metadata (`metadata/wac_cn_sample_manifest.csv`)

Columns:

- `label_annot`: original annotation string (often Chinese region name or `Mandarin`)
- `label_slug`: directory name under `wac_cn_audio_sample/`
- `relpath`: path relative to `github_public_sample/`

## Paper artifacts (`paper_artifacts/`)

Frozen-encoder confusion PDFs, t-SNE PNGs, and the data-collection pipeline figure used in the camera-ready draft.
