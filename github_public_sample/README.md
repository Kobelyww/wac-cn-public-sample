# `github_public_sample` bundle

## Audio (`wac_cn_audio_sample/`)

- **Bilibili-only:** every row copied from the HuBERT-ZH-B manifest satisfies `source_dataset == bilibili` (excludes MHACCS and read-Mandarin/THCHS anchor rows).
- Layout: one folder per macro label (English slug), **3 clips per label** when available.
- Current manifest slice includes six Bilibili-heavy labels (e.g. Northeast, Southeast, Beijing, Southwest, Guangdong, Minnan); **Henan / Shanxi / Mandarin** may be absent here because they are not labeled as `bilibili` in that manifest.

## Metadata (`metadata/wac_cn_sample_manifest.csv`)

Columns:

- `label_annot`: original annotation string
- `label_slug`: directory name under `wac_cn_audio_sample/`
- `relpath`: path relative to `github_public_sample/`
- `source_dataset`: always `bilibili` in this teaser

## Paper artifacts (`paper_artifacts/`)

Frozen-encoder confusion PDFs, t-SNE PNGs, and the data-collection pipeline figure from the paper. These artifacts correspond to the **full** probing protocol in the manuscript, not to Bilibili-only inputs.
