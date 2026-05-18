# WAC-CN public sample (ICASSP companion)

This repository hosts a **small public teaser** of Bilibili-ingested clips from the Web-sourced Accent Corpus for Chinese (WAC-CN), plus **paper figure assets** from the ICASSP probing manuscript.

## Bilibili-only audio policy

The folder `github_public_sample/wac_cn_audio_sample/` includes **only** rows with `source_dataset == bilibili` from the released HuBERT-ZH-B feature manifest. Clips from **MHACCS** and **read Mandarin (e.g. THCHS anchor)** are **not** copied here.

In the current manifest slice, Bilibili coverage spans **six** regional classes (no Henan/Shanxi/Mandarin rows under `bilibili` in that manifest), so the teaser contains **up to 3 clips × 6 classes = 18** WAV files.

The **`paper_artifacts/`** bundle still reflects the **full** experimental protocol in the paper (seven-way head including read Mandarin, geometry over the nine-way label inventory, etc.). It is **not** restricted to Bilibili sources.

## Contents

| Path | Description |
|------|-------------|
| `github_public_sample/wac_cn_audio_sample/` | Short clips (Bilibili-only), WAV |
| `github_public_sample/metadata/wac_cn_sample_manifest.csv` | Metadata and **relative** paths |
| `github_public_sample/paper_artifacts/` | Exp.2 / Exp.3 figures and tables, pipeline PNG |

The **full** corpus is **not** stored here because of size and redistribution policy.

## Regenerate the audio sample

Requires a local checkout with `features/hubert-zh-b/manifest.csv` and the audio files referenced there (paths may include `dataset/audio/...` and/or `data_wav/...`).

```bash
python3 scripts/build_github_public_sample.py
```

## Citation

If you use this sample, cite the ICASSP paper and respect platform terms for downstream use of web-derived audio.

## License

Sample clips are derived from user-uploaded public web video; **use is at your own risk** for research and education only. Do not redistribute outside scholarly context unless you independently verify rights.
