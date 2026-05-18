# WAC-CN public sample (ICASSP companion)

This repository hosts a **small public teaser** of the Web-sourced Accent Corpus for Chinese (WAC-CN) used in our ICASSP probing paper, plus **paper figure assets** (confusion matrices, t-SNE panels, pipeline diagram).

## Contents

| Path | Description |
|------|-------------|
| `github_public_sample/wac_cn_audio_sample/` | 27 short clips (3 per region label + Mandarin), WAV |
| `github_public_sample/metadata/wac_cn_sample_manifest.csv` | Clip metadata and **relative** paths |
| `github_public_sample/paper_artifacts/` | `experiment2_results/`, `experiment3_results/`, pipeline PNG |

The **full** corpus (~68h, upload-level splits, manifests) is **not** stored here because of size and redistribution policy. Use this repo to sanity-check labels, audio format, and to reproduce figures from released assets.

## Regenerate the audio sample

Requires a local checkout that still has `features/hubert-zh-b/manifest.csv` and `data_wav/` (not committed).

```bash
python3 scripts/build_github_public_sample.py
```

## Citation

If you use this sample, cite the ICASSP paper (see manuscript in the larger project) and respect host-platform terms for any downstream use of web-derived audio.

## License

Sample clips are derived from user-uploaded public web video; **use is at your own risk** for research and education only. Do not redistribute outside scholarly context unless you independently verify rights.
