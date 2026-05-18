# WAC-CN public sample (ICASSP companion)

This repository hosts a **small Bilibili-sourced audio teaser** for the Web-sourced Accent Corpus for Chinese (WAC-CN), **full experiment outputs** (Exp.~1–3 plus MLP probe supplement), and the data-ingest figure used in the paper.

## Bilibili-only audio

The folder `github_public_sample/wac_cn_audio_sample/` includes **only** rows with `source_dataset == bilibili` from our released HuBERT–Chinese-Base feature manifest. Other ingest streams and the read Mandarin register used elsewhere in the paper are **not** included in this audio bundle.

In the current manifest slice, Bilibili rows cover **six** regional labels, so the teaser is **up to 18** WAV files (3 clips per label when available).

## Experiment results

| Path | Contents |
|------|----------|
| `github_public_sample/results/experiment1_results/` | Binary probe tables and seeds |
| `github_public_sample/results/experiment1_ablation_results/` | Exp.~1 ablations |
| `github_public_sample/results/experiment2_results/` | Geometry metrics, t-SNE exports |
| `github_public_sample/results/experiment3_results/` | Seven-way probe, confusions, summaries |
| `github_public_sample/results/experiment3_supplement_mlp/` | MLP vs.\ linear probe supplement |

Figures and tables here match the artifacts referenced from the ICASSP manuscript.

## Figures

- `github_public_sample/figures/data_collection_pipeline_v2.png` — corpus acquisition overview.

## Metadata

- `github_public_sample/metadata/wac_cn_sample_manifest.csv` — Bilibili teaser clips and relative paths.

## Pretrained backbones (checkpoints)

Frozen encoders use the following Hugging Face Hub checkpoints (`repo_id`).

| Slug | Model / checkpoint |
|------|-------------------|
| `hubert-zh-b` | [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) |
| `hubert-en-b` | [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) |
| `w2v2-zh-b` | [TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |
| `w2v2-en-b` | [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) |
| `wavlm-b` | [microsoft/wavlm-base](https://huggingface.co/microsoft/wavlm-base) |
| `data2vec-audio-b` | [facebook/data2vec-audio-base](https://huggingface.co/facebook/data2vec-audio-base) |
| `unispeech-sat-b` | [microsoft/unispeech-sat-base](https://huggingface.co/microsoft/unispeech-sat-base) |
| `whisper-base` | [openai/whisper-base](https://huggingface.co/openai/whisper-base) |
| `ecapa-tdnn` | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| `resnet-voxceleb` | [speechbrain/spkrec-resnet-voxceleb](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb) |
| `xvect-voxceleb` | [speechbrain/spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) |
| `rawnet3` | [espnet/voxcelebs12_rawnet3](https://huggingface.co/espnet/voxcelebs12_rawnet3) |

The tables and confusion files in `results/experiment3_results/` use these short names as prefixes.

## Citation

If you use this material, cite the ICASSP paper and respect platform terms for web-derived audio.

## License

Sample clips come from user-uploaded public web video; **use is at your own risk** for research and education only. Do not redistribute outside scholarly context unless you independently verify rights.
