# Score2ConVec

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md)

Website: https://utaisynthesizer.net

> Give any ContentVec-based SVC voice model the ability to **sing from a score**.
> Score (MIDI notes + lyrics) → **ContentVec** → so-vits-svc 4.0 / 4.1 or RVC → singing.

Score2ConVec (Score-to-ContentVec) is a small, deterministic **SVS front-end** for **SVC** voice models.
It reads a musical score — notes + lyrics — and produces **ContentVec** content features, which any
ContentVec-based SVC backend decodes into a singing voice. It turns an SVC "cover" model into a full
"sing-from-score" singer, **without** training a heavyweight end-to-end SVS model.

```
  score  ──G2P + arrays──▶  Score2ConVec  ──▶  ContentVec [T, D] @50fps ──┐
 (notes + lyrics)                              (deterministic content)     ├─▶  SVC backend ─▶ singing .wav
                                                                           │    (so-vits-svc / RVC = the voice)
  f0 stream (note pitch, from the DAW) ─────────────────────────────────────┘
```

## Why decouple

Traditional score-based SVS (FFT/FastSpeech/DiffSinger-style singers) learns notes → spectrogram **end to
end**, with pitch, content, and timbre all entangled inside one decoder. Score2ConVec instead splits singing
into **three independent axes**:

- **Pitch (f0)** — a separate stream you supply from the DAW (exact note pitch + portamento/vibrato).
- **Content ("what is sung")** — produced by *this* model as speaker-invariant ContentVec.
- **Timbre ("who sings")** — comes entirely from the **SVC backend**. Same content → different backend → different singer.

The SVC backend already separates f0 from timbre; Score2ConVec factors **content** out on top of that. The
payoff is a **very low training barrier**: an SVC voice needs only ~10–15 minutes of dry vocals and **no manual
phoneme labeling**, yet gains score-driven singing. The content model is trained once, multilingually, and is
**backend-agnostic** and **deterministic** (same score → same output, no sampling).

## Supported backends

| Backend | ContentVec | Model | Checkpoint |
|---|---|---|---|
| **so-vits-svc 4.1** | vec768l12 (768-d) | `ScoreToCV` 768 | `cv_final.pt` |
| **RVC v2** | ContentVec 768 | `ScoreToCV` 768 | `cv_final.pt` (zero-retrain, feeds straight in) |
| **so-vits-svc 4.0** | vec256l9 (256-d) | `ScoreToCV` 256 | `cv256_final.pt` |

Same architecture, only the target ContentVec flavor differs. Full backend guides:
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md) and
[docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md).

**Languages (EAR-accepted):** zh, ja, en, de, fr, es, it.

## Install

```bash
git clone https://github.com/yasoukyoku/Score2ConVec.git
cd Score2ConVec

# Python 3.10 recommended. Install PyTorch for your CUDA build first (https://pytorch.org), then:
pip install -r requirements.txt
```

You also need, separately:

1. **A ContentVec-based SVC backend** — a [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
   checkout (4.0 or 4.1) or RVC, plus a voice model (`.pth` + `config.json`). This is *the voice*.
2. **Model weights** for Score2ConVec (below).

## Weights

The `ScoreToCV` checkpoints are **not** in the git tree (each ~188 MB). Download them from the
[**Releases**](https://github.com/yasoukyoku/Score2ConVec/releases) page and place them under `checkpoints/`:

| File | Dim | For | det_floor |
|---|---|---|---|
| `cv_final.pt` | 768 | so-vits-svc 4.1, RVC v2 | ~0.795 |
| `cv256_final.pt` | 256 | so-vits-svc 4.0 | 0.791 |

> The weights are for **research / non-commercial** use — see [Training data & licensing](#training-data--licensing).

## Quick start

**1) Check the front-end (G2P) — no models needed:**

```bash
python scripts/render_ust.py --ust your_song.ust --dump
```

This parses a UST (SynthV / UTAU export), maps each lyric to IPA phones, and prints the per-note result so
you can verify the lyric → phoneme mapping before rendering.

**2) Render (needs `cv_final.pt` + an SVC backend):**

```bash
# point the glue at your local so-vits-svc checkout and voice model
export SOVITS_ROOT=/path/to/so-vits-svc
export SOVITS_MODEL=/path/to/your_voice.pth
export SOVITS_CONFIG=/path/to/config.json          # (Windows: use `set NAME=...`)

python scripts/render_ust.py --ust your_song.ust --out processed/out
# -> processed/out/render_noteonly.wav
```

**Minimal Python (the I/O contract):**

```python
import torch, yaml
from src.model.score2cv import ScoreToCV

cfg = yaml.safe_load(open("configs/model_cv_final.yaml", encoding="utf-8"))
model = ScoreToCV(cfg).cuda().float().eval()
model.load_state_dict(torch.load("checkpoints/cv_final.pt", weights_only=False)["model"])

# build per-phone score arrays (see render_ust.build_arrays / DEPLOY §2-3):
# phonemes, note_pitch, phone_dur, note_dur, note_to_phone, speaker_id, lang_id, phone_mask, technique(zeros)
with torch.no_grad():
    out = model(**inputs)
    T   = int(out["frame_mask"][0].sum())
    cv  = model.infer_cv(out["frame_hidden"])[0, :T].cpu().numpy()   # [T, 768] ContentVec, de-normalized
# feed cv + an f0 stream to the SVC backend -> singing.
```

The full I/O contract (every input array, the f0 stream, chunking for long songs, the RVC recipe) is in
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md).

## Training / retargeting

The architecture is backend-agnostic — only the **target feature** changes. To train a voice or retarget to a
new ContentVec flavor (e.g. the 256-d retarget for so-vits 4.0):

1. Extract features over your audio — `scripts/extract_contentvec.py` (768) or `extract_contentvec256.py`
   (256), `extract_f0.py`, `extract_notes.py`.
2. Pack aligned `.npz` — `scripts/pack_npz.py` (768) or `build_npz256.py` (256), then `compute_cv_norm*.py`.
3. Train — `python scripts/train_cv.py --config configs/model_cv_final.yaml`.
4. **Validate by ear** through the target backend (a cv-space metric is never a ship gate).

Alignment uses a forced aligner (this project used [HubertFA](https://github.com/qixi-oss/HubertFA); MFA works
too) — not included here. Details: [docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md) §4.

## Repository layout

```
src/model/          ScoreToCV (score2cv.py), ScoreToF0, shared sub-modules
src/preprocessing/  IPA phoneme vocab (210 tokens, 9 langs), ContentVec / f0 / RMVPE extractors
src/training/       dataset + losses
configs/            model_cv_final.yaml (768), model_cv256.yaml (256), model_f0_single.yaml
scripts/            feature extraction, npz packing, training, and the render/inference front-end
docs/               per-backend deployment guides (768 / 256)
checkpoints/        (you place the downloaded .pt here)
```

## Limitations (honest scope)

- **7 languages** are ear-accepted (zh/ja/en/de/fr/es/it). ko/ru were dropped (alignment quality).
- The content model uses a **deterministic conditional-mean head** — clean and stable, but residual artifacts
  show up as slightly breathy/blurred consonants. This is a property of the head, not under-training.
- **f0 is parametric in the DAW.** The learned f0 model was retired (it undershot large interval jumps); use
  exact note pitch + portamento + tail vibrato. See DEPLOY §7.
- `scripts/render_ust.py` is a **first-pass reference front-end** — a real DAW supplies precise per-note timing
  and should expose the duration / grouping / f0 knobs itself.

## Training data & licensing

**Code** is MIT (see [LICENSE](LICENSE)). **Weights** are research / non-commercial.

The shipped models are trained on public singing corpora: **M4Singer** (zh), **GTSinger** (en/de/fr/es/it), and
Japanese singing databases (**kiritan_singing, PJS, Ofuton-P, Oniku, Itako, Natsume**). Several of these are
released for **non-commercial / research use only** — for example, **M4Singer is CC BY-NC-SA 4.0**. The released
weights are **not** an MIT relicensing of that data. Review each dataset's own license before any commercial use.

Credits required by the Japanese databases' terms (verbatim): 『©SSS』 · 『歌声DB制作:アマノケイ 音声提供者: 霧野蒼太』 · 『DB制作:おふとんP』 · 『御丹宮くるみ歌声データべース』 — for 東北きりたん歌唱データベース / 東北イタコ歌唱データベース (SSS LLC), 夏目悠李/男性歌声データベース (ATSUYA), おふとんP歌声データベース and 御丹宮くるみ歌声データべース; PJS (Koguchi & Takamichi; lyrics from 声優統計コーパス, 日本声優統計学会) is CC BY-SA 4.0.

Audio generated with these weights is the 「出力音声」 (output voice) of the Natsume Yuuri DB and falls under 「夏目悠李の出力音声に関する利用規約」 (https://ksdcm1ng.wixsite.com/njksofficial/%E8%A6%8F%E7%B4%84-rules): commercial use needs separate permission, and generated audio must not be used to build acoustic or pitch models.

**Licence position.** Consistent with Creative Commons' guidance that in many cases an AI model is not an adaptation of the works it was trained on (https://creativecommons.org/using-cc-licensed-works-for-ai-training/), we do not treat these weights as Adapted Material of the CC-licensed corpora, so the ShareAlike conditions of GTSinger / M4Singer (CC BY-NC-SA 4.0) and PJS (CC BY-SA 4.0) do not attach to them. The weights are nevertheless released for **non-commercial use only**, with the attribution above: several Japanese databases' usage agreements require it, and we respect the NonCommercial terms of the corpora that make up about 94% of the training data.

## Community

- **QQ group:** [1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord:** https://discord.gg/p3fGh942fJ

## Acknowledgements

Built on [ContentVec](https://github.com/auspicious3000/contentvec),
[RMVPE](https://github.com/Dream-High/RMVPE),
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc),
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI), and the singing corpora above.
