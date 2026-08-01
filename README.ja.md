# Score2ConVec

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語**

公式サイト: https://utaisynthesizer.net

> ContentVec ベースの SVC 音声モデルに、**譜面から歌う**能力を与えます。
> 譜面（MIDI ノート + 歌詞）→ **ContentVec** → so-vits-svc 4.0 / 4.1 または RVC → 歌声。

Score2ConVec（Score-to-ContentVec）は、**SVC** 音声モデルのための小さく決定的な **SVS フロントエンド**です。
譜面――ノート + 歌詞――を読み取り、**ContentVec** の内容特徴を出力します。ContentVec ベースの SVC バックエンドなら
どれでも、これを歌声にデコードできます。これにより、本来は「カバー」しかできなかった SVC モデルが、
重い end-to-end の SVS モデルを学習させることなく、「譜面を見て自ら歌う」歌い手になります。

```
   譜面   ──G2P + 配列──▶  Score2ConVec  ──▶  ContentVec [T, D] @50fps ──┐
(ノート + 歌詞)                            (決定的な「内容」)              ├─▶  SVC バックエンド ─▶ 歌声 .wav
                                                                          │   (so-vits-svc / RVC = 声色)
   f0 ピッチ列（ノートのピッチ、DAW 側から）──────────────────────────────┘
```

## なぜ分離するのか

従来の譜面ベース SVS（FFT / FastSpeech / DiffSinger 系の歌い手）は「ノート → スペクトログラム」を
**end-to-end** で学習し、ピッチ・内容・声色が 1 つの decoder の中で絡み合っています。Score2ConVec は逆に、
歌唱を**互いに独立した 3 つの軸**に分解します：

- **ピッチ（f0）** ―― DAW 側から与える独立したピッチ列（正確なノートピッチ + ポルタメント / ビブラート）。
- **内容（「何を歌うか」）** ―― *本モデル*が生成する、話者非依存（speaker-invariant）な ContentVec。
- **声色（「誰が歌うか」）** ―― 完全に **SVC バックエンド**から。同じ内容 → バックエンドを変える → 歌い手が変わる。

SVC バックエンドは f0 と声色をすでに分離しています。Score2ConVec はその上でさらに**内容**の軸を切り出します。
その見返りが**極めて低い学習コスト**です。SVC 音声はドライボーカル約 10〜15 分だけ、**手作業の音素ラベル付けは不要**で、
譜面駆動の歌唱を獲得できます。内容モデルは多言語で一度だけ学習すればよく、**バックエンド非依存**かつ**決定的**
（同じ譜面 → 同じ出力、サンプリングの揺らぎなし）です。

## 対応バックエンド

| バックエンド | ContentVec | モデル | 重み |
|---|---|---|---|
| **so-vits-svc 4.1** | vec768l12（768 次元） | `ScoreToCV` 768 | `cv_final.pt` |
| **RVC v2** | ContentVec 768 | `ScoreToCV` 768 | `cv_final.pt`（再学習不要、そのまま投入） |
| **so-vits-svc 4.0** | vec256l9（256 次元） | `ScoreToCV` 256 | `cv256_final.pt` |

アーキテクチャは同一で、ターゲットとする ContentVec の種類だけが異なります。各バックエンドの導入ガイドは
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md) と
[docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md) を参照してください。

**言語（耳で検証済み）：** 中・日・英・独・仏・西・伊。

## インストール

```bash
git clone https://github.com/yasoukyoku/Score2ConVec.git
cd Score2ConVec

# Python 3.10 推奨。まず CUDA に合わせて PyTorch を入れ（https://pytorch.org）、その後：
pip install -r requirements.txt
```

加えて、別途以下が必要です：

1. **ContentVec ベースの SVC バックエンド** ―― [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)（4.0
   または 4.1）または RVC のコードと、音声モデル（`.pth` + `config.json`）。**これが声色になります。**
2. **Score2ConVec のモデル重み**（下記）。

## 重み

`ScoreToCV` の重みは git ツリーには**含めていません**（各 約 188 MB）。
[**Releases**](https://github.com/yasoukyoku/Score2ConVec/releases) からダウンロードし、`checkpoints/` に置いてください：

| ファイル | 次元 | 用途 | det_floor |
|---|---|---|---|
| `cv_final.pt` | 768 | so-vits-svc 4.1、RVC v2 | ~0.795 |
| `cv256_final.pt` | 256 | so-vits-svc 4.0 | 0.791 |

> 重みは**研究 / 非商用**目的です ―― [学習データとライセンス](#学習データとライセンス)を参照。

## クイックスタート

**1) フロントエンド（G2P）の確認 ―― モデル不要：**

```bash
python scripts/render_ust.py --ust your_song.ust --dump
```

UST（SynthV / UTAU のエクスポート）を解析し、各歌詞を IPA 音素へ対応付けてノートごとに出力します。
レンダリング前に「歌詞 → 音素」の対応が正しいか確認できます。

**2) レンダリング（`cv_final.pt` + SVC バックエンドが必要）：**

```bash
# グルーコードをローカルの so-vits-svc と音声モデルに向ける
export SOVITS_ROOT=/path/to/so-vits-svc
export SOVITS_MODEL=/path/to/your_voice.pth
export SOVITS_CONFIG=/path/to/config.json          # （Windows は `set NAME=...`）

python scripts/render_ust.py --ust your_song.ust --out processed/out
# -> processed/out/render_noteonly.wav
```

**最小限の Python（I/O 契約）：**

```python
import torch, yaml
from src.model.score2cv import ScoreToCV

cfg = yaml.safe_load(open("configs/model_cv_final.yaml", encoding="utf-8"))
model = ScoreToCV(cfg).cuda().float().eval()
model.load_state_dict(torch.load("checkpoints/cv_final.pt", weights_only=False)["model"])

# 音素ごとの譜面配列を作る（render_ust.build_arrays / DEPLOY §2-3 を参照）：
# phonemes, note_pitch, phone_dur, note_dur, note_to_phone, speaker_id, lang_id, phone_mask, technique(全ゼロ)
with torch.no_grad():
    out = model(**inputs)
    T   = int(out["frame_mask"][0].sum())
    cv  = model.infer_cv(out["frame_hidden"])[0, :T].cpu().numpy()   # [T, 768] ContentVec、逆正規化済み
# cv と f0 ピッチ列を SVC バックエンドへ渡す -> 歌声。
```

完全な I/O 契約（各入力配列、f0 ピッチ列、長い曲のチャンク分割、RVC への投入レシピ）は
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md) を参照してください。

## 学習 / リターゲット

アーキテクチャはバックエンド非依存で、変えるのは**ターゲット特徴**だけです。音声を学習する、あるいは新しい
ContentVec の種類（例：so-vits 4.0 用の 256 次元）へリターゲットするには：

1. 音声から特徴抽出 ―― `scripts/extract_contentvec.py`（768）または `extract_contentvec256.py`（256）、
   `extract_f0.py`、`extract_notes.py`。
2. アライン済みの `.npz` を作成 ―― `scripts/pack_npz.py`（768）または `build_npz256.py`（256）、続いて
   `compute_cv_norm*.py`。
3. 学習 ―― `python scripts/train_cv.py --config configs/model_cv_final.yaml`。
4. **耳で検証** ―― ターゲットバックエンドを通して聴く（cv 空間の指標はリリース基準にはならない）。

アライメントには forced aligner が必要です（本プロジェクトは [HubertFA](https://github.com/qixi-oss/HubertFA) を
使用、MFA でも可）。本リポジトリには含まれません。詳細は
[docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md) §4。

## リポジトリ構成

```
src/model/          ScoreToCV (score2cv.py)、ScoreToF0、共有サブモジュール
src/preprocessing/  IPA 音素表（210 tokens、9 言語）、ContentVec / f0 / RMVPE 抽出器
src/training/       dataset + losses
configs/            model_cv_final.yaml (768)、model_cv256.yaml (256)、model_f0_single.yaml
scripts/            特徴抽出、npz パッキング、学習、レンダリング / 推論フロントエンド
docs/               各バックエンドのデプロイガイド（768 / 256）
checkpoints/        （ダウンロードした .pt をここに置く）
```

## 制約（正直な範囲）

- **7 言語**が耳で検証済み（中/日/英/独/仏/西/伊）。韓・露はアライメント品質の問題で除外。
- 内容モデルは**決定的な条件付き平均 head** を使用 ―― クリーンで安定しますが、残る粗さは子音がやや息っぽい /
  ぼやける形で現れます。これは head の性質であり、学習不足ではありません。
- **f0 は DAW 側のパラメトリック。** 学習型 f0 モデルは引退しました（大きな音程跳躍で「アンダーシュート」した
  ため）。正確なノートピッチ + ポルタメント + 語尾ビブラートを使ってください。DEPLOY §7 参照。
- `scripts/render_ust.py` は**初版の参照フロントエンド**です ―― 実際の DAW は正確なノート単位のタイミングを与え、
  デュレーション / グルーピング / f0 のノブを自前で公開すべきです。

## 学習データとライセンス

**コード**は MIT（[LICENSE](LICENSE) 参照）。**重み**は研究 / 非商用です。

公開モデルは公開歌声コーパスで学習しています：**M4Singer**（中）、**GTSinger**（英/独/仏/西/伊）、および
日本語歌声データベース（**kiritan_singing、PJS、Ofuton-P、Oniku、Itako、Natsume**）。このうちいくつかは
**非商用 / 研究目的のみ**で公開されています ―― 例えば **M4Singer は CC BY-NC-SA 4.0** です。公開された重みは
これらのデータを MIT で再ライセンスしたものでは**ありません**。商用利用の前に、各データセット自身のライセンスを
確認してください。

## コミュニティ

- **QQ グループ：** [1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord：** https://discord.gg/p3fGh942fJ

## 謝辞

[ContentVec](https://github.com/auspicious3000/contentvec)、
[RMVPE](https://github.com/Dream-High/RMVPE)、
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)、
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)、および上記の歌声コーパスに基づいて構築。
