# Score2ConVec

[English](README.md) · **简体中文** · [日本語](README.ja.md)

网站: https://utaisynthesizer.net

> 让任何基于 ContentVec 的 SVC 音色模型获得**看谱歌唱**的能力。
> 乐谱（MIDI 音符 + 歌词）→ **ContentVec** → so-vits-svc 4.0 / 4.1 或 RVC → 歌声。

Score2ConVec（Score-to-ContentVec）是一个小巧、确定性的 **SVS 前端**，服务于 **SVC** 音色模型。
它读取乐谱——音符 + 歌词——输出 **ContentVec** 内容特征；任何基于 ContentVec 的 SVC 后端都能把它解码成歌声。
于是一个原本只能"翻唱"的 SVC 模型，就变成了能"看谱自己唱"的歌手，**无需**再训练笨重的端到端 SVS 模型。

```
   乐谱   ──G2P + 数组──▶  Score2ConVec  ──▶  ContentVec [T, D] @50fps ──┐
(音符 + 歌词)                              (确定性的"内容")               ├─▶  SVC 后端 ─▶ 歌声 .wav
                                                                          │   (so-vits-svc / RVC = 音色)
   f0 音高流（音符音高，来自 DAW）─────────────────────────────────────────┘
```

## 为什么要解耦

传统的看谱式 SVS（FFT / FastSpeech / DiffSinger 那一类歌手）是**端到端**地学习"音符 → 频谱"，音高、内容、音色
全部纠缠在同一个 decoder 里。Score2ConVec 反其道而行，把歌唱拆成**三个互相独立的维度**：

- **音高（f0）**——一条独立的音高流，由你在 DAW 侧给定（精确音符音高 + 滑音 / 颤音）。
- **内容（"唱的是什么"）**——由*本模型*生成，是说话人无关（speaker-invariant）的 ContentVec。
- **音色（"谁在唱"）**——完全来自 **SVC 后端**。同一份内容 → 换后端 → 换歌手。

SVC 后端本身已经把 f0 与音色解耦了；Score2ConVec 在此之上再把**内容**这一维也拆出来。带来的直接好处是
**极低的训练门槛**：一个 SVC 音色只需约 10–15 分钟干声、**完全不用手工打标**，就能获得看谱歌唱能力。
而内容模型只需多语言训练一次，**与后端无关**、且**确定性**（同一份谱 → 同一份输出，无采样随机性）。

## 支持的后端

| 后端 | ContentVec | 模型 | 权重 |
|---|---|---|---|
| **so-vits-svc 4.1** | vec768l12（768 维） | `ScoreToCV` 768 | `cv_final.pt` |
| **RVC v2** | ContentVec 768 | `ScoreToCV` 768 | `cv_final.pt`（零重训，直接喂入） |
| **so-vits-svc 4.0** | vec256l9（256 维） | `ScoreToCV` 256 | `cv256_final.pt` |

架构完全相同，只是目标 ContentVec 的种类不同。完整的后端接入指南见
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md) 与
[docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md)。

**语言（经耳朵验收）：** 中、日、英、德、法、西、意。

## 安装

```bash
git clone https://github.com/yasoukyoku/Score2ConVec.git
cd Score2ConVec

# 推荐 Python 3.10。请先按你的 CUDA 版本安装 PyTorch（https://pytorch.org），然后：
pip install -r requirements.txt
```

此外还需要单独准备：

1. **一个基于 ContentVec 的 SVC 后端**——[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)（4.0 或
   4.1）或 RVC 的代码，外加一个音色模型（`.pth` + `config.json`）。**这就是音色来源。**
2. **Score2ConVec 的模型权重**（见下）。

## 权重

`ScoreToCV` 权重**不**放在 git 仓库里（每个约 188 MB）。请从
[**Releases**](https://github.com/yasoukyoku/Score2ConVec/releases) 页面下载，放到 `checkpoints/` 下：

| 文件 | 维度 | 用于 | det_floor |
|---|---|---|---|
| `cv_final.pt` | 768 | so-vits-svc 4.1、RVC v2 | ~0.795 |
| `cv256_final.pt` | 256 | so-vits-svc 4.0 | 0.791 |

> 权重仅供**研究 / 非商业**用途——见 [训练数据与许可](#训练数据与许可)。

## 快速上手

**1）先验证前端（G2P）——不需要任何模型：**

```bash
python scripts/render_ust.py --ust your_song.ust --dump
```

它会解析 UST（SynthV / UTAU 导出），把每个歌词映射到 IPA 音素并逐音符打印，方便你在渲染前先检查
"歌词 → 音素"的映射是否正确。

**2）渲染（需要 `cv_final.pt` + 一个 SVC 后端）：**

```bash
# 把胶水层指向你本地的 so-vits-svc 代码与音色模型
export SOVITS_ROOT=/path/to/so-vits-svc
export SOVITS_MODEL=/path/to/your_voice.pth
export SOVITS_CONFIG=/path/to/config.json          # （Windows 用 `set NAME=...`）

python scripts/render_ust.py --ust your_song.ust --out processed/out
# -> processed/out/render_noteonly.wav
```

**最小 Python 示例（I/O 约定）：**

```python
import torch, yaml
from src.model.score2cv import ScoreToCV

cfg = yaml.safe_load(open("configs/model_cv_final.yaml", encoding="utf-8"))
model = ScoreToCV(cfg).cuda().float().eval()
model.load_state_dict(torch.load("checkpoints/cv_final.pt", weights_only=False)["model"])

# 构造逐音素的谱面数组（见 render_ust.build_arrays / DEPLOY §2-3）：
# phonemes, note_pitch, phone_dur, note_dur, note_to_phone, speaker_id, lang_id, phone_mask, technique(全零)
with torch.no_grad():
    out = model(**inputs)
    T   = int(out["frame_mask"][0].sum())
    cv  = model.infer_cv(out["frame_hidden"])[0, :T].cpu().numpy()   # [T, 768] ContentVec，已反归一化
# 把 cv 和一条 f0 音高流一起喂给 SVC 后端 -> 歌声。
```

完整的 I/O 约定（每个输入数组、f0 音高流、长曲分块、RVC 接入配方）见
[docs/DEPLOY_768_sovits41_rvc.md](docs/DEPLOY_768_sovits41_rvc.md)。

## 训练 / 重定向

架构与后端无关——需要改的只有**目标特征**。要训练一个音色、或重定向到新的 ContentVec 种类（例如 so-vits 4.0
用的 256 维）：

1. 在你的音频上抽特征——`scripts/extract_contentvec.py`（768）或 `extract_contentvec256.py`（256）、
   `extract_f0.py`、`extract_notes.py`。
2. 打包对齐好的 `.npz`——`scripts/pack_npz.py`（768）或 `build_npz256.py`（256），再跑 `compute_cv_norm*.py`。
3. 训练——`python scripts/train_cv.py --config configs/model_cv_final.yaml`。
4. **用耳朵验收**——过目标后端听（cv 空间的任何指标都不能作为发布标准）。

对齐需要一个强制对齐器（本项目用的是 [HubertFA](https://github.com/qixi-oss/HubertFA)，MFA 也可以）——本仓库不含。
细节见 [docs/DEPLOY_256_sovits40.md](docs/DEPLOY_256_sovits40.md) §4。

## 仓库结构

```
src/model/          ScoreToCV (score2cv.py)、ScoreToF0、共享子模块
src/preprocessing/  IPA 音素表（210 tokens，9 语种）、ContentVec / f0 / RMVPE 抽取器
src/training/       dataset + losses
configs/            model_cv_final.yaml (768)、model_cv256.yaml (256)、model_f0_single.yaml
scripts/            特征抽取、npz 打包、训练、以及渲染 / 推理前端
docs/               各后端的部署指南（768 / 256）
checkpoints/        （把下载的 .pt 放这里）
```

## 局限（如实说明）

- **7 种语言**通过耳朵验收（中/日/英/德/法/西/意）。韩、俄因对齐质量问题被舍弃。
- 内容模型用的是**确定性的条件均值 head**——干净、稳定，但残留瑕疵会表现为辅音略微发虚 / 发糊。这是 head 的固有
  性质，不是训练不足。
- **f0 走 DAW 参数化。** 学习式 f0 模型已退役（它在大音程跳进时会"欠冲"）；请用精确音符音高 + 滑音 + 尾音颤音。
  见 DEPLOY §7。
- `scripts/render_ust.py` 只是一个**一版参考前端**——真正的 DAW 会给出精确的逐音符时值，并应把时值 / 分组 / f0
  这些旋钮自己暴露出来。

## 训练数据与许可

**代码**采用 MIT（见 [LICENSE](LICENSE)）。**权重**仅供研究 / 非商业用途。

发布的模型训练自公开歌声语料：**M4Singer**（中）、**GTSinger**（英/德/法/西/意），以及日语歌声数据库
（**kiritan_singing、PJS、Ofuton-P、Oniku、Itako、Natsume**）。其中若干仅供**非商业 / 研究用途**——例如
**M4Singer 为 CC BY-NC-SA 4.0**。发布的权重**并非**把这些数据重新以 MIT 授权。任何商业使用前，请先查阅每个
数据集各自的许可证。

日语数据库规约要求的署名(原文):『©SSS』 · 『歌声DB制作:アマノケイ 音声提供者: 霧野蒼太』 · 『DB制作:おふとんP』 · 『御丹宮くるみ歌声データべース』 —— 分别对应 東北きりたん歌唱データベース / 東北イタコ歌唱データベース(SSS 合同会社)、夏目悠李/男性歌声データベース(ATSUYA)、おふとんP歌声データベース、御丹宮くるみ歌声データべース;PJS(小口・高道;歌词来自 声優統計コーパス / 日本声優統計学会)为 CC BY-SA 4.0。

用这些权重生成的音频属于夏目悠李 DB 的「出力音声」,适用「夏目悠李の出力音声に関する利用規約」(https://ksdcm1ng.wixsite.com/njksofficial/%E8%A6%8F%E7%B4%84-rules):商用须另行取得许可,且不得用生成的音频制作音响模型或音高模型。

## 交流社群

- **QQ 群：** [1058227212](https://qun.qq.com/universal-share/share?ac=1&authKey=3uD5AoM8e50y00vhOYOZsa2VI341dBNfr07S2IK9wraewz0rcFHpSzONYJ9QrTP7&busi_data=eyJncm91cENvZGUiOiIxMDU4MjI3MjEyIiwidG9rZW4iOiJONGpqQ2MzM3h3N3BDMVBMRzZiSUFOU05YWnRnbHBxdTZDUElZYlZOSGN3VnhCaEc5eWludlJBYlltK3hkdlFwIiwidWluIjoiMjc2Njc2NDM1NSJ9&data=VyWCaG06iaMLBFcfEx_fjE2Tme2X7YvJsUIUjJ51zk6XymaED6Z6TEC_zOvAdm9q2MbzbYbpuO4ukQHZ1GBHLw&svctype=4&tempid=h5_group_info)
- **Discord：** https://discord.gg/p3fGh942fJ

## 致谢

基于 [ContentVec](https://github.com/auspicious3000/contentvec)、
[RMVPE](https://github.com/Dream-High/RMVPE)、
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)、
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，以及上述歌声语料构建。
