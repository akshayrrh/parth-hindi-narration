"""Render the Hindi female narration with AI4Bharat Indic Parler-TTS (Apache-2.0, gated on Hugging Face).

Voice: the model's Hindi speaker "Divya" in a sweet, gentle style, chosen by the owner on 12 Nov 2026.
Same clip names as tools/render_narration.py (verses, r-<id> reflections, said-<k|a|s|d>).
Slow on a CPU (about twelve times slower than real time), so it is resumable and shardable.
Needs: pip install torch torchaudio git+https://github.com/huggingface/parler-tts.git soundfile, ffmpeg,
and HF_TOKEN in the environment (an account that has been granted access to the model).
Usage: python3 tools/render_parler.py OUT_DIR [--chapter 12] [--shard 0/20] [--seconds 3000] [--reflections-only]
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
MODEL = "ai4bharat/indic-parler-tts"
VOICE = ("Divya speaks in a sweet, warm and gentle voice, slowly and calmly, with a soft, devotional tone. "
         "The recording is very clear and close, with no background noise.")
SEED = 7

def speech(text):  # as in render_narration.py: drop verse-number markers, read glosses as asides
    t = re.sub(r"\([०-९0-9\s,\-–]+\)", " ", text)
    t = re.sub(r"\(([^()]*?)-\)\s*", r"\1-", t)  # (कर्म-)योगी reads as one word
    t = re.sub(r"[()\[\]—–]", ", ", t)
    t = re.sub(r"[‘’“”\"=]", "", t)
    return re.sub(r"\s+", " ", re.sub(r"(\s*,)+", ",", t)).strip(" ,")

def pieces(text, limit=170):
    """Sentences grouped into pieces the model reads reliably."""
    parts, out = [p.strip() for p in re.split(r"(?<=[।;?!])\s+", text) if p.strip()], []
    for p in parts:
        while len(p) > limit and "," in p[:limit]:
            cut = p.rindex(",", 0, limit) + 1
            out.append(p[:cut].strip()); p = p[cut:].strip()
        if out and len(out[-1]) + len(p) < limit and not out[-1].endswith(","):
            out[-1] += " " + p
        else:
            out.append(p)
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pack = json.loads((ROOT / "data/feed/gita-hi-sapre-1916.json").read_text(encoding="utf-8"))
    items = json.loads((ROOT / "data/reflections/hi.json").read_text(encoding="utf-8"))["items"]
    jobs = [("said-k", "भगवान ने कहा।"), ("said-a", "अर्जुन ने कहा।"), ("said-s", "संजय ने कहा।"), ("said-d", "महाराज धृतराष्ट्र ने कहा।")]
    if "--reflections-only" in sys.argv:
        jobs = []
    jobs += [("r-" + k, f'{v["title"]}। {v["reflection"]} {v["prompt"]}') for k, v in sorted(items.items())]
    if "--reflections-only" not in sys.argv:
        jobs += [(vid, text) for vid, text in zip(pack["order"], pack["t"]) if text]
    if arg("--chapter"):
        jobs = [j for j in jobs if j[0].startswith(arg("--chapter") + ".") or j[0].startswith("said-")]
    i, n = map(int, arg("--shard", "0/1").split("/"))
    jobs = [j for j in jobs[i::n] if not (OUT / (j[0] + ".mp3")).exists()]
    deadline = time.time() + float(arg("--seconds", "1e9"))
    token = os.environ["HF_TOKEN"]
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL, token=token).eval()
    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    d = dtok(VOICE, return_tensors="pt")
    sr, gap = model.config.sampling_rate, None
    for name, text in jobs:
        if time.time() > deadline:
            break
        audio = []
        for piece in pieces(speech(text)):
            torch.manual_seed(SEED)
            q = tok(piece, return_tensors="pt")
            with torch.no_grad():
                a = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                   prompt_input_ids=q.input_ids, prompt_attention_mask=q.attention_mask)
            audio += [a.cpu().numpy().squeeze(), np.zeros(int(sr * 0.3), dtype=np.float32)]
        wav = OUT / (name + ".wav")
        sf.write(wav, np.concatenate(audio[:-1]), sr)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", "loudnorm=I=-19:TP=-2", "-ac", "1",
                        "-ar", "24000", "-b:a", "48k", str(OUT / (name + ".mp3"))], check=True)
        wav.unlink()
        print(name, flush=True)
    print("left", sum(1 for j in jobs if not (OUT / (j[0] + ".mp3")).exists()), flush=True)

main()
