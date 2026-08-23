"""Cross-VLM at retrieval level (bounded, matched): Qwen2-VL-2B captions a stratified
100/class D1 gallery subset + 100/class clean-D2 query subset (seed 42), CLIP-text
retrieval, caption-only DA. Compared to Florence caption-only DA on the SAME subset
(cached Florence caption embeddings), so the ONLY difference is the VLM. Resumable."""
from pathlib import Path
import json, numpy as np, torch, open_clip, random
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]; RES = ROOT / "results"
DEV = "cuda:0"; DEDUP = 0.95; EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CANON = {'aeromoniasis': 'aeromoniasis', 'bacterial diseases - aeromoniasis': 'aeromoniasis',
         'bacterial gill disease': 'bacterial_gill', 'bacterial red disease': 'bacterial_red',
         'fungal diseases saprolegniasis': 'fungal', 'parasitic diseases': 'parasitic',
         'viral diseases white tail disease': 'viral_white_tail',
         'viral white tail disease': 'viral_white_tail', 'healthy fish': 'healthy', 'eus': 'EUS'}


def canon(s):
    s = s.lower().replace('_', ' ').strip()
    for k, v in CANON.items():
        if k in s:
            return v
    return 'UNKNOWN'


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


random.seed(42)


def strat(paths, per):
    by = {}
    for p in paths:
        by.setdefault(canon(p.stem), []).append(p)
    out = []
    for c in sorted(by):
        if c in ('EUS', 'UNKNOWN'):
            continue
        random.shuffle(by[c])
        out += by[c][:per]
    return out


zq = np.load(RES / "_cache_d2_query.npz", allow_pickle=True); qv_all = zq["visual"]
zg = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True); gv_all = zg["visual"]
d2paths = sorted(p for p in (ROOT / "data/datasets/D2").rglob("*") if p.suffix.lower() in EXTS)
clean_mask = (qv_all @ gv_all.T).max(1) < DEDUP
d2clean = [p for p, c in zip(d2paths, clean_mask) if c]
d1paths = sorted((ROOT / "data/datasets/D1/train").glob("*.jpg"))
gal = strat(d1paths, 100); qry = strat(d2clean, 100)
print(f"[1/3] gallery {len(gal)} + query {len(qry)} = {len(gal) + len(qry)} images", flush=True)

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
mdl = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16).to(DEV).eval()


def cap(p):
    msgs = [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": "Describe this image in detail."}]}]
    t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[t], images=[Image.open(p).convert("RGB")], return_tensors="pt").to(DEV)
    with torch.no_grad():
        g = mdl.generate(**inp, max_new_tokens=96, do_sample=False)
    return proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, g)],
                             skip_special_tokens=True)[0].strip()


CACHE = RES / "qwen_subset_captions.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for prior in ["qwen_d1_captions.json", "qwen_d2_captions.json"]:
    if (RES / prior).exists():
        cache.update({k: v for k, v in json.loads((RES / prior).read_text()).items() if k not in cache})
allp = gal + qry
print(f"[2/3] captioning ({sum(1 for p in allp if p.name not in cache)} new) ...", flush=True)
for i, p in enumerate(tqdm(allp)):
    if p.name not in cache:
        try:
            cache[p.name] = cap(p)
        except Exception:
            cache[p.name] = ""
    if i % 50 == 0:
        CACHE.write_text(json.dumps(cache))
CACHE.write_text(json.dumps(cache))
del mdl; torch.cuda.empty_cache()

print("[3/3] CLIP-text retrieval DA ...", flush=True)
clip, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEV)
clip.eval(); tok = open_clip.get_tokenizer("ViT-B-32")


def et(t):
    with torch.no_grad():
        f = clip.encode_text(tok([t or "fish"]).to(DEV))
        return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float32)


gql = np.array([canon(p.stem) for p in gal]); gqe = l2(np.array([et(cache.get(p.name, "")) for p in gal]))
qql = np.array([canon(p.stem) for p in qry]); qqe = l2(np.array([et(cache.get(p.name, "")) for p in qry]))
qwen_da = round(float((gql[(qqe @ gqe.T).argmax(1)] == qql).mean()), 4)

gc_all = zg["caption"]; name2gc = {p.name: gc_all[i] for i, p in enumerate(d1paths)}
qc_all = zq["caption"]; name2qc = {p.name: qc_all[i] for i, p in enumerate(d2paths)}
fge = l2(np.array([name2gc[p.name] for p in gal])); fqe = l2(np.array([name2qc[p.name] for p in qry]))
flor_da = round(float((gql[(fqe @ fge.T).argmax(1)] == qql).mean()), 4)

out = {'design': 'stratified 100/class matched subset, seed 42, caption-only k=1 retrieval, 7 classes',
       'n_gallery': len(gal), 'n_query': len(qry),
       'florence_caption_only_DA': flor_da, 'qwen_caption_only_DA': qwen_da,
       'florence_full_clean_D2_ref': 0.4954, 'max_new_tokens': 96}
json.dump(out, open(RES / "qwen_pipeline_retrieval.json", "w"), indent=1)
print("RESULT:", json.dumps(out, indent=1), flush=True)
