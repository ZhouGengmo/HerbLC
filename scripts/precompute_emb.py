"""Pre-compute the frozen-encoder support sets behind HerbLC's molecular channel.

    python scripts/precompute_emb.py --dataset tcm_suite   --modality compound
    python scripts/precompute_emb.py --dataset tcm_suite   --modality target
    python scripts/precompute_emb.py --dataset ethnobotany --modality compound
    python scripts/precompute_emb.py --dataset ethnobotany --modality target

    compound_emb.npz                compound_ids (N,)   compound_embs (N, 512)
    herb_to_compound_indices.json   {herb_id: [row into compound_emb.npz, ...]}
    target_emb.npz                  target_ids  (M,)    target_embs  (M, 1280)
    disease_to_target_indices.json  {disease_id: [row into target_emb.npz, ...]}

"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

UNIMOL_DIR = PACKAGE_ROOT / "herblc/unimol_hf"
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
MAX_RESIDUES = 1022


# encoders

_METAL = r"(?:Zn|Cu|Fe|Mn|Co|Ni|Mg|Ca|Ba|La|Al|Cr|Pb|Cd|Ag|Au|Pt|Ti|Sb|Bi|Sn|Hg|K|Na|Li)"
_NONORG = r"(?:Se|As|Si|Te|Mo|W|V)"


def normalize_smiles(smiles: str) -> str | None:
    s = smiles.strip()
    if not s:
        return None
    s = re.sub(r"(?<!\[)nH(?!\])", "[nH]", s)
    s = re.sub(r"(?<!\[)([NOSP])H(\d?)([+-])(?!\])",
               lambda m: "[%sH%s%s]" % (m.group(1), m.group(2), m.group(3)), s)
    s = re.sub(r"(?<!\[)(" + _METAL + r")([+-])(\d)(?!\])",
               lambda m: "[%s%s%s]" % (m.group(1), m.group(2), m.group(3)), s)
    s = re.sub(r"(?<!\[)([NOSPnos])([+-])(?!\])",
               lambda m: "[%s%s]" % (m.group(1), m.group(2)), s)
    s = re.sub(r"(?<!\[)(" + _NONORG + r")(H\d?)?(?![a-z\]])",
               lambda m: "[%s%s]" % (m.group(1), m.group(2) or ""), s)
    s = re.sub(r"(?<!\[)(" + _METAL + r")(?![a-z\]])", lambda m: "[%s]" % m.group(1), s)
    if "." in s:
        frags = [(f, Chem.MolFromSmiles(f)) for f in s.split(".") if f]
        parsed = [(m.GetNumHeavyAtoms(), f) for f, m in frags if m is not None]
        if parsed:
            s = max(parsed)[1]
    return s


def _unimol_batch(batch_smi, tokenizer, model, device) -> np.ndarray:
    inputs = tokenizer(batch_smi, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        out = model(**inputs, return_repr=True)
    cls = out["cls_repr"] if isinstance(out, dict) else out[0]
    return cls.cpu().float().numpy()


def encode_compounds(smiles: list[str], device: str, batch: int, ckpt: Path) -> np.ndarray:
    """Uni-Mol repr, (len(smiles), 512) """
    from herblc.unimol_hf import UnimolConfig, UnimolModel, UnimolTokenizer

    print(f"[unimol] {len(smiles)} compounds, batch={batch}, ckp={ckpt}")
    # remove_hs matches the mol_pre_no_h pretrain checkpoint.
    tokenizer = UnimolTokenizer.from_pretrained(UNIMOL_DIR, remove_hs=True)
    config = UnimolConfig.from_pretrained(UNIMOL_DIR)
    config.pretrain_path = str(ckpt)
    dev = torch.device(device)
    model = UnimolModel(config).to(dev).eval()

    chunks: list[np.ndarray] = []
    for i in range(0, len(smiles), batch):
        part = smiles[i:i + batch]
        try:
            embs = _unimol_batch(part, tokenizer, model, dev)
        except Exception as e:
            raise RuntimeError(f"SMILES at index {i}-{i + len(part)} failed Uni-Mol encoding: "
                               f"{part!r} ({type(e).__name__}: {e})") from e
        chunks.append(embs)
        if (i // batch) % 10 == 0:
            print(f"  [unimol] {i + len(part)}/{len(smiles)}", flush=True)
    return np.concatenate(chunks, axis=0)


def encode_sequences(seqs: list[str], device: str, batch: int, dtype: str,
                     hf_cache_dir: str = None) -> np.ndarray:
    """ESM-2 repr (len(seqs), 1280)"""
    from transformers import AutoModel, AutoTokenizer

    uniq = list(dict.fromkeys(seqs))
    seq2row = {s: i for i, s in enumerate(uniq)}
    print(f"[esm] {len(seqs)} targets → {len(uniq)} unique sequences, batch={batch}")

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    dev = torch.device(device)
    tok = AutoTokenizer.from_pretrained(ESM_MODEL, cache_dir=hf_cache_dir)
    model = AutoModel.from_pretrained(ESM_MODEL, torch_dtype=torch_dtype,
                                      cache_dir=hf_cache_dir).to(dev).eval()

    out = np.zeros((len(uniq), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, len(uniq), batch):
            chunk = uniq[k:k + batch]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=MAX_RESIDUES + 2, return_tensors="pt")
            ids = enc["input_ids"].to(dev)
            attn = enc["attention_mask"].to(dev)
            h = model(input_ids=ids, attention_mask=attn).last_hidden_state
            pm = attn.clone().to(h.dtype)
            pm[:, 0] = 0                                                    # CLS
            pm[torch.arange(pm.size(0), device=dev), attn.sum(dim=1) - 1] = 0  # EOS
            pooled = (h * pm.unsqueeze(-1)).sum(dim=1) / pm.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out[k:k + len(chunk)] = pooled.cpu().float().numpy()
            if (k // batch) % 50 == 0:
                print(f"  [esm] {min(k + batch, len(uniq))}/{len(uniq)}", flush=True)
    return np.stack([out[seq2row[s]] for s in seqs])


# shared helpers

def dump_json(obj, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def det_permute(entity_id: str, rows: set[int]) -> list[int]:
    """Shuffle an entity's support set."""
    base = sorted(rows)
    rng = np.random.RandomState(zlib.crc32(entity_id.encode()) & 0xFFFFFFFF)
    return [base[i] for i in rng.permutation(len(base))]


def load_mondo2mesh(sssom_path: Path) -> dict[str, set[str]]:
    """MONDO → {MESH:...}."""
    mapping: dict[str, set[str]] = {}
    with sssom_path.open(encoding="utf-8") as f:
        rows = csv.DictReader([l for l in f if not l.startswith("#")], delimiter="\t")
        for r in rows:
            if (r["predicate_id"] == "skos:exactMatch"
                    and r["subject_id"].startswith("MONDO:")
                    and r["object_id"].lower().startswith("mesh:")):
                mapping.setdefault(r["subject_id"], set()).add(
                    "MESH:" + r["object_id"].split(":", 1)[1].strip().upper())
    return mapping


def _tsv_column(path: Path, col: int) -> set[str]:
    return {line.split("\t")[col] for line in
            path.read_text(encoding="utf-8").splitlines()[1:] if line}


# TCM-suite adapters

def tcm_suite_compounds(data_dir: Path) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """compound.csv + herb-compound.csv → (compound ids, SMILES, herb → rows)."""
    hc = pd.read_csv(data_dir / "herb-compound.csv", usecols=["herbId", "id"],
                     dtype={"herbId": "Int64", "id": "Int64"})
    if hc.isna().any().any():
        raise ValueError(f"{data_dir / 'herb-compound.csv'}: nulls in herbId/id")
    herb_rows = [(int(h), int(c)) for h, c in zip(hc["herbId"], hc["id"])]
    required = {c for _, c in herb_rows}

    df = pd.read_csv(data_dir / "compound.csv", usecols=["id1", "molecularFormula"],
                     dtype={"id1": "Int64", "molecularFormula": "string"})
    if df["id1"].isna().any():
        raise ValueError(f"{data_dir / 'compound.csv'}: id1 contains nulls")
    df = df[df["id1"].astype(int).isin(required)].dropna(subset=["molecularFormula"])

    ids, smiles = [], []
    for cid, raw in zip(df["id1"].astype(int), df["molecularFormula"].astype(str)):
        smi = raw
        if Chem.MolFromSmiles(smi) is None:
            smi = normalize_smiles(raw)
            if smi is None or Chem.MolFromSmiles(smi) is None:
                continue
        ids.append(cid)
        smiles.append(smi)

    cid_to_row = {c: i for i, c in enumerate(ids)}
    index: dict[str, list[int]] = {}
    for herb, cid in herb_rows:
        if cid in cid_to_row:
            index.setdefault(str(herb), []).append(cid_to_row[cid])
    return ids, smiles, index


def tcm_suite_targets(data_dir: Path) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """target_protein-disease.csv + target_seqs.json → (target ids, sequences, disease → rows)."""
    df = pd.read_csv(data_dir / "target_protein-disease.csv",
                     usecols=["diseaseId", "geneId", "gene"],
                     dtype={"diseaseId": "string", "geneId": "Int64", "gene": "string"})
    df = df.dropna(subset=["diseaseId", "geneId", "gene"])
    df["gene_u"] = df["gene"].str.strip().str.upper()
    id2gene = dict(zip(df["geneId"].astype(int), df["gene_u"]))

    gene2seq = json.loads((data_dir / "target_seqs.json").read_text(encoding="utf-8"))
    ids = [t for t in sorted(id2gene) if id2gene[t] in gene2seq]

    tid_to_row = {t: i for i, t in enumerate(ids)}
    index: dict[str, set[int]] = {}
    for did, tid in zip(df["diseaseId"], df["geneId"].astype(int)):
        if tid in tid_to_row:
            index.setdefault(str(did), set()).add(tid_to_row[tid])
    return ids, [gene2seq[id2gene[t]] for t in ids], {d: sorted(v) for d, v in index.items()}


# Ethnobotany adapters

def ethno_compounds(data_dir: Path, hghda_dir: Path
                    ) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """compound_smiles.json + herb-component.csv → (PubChem CIDs, SMILES, herb → rows)."""
    resolved = json.loads((data_dir / "compound_smiles.json").read_text(encoding="utf-8"))
    valid = sorted((int(cid), smi) for cid, smi in resolved.items()
                   if Chem.MolFromSmiles(smi) is not None)
    ids = [c for c, _ in valid]
    smiles = [s for _, s in valid]
    cid_to_row = {c: i for i, c in enumerate(ids)}

    herbs = _tsv_column(data_dir / "positives.tsv", 0)
    rows: dict[str, set[int]] = {}
    with (hghda_dir / "herb-component.csv").open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["plant_curie"] in herbs and r["chemical_curie"].startswith("pubchem:"):
                cid = int(r["chemical_curie"].split(":", 1)[1])
                if cid in cid_to_row:
                    rows.setdefault(r["plant_curie"], set()).add(cid_to_row[cid])
    return ids, smiles, {h: det_permute(h, v) for h, v in rows.items()}


def ethno_targets(data_dir: Path, hghda_dir: Path, sssom: Path
                  ) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """target_seqs.json + target-disease.csv → (row ids, sequences, disease → rows). """
    gene2seq = json.loads((data_dir / "target_seqs.json").read_text(encoding="utf-8"))
    genes = sorted(gene2seq)
    gene_to_row = {g: i for i, g in enumerate(genes)}

    mondo2mesh = load_mondo2mesh(sssom)
    mesh2genes: dict[str, set[str]] = {}
    with (hghda_dir / "target-disease.csv").open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            disease = (r.get("DiseaseID") or "").strip().upper()
            gene = (r.get("# GeneSymbol") or r.get("GeneSymbol") or "").strip().upper()
            if disease.startswith("MESH:") and gene in gene_to_row:
                mesh2genes.setdefault(disease, set()).add(gene)

    index = {}
    for mondo in _tsv_column(data_dir / "positives.tsv", 1):
        rows = {gene_to_row[g] for mesh in mondo2mesh.get(mondo, set())
                for g in mesh2genes.get(mesh, set())}
        if rows:
            index[mondo] = det_permute(mondo, rows)
    return list(range(len(genes))), [gene2seq[g] for g in genes], index


# entry point

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["tcm_suite", "ethnobotany"])
    ap.add_argument("--modality", required=True, choices=["compound", "target"])
    ap.add_argument("--data-dir", type=Path,
                    help="the benchmark directory")
    ap.add_argument("--hghda-dir", type=Path, default=Path("./data/hda/HGHDA/src/ethnobotany"))
    ap.add_argument("--sssom", type=Path, default=None,
                    help="MONDO-to-MeSH mapping")
    ap.add_argument("--unimol-ckpt", type=Path,
                    default=Path("./checkpoints/mol_pre_no_h_220816.pt"))
    ap.add_argument("--batch", type=int, default=None,
                    help="default 64 for compounds, 8 for targets")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hf-cache-dir",
                    help="where ESM-2 is cached")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16",
                    help="ESM-2 weight dtype")
    args = ap.parse_args()

    out_dir = args.data_dir or Path("./data/hda") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = args.batch if args.batch is not None else (64 if args.modality == "compound" else 8)

    if args.dataset == "tcm_suite":
        if args.modality == "compound":
            ids, payload, index = tcm_suite_compounds(out_dir)
        else:
            ids, payload, index = tcm_suite_targets(out_dir)
    else:
        if args.modality == "compound":
            ids, payload, index = ethno_compounds(out_dir, args.hghda_dir)
        else:
            sssom = args.sssom or (out_dir / "mondo_exactmatch_mesh.sssom.tsv")
            ids, payload, index = ethno_targets(out_dir, args.hghda_dir, sssom)

    if args.modality == "compound":
        embs = encode_compounds(payload, args.device, batch, args.unimol_ckpt)
        id_key, emb_key = "compound_ids", "compound_embs"
        npz_name, json_name = "compound_emb.npz", "herb_to_compound_indices.json"
    else:
        embs = encode_sequences(payload, args.device, batch, args.dtype, args.hf_cache_dir)
        id_key, emb_key = "target_ids", "target_embs"
        npz_name, json_name = "target_emb.npz", "disease_to_target_indices.json"
    if embs.shape[0] != len(ids):
        raise RuntimeError(f"encoded {embs.shape[0]} rows for {len(ids)} ids")

    npz_path = out_dir / npz_name
    np.savez_compressed(npz_path, **{id_key: np.array(ids, dtype=np.int64),
                                     emb_key: embs.astype(np.float32)})
    print(f"[save] {npz_path}")
    dump_json(index, out_dir / json_name)
    print(f"[save] {out_dir / json_name}")


if __name__ == "__main__":
    main()
