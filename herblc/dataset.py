import json
import os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


# HDA (Herb-Disease Association)


class HDADataset(TorchDataset):
    """HDA dataset for TCM-suite / Ethnobotany."""

    def __init__(self, positive_pairs: list, herb_id2name: dict, disease_id2name: dict):
        self.pairs = positive_pairs      # [[herb_id, disease_id], ...]
        self.herb_id2name = herb_id2name
        self.disease_id2name = disease_id2name

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        h_id, d_id = self.pairs[idx]
        return {
            "herb_name": self.herb_id2name[str(h_id)],
            "disease_name": self.disease_id2name[str(d_id)],
            "herb_id": str(h_id),
            "disease_id": str(d_id),
        }


class HDADataCollator:
    """Build both channels of a batch."""

    def __init__(
        self,
        text_tokenizer,
        max_text_length: int = 512,
        compound_emb_lookup: np.ndarray = None,
        herb_to_compound_indices: dict = None,
        target_emb_lookup: np.ndarray = None,
        disease_to_target_indices: dict = None,
        max_components: Optional[int] = None,
        emit_global_idx: bool = False,
    ):
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        if (compound_emb_lookup is None) ^ (herb_to_compound_indices is None):
            raise ValueError("compound_emb_lookup and herb_to_compound_indices must be provided together")
        if (target_emb_lookup is None) ^ (disease_to_target_indices is None):
            raise ValueError("target_emb_lookup and disease_to_target_indices must be provided together")
        self.compound_emb_lookup = compound_emb_lookup  # (N_compound, compound_dim)
        self.herb_to_compound_indices = herb_to_compound_indices or {}
        self.target_emb_lookup = target_emb_lookup      # (N_target, target_dim)
        self.disease_to_target_indices = disease_to_target_indices or {}
        self.max_components = max_components
        self.emit_global_idx = emit_global_idx

    def __call__(self, batch: list) -> dict:
        herb_ids = [item["herb_id"] for item in batch]
        disease_ids = [item["disease_id"] for item in batch]

        herb_enc = self.text_tokenizer(
            [item["herb_name"] for item in batch], padding=True, truncation=True,
            max_length=self.max_text_length, return_tensors="pt",
        )
        disease_enc = self.text_tokenizer(
            [item["disease_name"] for item in batch], padding=True, truncation=True,
            max_length=self.max_text_length, return_tensors="pt",
        )

        result = {
            "herb_input_ids": herb_enc["input_ids"],
            "herb_attention_mask": herb_enc["attention_mask"],
            "disease_input_ids": disease_enc["input_ids"],
            "disease_attention_mask": disease_enc["attention_mask"],
            "herb_ids": herb_ids,
            "disease_ids": disease_ids,
            "return_loss": True,  # Trainer takes the loss the model returns
        }
        if self.compound_emb_lookup is not None:
            embs, mask = build_bag(
                herb_ids, self.compound_emb_lookup, self.herb_to_compound_indices, self.max_components)
            result["herb_compound_embs"] = embs
            result["herb_compound_mask"] = mask
            if self.emit_global_idx:
                result["compound_global_index"] = build_global_index(
                    herb_ids, self.herb_to_compound_indices, embs.shape[1], self.max_components)
        if self.target_emb_lookup is not None:
            embs, mask = build_bag(
                disease_ids, self.target_emb_lookup, self.disease_to_target_indices, self.max_components)
            result["disease_target_embs"] = embs
            result["disease_target_mask"] = mask
            if self.emit_global_idx:
                result["target_global_index"] = build_global_index(
                    disease_ids, self.disease_to_target_indices, embs.shape[1], self.max_components)
        return result


def build_bag(ids: list, lookup: np.ndarray, id_to_indices: dict, max_components: Optional[int] = None):
    per_idx = [id_to_indices.get(str(x), []) for x in ids]
    # an entity with no components falls back to the language channel alone
    if max_components is not None:
        per_idx = [idx[:max_components] for idx in per_idx]
    n_cols = max(max((len(idx) for idx in per_idx), default=0), 1)
    embs = torch.zeros(len(ids), n_cols, lookup.shape[1], dtype=torch.float32)
    mask = torch.zeros(len(ids), n_cols, dtype=torch.bool)
    for i, idx in enumerate(per_idx):
        if idx:
            embs[i, : len(idx)] = torch.from_numpy(lookup[idx])
            mask[i, : len(idx)] = True
    return embs, mask


def build_global_index(ids: list, id_to_indices: dict, n_cols: int,
                       max_components: Optional[int] = None) -> torch.Tensor:
    out = torch.zeros(len(ids), n_cols, dtype=torch.long)
    for i, x in enumerate(ids):
        idx = id_to_indices.get(str(x), [])
        idx = (idx[:max_components] if max_components is not None else idx)[:n_cols]
        if idx:
            out[i, : len(idx)] = torch.tensor(idx, dtype=torch.long)
    return out


def load_bag(data_dir, kind, owner, expected_dim):
    """Read one entity's measure"""
    npz_path = os.path.join(data_dir, f"{kind}_emb.npz")
    json_path = os.path.join(data_dir, f"{owner}_to_{kind}_indices.json")
    for path in (npz_path, json_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run scripts/precompute_emb.py --modality {kind} first.")
    with open(json_path) as f:
        index = json.load(f)
    with np.load(npz_path) as z:
        embs = z[f"{kind}_embs"].astype(np.float32)
    if embs.shape[1] != expected_dim:
        raise ValueError(f"{npz_path} has dim {embs.shape[1]}, expected {expected_dim}")
    print(f"  {kind} bag: {embs.shape[0]} embeddings of dim {embs.shape[1]}, "
          f"{len(index)} {owner}s with at least one {kind}")
    return embs, index

def load_hda_dataset(data_dir: str, cv_fold: int = 0):
    """Load processed HDA data for one CV fold."""
    with open(os.path.join(data_dir, "herb_id2name.json"), "r") as f:
        herb_id2name = json.load(f)
    with open(os.path.join(data_dir, "disease_id2name.json"), "r") as f:
        disease_id2name = json.load(f)
    with open(os.path.join(data_dir, f"fold_{cv_fold}.json"), "r") as f:
        fold_data = json.load(f)

    train_positives = fold_data["train_positives"]
    train_all_rating = {str(k): {str(vv) for vv in v} for k, v in fold_data["train_all_rating"].items()}
    dataset_name = os.path.basename(os.path.normpath(data_dir))

    train_dataset = HDADataset(train_positives, herb_id2name, disease_id2name)
    print(f"  HDA {dataset_name} fold {cv_fold}: {len(train_positives)} train positives, "
          f"{len(fold_data['test_pairs'])} test pairs")

    metadata = {
        "herb_id2name": herb_id2name,
        "disease_id2name": disease_id2name,
        "all_rating": train_all_rating,
        "test_pairs": fold_data["test_pairs"],
        "test_labels": fold_data["test_labels"],
        "cv_fold": cv_fold,
        "dataset_name": dataset_name,
    }
    return train_dataset, metadata


# HTI (Herb-Target Interaction)


def _load_hti_name2id(path):
    """Load herb and gene mappings."""
    herb_id2name, gene_id2name = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name, idx, etype = line.strip().split("\t")
            idx = int(idx)
            if etype == "herb":
                herb_id2name[idx] = name
            elif etype == "gene":
                gene_id2name[idx] = name
    return herb_id2name, gene_id2name


class HTIDataset(TorchDataset):

    def __init__(self, herb_set, herb_id2name, gene_id2seq):
        self.herb_id2name = herb_id2name
        self.gene_id2seq = gene_id2seq

        # Expand to (herb_id, gene_id) pairs
        self.pairs = []
        for herb_id, gene_ids in herb_set.items():
            for gene_id in gene_ids:
                if gene_id in gene_id2seq:  # skip genes without sequence
                    self.pairs.append((herb_id, gene_id))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        herb_id, gene_id = self.pairs[idx]
        return {
            "herb_name": self.herb_id2name[herb_id],
            "gene_seq": self.gene_id2seq[gene_id],
            "herb_id": herb_id,
            "gene_id": gene_id,
        }


class HTIDataCollator:
    """Collator for HTI dataset."""

    def __init__(self, text_tokenizer, protein_tokenizer,
                 max_text_length: int = 512, max_protein_length: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_text_length = max_text_length
        self.max_protein_length = max_protein_length

    def __call__(self, batch):
        herb_tokens = self.text_tokenizer(
            [item["herb_name"] for item in batch], padding=True, truncation=True,
            max_length=self.max_text_length, return_tensors="pt"
        )
        gene_tokens = self.protein_tokenizer(
            [item["gene_seq"] for item in batch], padding=True, truncation=True,
            max_length=self.max_protein_length, return_tensors="pt"
        )
        return {
            "herb_input_ids": herb_tokens["input_ids"],
            "herb_attention_mask": herb_tokens["attention_mask"],
            "gene_input_ids": gene_tokens["input_ids"],
            "gene_attention_mask": gene_tokens["attention_mask"],
            "herb_ids": torch.tensor([item["herb_id"] for item in batch], dtype=torch.long),
            "gene_ids": torch.tensor([item["gene_id"] for item in batch], dtype=torch.long),
            "return_loss": True,  # Trainer takes the loss the model returns
        }


def load_hti_dataset(data_dir):
    """Load the HTI benchmark from one self-contained directory."""
    herb_id2name, gene_id2name = _load_hti_name2id(os.path.join(data_dir, "name2id.txt"))
    print(f"  HTI: {len(herb_id2name)} herbs, {len(gene_id2name)} genes")

    with open(os.path.join(data_dir, "gene_to_sequence.json"), "r", encoding="utf-8") as f:
        gene_seq_data = json.load(f)

    # Map gene_id → protein sequence
    gene_id2seq = {}
    missing = []
    for gene_id, gene_name in gene_id2name.items():
        if gene_name in gene_seq_data and gene_seq_data[gene_name].get("sequence"):
            gene_id2seq[gene_id] = gene_seq_data[gene_name]["sequence"]
        else:
            missing.append(gene_name)
    print(f"  Gene sequences: {len(gene_id2seq)}/{len(gene_id2name)} mapped"
          f" ({len(missing)} missing)")

    # Load splits
    training_herb_set, _, _ = np.load(
        os.path.join(data_dir, "training_set.npy"), allow_pickle=True
    )
    val_herb_set, _, _ = np.load(
        os.path.join(data_dir, "val_set.npy"), allow_pickle=True
    )
    testing_herb_set, _, _ = np.load(
        os.path.join(data_dir, "testing_set.npy"), allow_pickle=True
    )
    known = np.load(
        os.path.join(data_dir, "set_all.npy"), allow_pickle=True
    ).item()
    train_only = {herb_id: set(gene_ids) for herb_id, gene_ids in training_herb_set.items()}
    heldout_rating = {
        herb_id: heldout
        for herb_id, gene_ids in known.items()
        if (heldout := set(gene_ids) - train_only.get(herb_id, set()))
    }

    train_pairs = sum(len(v) for v in training_herb_set.values())
    val_pairs = sum(len(v) for v in val_herb_set.values())
    test_pairs = sum(len(v) for v in testing_herb_set.values())
    print(f"  Splits: train={train_pairs}, val={val_pairs}, test={test_pairs}")

    # Create datasets
    train_dataset = HTIDataset(training_herb_set, herb_id2name, gene_id2seq)
    val_dataset = HTIDataset(val_herb_set, herb_id2name, gene_id2seq)
    test_dataset = HTIDataset(testing_herb_set, herb_id2name, gene_id2seq)

    metadata = {
        "herb_id2name": herb_id2name,
        "gene_id2name": gene_id2name,
        "gene_id2seq": gene_id2seq,
        "all_rating": train_only,
        "heldout_rating": heldout_rating,
        "training_herb_set": training_herb_set,
        "testing_herb_set": testing_herb_set,
        "val_herb_set": val_herb_set,
    }

    return train_dataset, val_dataset, test_dataset, metadata
