#!/usr/bin/env python
import argparse
import json
import math
import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from transformers import AutoTokenizer
from herblc.configuration_herblc import HerbLCConfig
from herblc.dataset import (build_bag, build_global_index, load_bag,
                            load_hda_dataset, load_hti_dataset)
from herblc.modeling_herblc import HerbLCHDAModel, HerbLCHTIModel

METRIC_NAMES = ["F1", "AUROC", "AUPRC"]
TOP_K = (1, 3, 5, 10)


# HDA (Herb-Disease Association)


def _dataset_title(name: str) -> str:
    return {"tcm_suite": "TCM-suite", "ethnobotany": "Ethnobotany"}.get(name, name)


def _encode(items: list, encode_fn, tokenizer, device: torch.device, batch_size: int, max_length: int) -> torch.Tensor:
    """Run one encoder over `items` in batches; returns (len(items), dim)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            tokens = tokenizer(items[i:i + batch_size], padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            out.append(encode_fn(tokens["input_ids"], tokens["attention_mask"]).float().cpu())
    return torch.cat(out)


def _encode_unique(texts: list, encode_fn, tokenizer, device: torch.device, batch_size: int, max_length: int) -> dict:
    """Encode each distinct string once; returns {text: embedding}."""
    unique = list(dict.fromkeys(texts))
    return dict(zip(unique, _encode(unique, encode_fn, tokenizer, device, batch_size, max_length)))


@torch.no_grad()
def _score_hda_pairs(model, metadata, device, text_tokenizer, max_text_length, batch_size):
    """Fused score of every test pair"""
    herb_id2name = metadata["herb_id2name"]
    disease_id2name = metadata["disease_id2name"]
    test_pairs = metadata["test_pairs"]
    herb_to_compounds = metadata["herb_to_compound_indices"]
    disease_to_targets = metadata["disease_to_target_indices"]
    cap = metadata["max_components"]

    herb_names = [herb_id2name[str(h)] for h, _ in test_pairs]
    disease_names = [disease_id2name[str(d)] for _, d in test_pairs]
    print(f"  encoding {len(set(herb_names))} herbs and {len(set(disease_names))} diseases "
          f"behind {len(test_pairs)} test pairs")
    herb_emb = _encode_unique(herb_names, model.encode_herb, text_tokenizer,
                              device, batch_size, max_text_length)
    disease_emb = _encode_unique(disease_names, model.encode_disease, text_tokenizer,
                                 device, batch_size, max_text_length)

    scores = []
    for start in range(0, len(test_pairs), batch_size):
        chunk = test_pairs[start:start + batch_size]
        herb_ids = [str(h) for h, _ in chunk]
        disease_ids = [str(d) for _, d in chunk]
        herb_text = torch.stack([herb_emb[herb_id2name[h]] for h in herb_ids]).to(device)
        disease_text = torch.stack([disease_emb[disease_id2name[d]] for d in disease_ids]).to(device)
        comp_embs, comp_mask = build_bag(
            herb_ids, metadata["compound_emb_lookup"], herb_to_compounds, cap)
        targ_embs, targ_mask = build_bag(
            disease_ids, metadata["target_emb_lookup"], disease_to_targets, cap)
        targ_gidx = None
        if model.target_idf_weight > 0:
            targ_gidx = build_global_index(
                disease_ids, disease_to_targets, targ_mask.shape[1], cap).to(device)
        fused = model._dual_channel_score(
            herb_text, comp_embs, comp_mask, disease_text, targ_embs, targ_mask,
            targ_gidx=targ_gidx, paired=True)
        scores.extend(fused.float().cpu().tolist())
    return scores


def evaluate_hda_fold(model, metadata: dict, device: torch.device, text_tokenizer,
                      max_text_length: int = 512, batch_size: int = 256) -> dict:
    scores = _score_hda_pairs(model, metadata, device, text_tokenizer, max_text_length, batch_size)

    labels_arr = np.array(metadata["test_labels"])
    scores_arr = np.array(scores)
    auroc = roc_auc_score(labels_arr, scores_arr)
    auprc = average_precision_score(labels_arr, scores_arr)
    prec_curve, rec_curve, _ = precision_recall_curve(labels_arr, scores_arr)
    f1_values = 2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1] + 1e-10)

    return {
        "dataset_name": metadata["dataset_name"],
        "F1": float(f1_values.max()),
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
    }


def print_hda_results(results: dict, fold: int = 0) -> None:
    dataset_name = results["dataset_name"]

    print()
    print("=" * 70)
    print(f"  HDA Results ({_dataset_title(dataset_name)}, fold {fold})")
    print("=" * 70)
    header = f"{'Method':<16}" + "".join(f"{metric:>12}" for metric in METRIC_NAMES)
    print(header)
    print("-" * len(header))
    row = f"{'HerbLC':<16}" + "".join(f"{results[metric]:>12.4f}" for metric in METRIC_NAMES)
    print(row)
    print("=" * 70)
    print()


def _collect_fold_results(results_dir: str) -> dict:
    """One run's per-fold results, written either into results_dir itself or into fold_i/."""
    fold_results = {}
    for i in range(5):
        for path in (os.path.join(results_dir, f"hda_results_fold{i}.json"),
                     os.path.join(results_dir, f"fold_{i}", f"hda_results_fold{i}.json")):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    fold_results[i] = json.load(f)
                break
    return fold_results


def aggregate_hda_folds(results_dir: str) -> None:
    fold_results = _collect_fold_results(results_dir)
    if not fold_results:
        raise ValueError(f"no hda_results_fold*.json under {results_dir}")

    fold_list = [fold_results[i] for i in sorted(fold_results)]
    dataset_name = fold_list[0]["dataset_name"]

    print()
    print("=" * 80)
    print(f"  HDA Aggregation ({_dataset_title(dataset_name)}, {len(fold_list)} folds found)")
    print("=" * 80)
    header = f"{'Method':<16}" + "".join(f"{metric:>16}" for metric in METRIC_NAMES)
    print(header)
    print("-" * len(header))

    row = f"{'HerbLC':<16}"
    for metric in METRIC_NAMES:
        values = [result[metric] for result in fold_list]
        row += f"{np.mean(values):>8.4f}+/-{np.std(values):<5.4f}"
    print(row)

    print("=" * 80)
    print()
    print("Per-fold breakdown:")
    header = f"{'Fold':<8}" + "".join(f"{metric:>12}" for metric in METRIC_NAMES)
    print(header)
    for i in sorted(fold_results):
        result = fold_results[i]
        row = f"{'fold_' + str(i):<8}" + "".join(f"{result[metric]:>12.4f}" for metric in METRIC_NAMES)
        print(row)
    print()


# HTI (Herb-Target Interaction)


def compute_hr_ndcg(indices_sort_top, num_positives, top_k):
    """Compute HR@K and NDCG@K for a single herb (HTINet2 protocol)."""
    ndcg_max = []
    cumsum = 0.0
    for i in range(top_k):
        cumsum += 1.0 / math.log(i + 2)
        ndcg_max.append(cumsum)

    max_hr = min(top_k, num_positives) * 1.0
    max_ndcg = ndcg_max[min(top_k, num_positives) - 1]

    hr_count = 0.0
    ndcg_sum = 0.0
    for rank, idx in enumerate(indices_sort_top[:top_k]):
        if idx < num_positives:
            hr_count += 1.0
            ndcg_sum += 1.0 / math.log(rank + 2)

    return hr_count / max_hr, ndcg_sum / max_ndcg


def evaluate_hti(model, metadata, device, text_tokenizer, protein_tokenizer,
                 max_text_length=512, max_protein_length=1024,
                 batch_size_text=64, batch_size_protein=8, top_k_list=TOP_K):
    herb_id2name = metadata["herb_id2name"]
    gene_id2name = metadata["gene_id2name"]
    gene_id2seq = metadata["gene_id2seq"]

    herb_ids_ordered = sorted(herb_id2name)
    gene_ids_ordered = sorted(gene_id2name)
    herb_pos = {h: i for i, h in enumerate(herb_ids_ordered)}
    gene_pos = {g: j for j, g in enumerate(gene_ids_ordered)}

    print(f"  encoding {len(herb_ids_ordered)} herbs")
    herb_embeddings = _encode([herb_id2name[h] for h in herb_ids_ordered], model.encode_text,
                              text_tokenizer, device, batch_size_text, max_text_length).numpy()

    print(f"  encoding {len(gene_ids_ordered)} targets")
    # a target without a sequence stays in the candidate pool
    gene_seqs_ordered = [gene_id2seq.get(g, "X") for g in gene_ids_ordered]
    gene_embeddings = _encode(gene_seqs_ordered, model.encode_protein, protein_tokenizer,
                              device, batch_size_protein, max_protein_length).numpy()

    sim_matrix = herb_embeddings @ gene_embeddings.T  # (herbs, targets)
    return _retrieval_metrics(
        sim_matrix, metadata["testing_herb_set"], metadata["training_herb_set"],
        gene_ids_ordered, herb_pos, gene_pos, top_k_list
    )


def _retrieval_metrics(sim_matrix, eval_herb_set, train_herb_set, gene_ids_ordered,
                       herb_pos, gene_pos, top_k_list):
    all_genes = set(gene_ids_ordered)
    results = {f"HR@{k}": [] for k in top_k_list}
    results.update({f"NDCG@{k}": [] for k in top_k_list})

    for herb_id in eval_herb_set:
        positive_genes = list(eval_herb_set[herb_id])
        num_pos = len(positive_genes)
        if num_pos == 0:
            continue

        train_pos = train_herb_set.get(herb_id, set())
        negative_genes = list(all_genes - train_pos - set(positive_genes))

        # positives first, then negatives
        candidate_genes = positive_genes + negative_genes
        candidate_scores = sim_matrix[herb_pos[herb_id]][[gene_pos[g] for g in candidate_genes]]

        for k in top_k_list:
            if len(candidate_scores) <= k:
                top_indices = np.argsort(-candidate_scores)
            else:
                top_indices_unsorted = np.argpartition(-candidate_scores, k)[:k]
                top_indices = top_indices_unsorted[
                    np.argsort(-candidate_scores[top_indices_unsorted])
                ]

            hr, ndcg = compute_hr_ndcg(top_indices, num_pos, k)
            results[f"HR@{k}"].append(hr)
            results[f"NDCG@{k}"].append(ndcg)

    final = {}
    for metric, values in results.items():
        final[metric] = round(np.mean(values), 4)
    return final


def print_hti_results(results: dict, top_k_list=TOP_K) -> None:
    header = f"{'Method':<20}" + "".join(f"{'HR@'+str(k):<10}{'NDCG@'+str(k):<10}"
                                         for k in top_k_list)
    print()
    print("=" * len(header))
    print("  HTI Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'HerbLC':<20}" + "".join(f"{results[f'HR@{k}']:<10}{results[f'NDCG@{k}']:<10}"
                                      for k in top_k_list))
    print("=" * len(header))
    print()


# Entry point


def parse_args():
    parser = argparse.ArgumentParser(description="HerbLC evaluation")
    parser.add_argument("--task", choices=["hda", "hti"])
    parser.add_argument("--checkpoint",
                        help="the run dir itself or one of its checkpoint-N subdir")
    parser.add_argument("--data-dir")
    parser.add_argument("--hda-cv-fold", type=int, default=0, help="cross-validation fold, 0-4")
    parser.add_argument("--output-dir", help="where to write the results; defaults to --checkpoint")
    parser.add_argument("--hf-cache-dir",
                        help="where the pretrained encoders are cached")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="pairs per step for HDA, herb names per step for HTI")
    parser.add_argument("--protein-batch-size", type=int, default=8,
                        help="target sequences per step, HTI only")
    parser.add_argument("--aggregate", action="store_true",
                        help="average the 5 HDA folds")
    parser.add_argument("--results-dir", help="with --aggregate, the dir holding fold_0..4")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        if args.results_dir is None:
            raise ValueError("--aggregate requires --results-dir")
        aggregate_hda_folds(args.results_dir)
        return
    for name in ("task", "checkpoint", "data_dir"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name.replace('_', '-')} is required")

    out_dir = args.output_dir or args.checkpoint
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = HerbLCConfig.from_pretrained(args.checkpoint)
    print(config)
    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model, cache_dir=args.hf_cache_dir)
    text_tokenizer.padding_side = "left"

    model_cls = HerbLCHDAModel if args.task == "hda" else HerbLCHTIModel
    model = model_cls.from_pretrained(args.checkpoint, backbone_cache_dir=args.hf_cache_dir)
    model.eval()
    model.to(device)

    if args.task == "hda":
        _, metadata = load_hda_dataset(args.data_dir, cv_fold=args.hda_cv_fold)
        compound_embs, herb_to_compounds = load_bag(
            args.data_dir, "compound", "herb", config.hda_compound_emb_dim)
        target_embs, disease_to_targets = load_bag(
            args.data_dir, "target", "disease", config.hda_target_emb_dim)
        metadata.update({
            "compound_emb_lookup": compound_embs,
            "herb_to_compound_indices": herb_to_compounds,
            "target_emb_lookup": target_embs,
            "disease_to_target_indices": disease_to_targets,
            "max_components": config.hda_max_components,
        })
        results = evaluate_hda_fold(
            model=model, metadata=metadata, device=device,
            text_tokenizer=text_tokenizer, max_text_length=args.max_text_length,
            batch_size=args.batch_size,
        )
        print_hda_results(results, fold=args.hda_cv_fold)
        results_path = os.path.join(out_dir, f"hda_results_fold{args.hda_cv_fold}.json")
    else:
        _, _, _, metadata = load_hti_dataset(args.data_dir)
        protein_tokenizer = AutoTokenizer.from_pretrained(config.protein_model,
                                                          cache_dir=args.hf_cache_dir)
        results = evaluate_hti(
            model=model, metadata=metadata, device=device,
            text_tokenizer=text_tokenizer, protein_tokenizer=protein_tokenizer,
            max_text_length=args.max_text_length, max_protein_length=args.max_protein_length,
            batch_size_text=args.batch_size, batch_size_protein=args.protein_batch_size,
        )
        print_hti_results(results)
        results_path = os.path.join(out_dir, "hti_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()
