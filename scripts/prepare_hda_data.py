"""Turn the raw herb-disease releases into the directory train.py reads.

    python scripts/prepare_hda_data.py --dataset tcm_suite
    python scripts/prepare_hda_data.py --dataset ethnobotany

Writes into ``--data-dir`` (default ``./data/hda/<dataset>``):

    fold_{0..4}.json      
    herb_id2name.json     
    disease_id2name.json
"""

import argparse
import csv
import json
import os
import random

NEG_SEED, N_FOLDS = 42, 5


# shared helpers

def save_json(path: str, obj, indent=None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_csv(path: str):
    return csv.DictReader(open(path, encoding="utf-8", errors="replace"))


# TCM-suite

def load_herb_names(herb_csv: str) -> dict[str, str]:
    herb_id2name: dict[str, str] = {}
    for row in read_csv(herb_csv):
        english = row.get("englishName", "").strip()
        latin = row.get("latinName", "").strip()
        if english and latin:
            name = f"{english} ({latin})"
        else:
            name = english or latin or f"Herb {row['id1']}"
        herb_id2name[row["id1"]] = name
    return herb_id2name


def load_cv_fold(cv_dir: str, idx: int) -> tuple[list[list[int]], list[list[int]], list[int]]:
    fold_num = idx + 1
    fold_dir = os.path.join(cv_dir, f"cross-validation{fold_num}")
    test_file = os.path.join(fold_dir, f"test{fold_num}.csv")
    # fold 3 name mistake
    if not os.path.exists(test_file) and fold_num == 3:
        test_file = os.path.join(fold_dir, "test2.csv")

    train_positives: list[list[int]] = []
    with open(os.path.join(fold_dir, f"train{fold_num}.csv"), encoding="utf-8") as f:
        for line in f:
            herb_id, disease_id, rating = line.strip().split(",")
            if float(rating) == 1.0:
                train_positives.append([int(herb_id), int(disease_id)])

    test_pairs: list[list[int]] = []
    test_labels: list[int] = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            herb_id, disease_id, rating = line.strip().split(",")
            test_pairs.append([int(herb_id), int(disease_id)])
            test_labels.append(int(float(rating)))
    return train_positives, test_pairs, test_labels


def prepare_tcm_suite(data_dir: str) -> None:
    herb_id2name = load_herb_names(os.path.join(data_dir, "herb.csv"))
    disease_id2name = {row["id1"]: row["name"]
                       for row in read_csv(os.path.join(data_dir, "disease.csv"))}

    cv_dir = os.path.join(data_dir, "cross-validation")
    seen_herbs: set[str] = set()
    seen_diseases: set[str] = set()
    for idx in range(N_FOLDS):
        train_positives, test_pairs, test_labels = load_cv_fold(cv_dir, idx)
        train_all_rating: dict[str, list[int]] = {}
        for herb_id, disease_id in train_positives:
            train_all_rating.setdefault(str(herb_id), []).append(disease_id)
        for herb_id, disease_id in train_positives + test_pairs:
            seen_herbs.add(str(herb_id))
            seen_diseases.add(str(disease_id))
        save_json(os.path.join(data_dir, f"fold_{idx}.json"), {
            "train_positives": train_positives,
            "test_pairs": test_pairs,
            "test_labels": test_labels,
            "train_all_rating": train_all_rating,
        })

    for herb_id in sorted(seen_herbs - set(herb_id2name)):
        herb_id2name[herb_id] = f"Herb {herb_id}"
    for disease_id in sorted(seen_diseases - set(disease_id2name)):
        disease_id2name[disease_id] = f"Disease {disease_id}"
    save_json(os.path.join(data_dir, "herb_id2name.json"), herb_id2name, indent=2)
    save_json(os.path.join(data_dir, "disease_id2name.json"), disease_id2name, indent=2)


# Ethnobotany

def load_mondo2mesh(path: str) -> dict[str, set[str]]:
    with open(path, encoding="utf-8") as f:
        rows = [line for line in f if not line.startswith("#")]
    mapping: dict[str, set[str]] = {}
    for r in csv.DictReader(rows, delimiter="\t"):
        subject, obj = r["subject_id"].strip(), r["object_id"].strip()
        if (r["predicate_id"] == "skos:exactMatch" and subject.startswith("MONDO:")
                and obj.lower().startswith("mesh:")):
            mapping.setdefault(subject, set()).add(
                "MESH:" + obj.split(":", 1)[1].strip().upper())
    return mapping


def prepare_ethnobotany(data_dir: str, hghda_dir: str, sssom: str) -> None:
    mondo2mesh = load_mondo2mesh(sssom)

    herbs_with_compound = {r["plant_curie"]
                           for r in read_csv(os.path.join(hghda_dir, "herb-component.csv"))
                           if r["chemical_curie"].startswith("pubchem:")}
    diseases_with_target = set()
    for r in read_csv(os.path.join(hghda_dir, "target-disease.csv")):
        disease = (r.get("DiseaseID") or "").strip()
        gene = (r.get("# GeneSymbol") or r.get("GeneSymbol") or "").strip()
        if disease.startswith("MESH:") and gene:
            diseases_with_target.add(disease.upper())

    pairs, herb_names = [], {}
    for r in read_csv(os.path.join(hghda_dir, "herb-disease.csv")):
        pairs.append((r["plant_curie"], r["disease_curie"]))
        herb_names.setdefault(r["plant_curie"], r["plant_name"])
    # a pair survives only if the herb has components and the disease has targets
    positives = sorted({(herb, disease) for herb, disease in dict.fromkeys(pairs)
                        if herb in herbs_with_compound
                        and (mondo2mesh.get(disease, set()) & diseases_with_target)})
    herbs = sorted({h for h, _ in positives})
    diseases = sorted({d for _, d in positives})

    rng = random.Random(NEG_SEED)
    positive_set, negative_set, negatives = set(positives), set(), []
    while len(negatives) < len(positives):
        pair = (rng.choice(herbs), rng.choice(diseases))
        if pair not in positive_set and pair not in negative_set:
            negative_set.add(pair)
            negatives.append(pair)

    all_disease_names = json.load(open(os.path.join(data_dir, "disease_id2name.json")))
    save_json(os.path.join(data_dir, "herb_id2name.json"), {h: herb_names[h] for h in herbs}, indent=2)
    save_json(os.path.join(data_dir, "disease_id2name.json"),
              {d: all_disease_names[d] for d in diseases}, indent=2)

    labeled = [(h, d, 1) for h, d in positives] + [(h, d, 0) for h, d in negatives]
    for idx in range(N_FOLDS):
        train_positives = [[h, d] for i, (h, d, y) in enumerate(labeled)
                           if i % N_FOLDS != idx and y == 1]
        train_all_rating: dict[str, set[str]] = {}
        for herb, disease in train_positives:
            train_all_rating.setdefault(herb, set()).add(disease)
        held_out = [(h, d, y) for i, (h, d, y) in enumerate(labeled) if i % N_FOLDS == idx]
        save_json(os.path.join(data_dir, f"fold_{idx}.json"), {
            "train_positives": train_positives,
            "test_pairs": [[h, d] for h, d, _ in held_out],
            "test_labels": [y for _, _, y in held_out],
            "train_all_rating": {h: sorted(ds) for h, ds in train_all_rating.items()},
        })

    for name, rows in (("positives.tsv", positives), ("negatives.tsv", negatives)):
        with open(os.path.join(data_dir, name), "w") as f:
            f.write("plant_curie\tdisease_curie\n")
            for herb, disease in rows:
                f.write(f"{herb}\t{disease}\n")


# entry point

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["tcm_suite", "ethnobotany"])
    ap.add_argument("--data-dir",
                    help="the benchmark directory")
    ap.add_argument("--hghda-dir", default="./data/hda/HGHDA/src/ethnobotany")
    ap.add_argument("--sssom", default="./data/hda/ethnobotany/mondo_exactmatch_mesh.sssom.tsv",
                    help="MONDO-to-MeSH mapping")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join("./data/hda", args.dataset)
    if args.dataset == "tcm_suite":
        prepare_tcm_suite(data_dir)
    else:
        prepare_ethnobotany(data_dir, args.hghda_dir, args.sssom)
    print(f"[done] {data_dir}")


if __name__ == "__main__":
    main()
