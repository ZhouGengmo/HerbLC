# HerbLC

Code for *HerbLC: understanding multi-component multi-target herbal action through language and
chemistry*.

## Install

```bash
pip install -r requirements.txt
```

## Data

Download the raw release of each benchmark:

| Benchmark | Source |
|---|---|
| TCM-suite (herb–disease) | https://zenodo.org/records/10432947 |
| Ethnobotany (herb–disease) | https://github.com/bioxjz/HGHDA |
| HTI (herb–target) | https://github.com/2020MEAI/HTINet2 |

### Pretrained encoder weights


Uni-Mol `mol_pre_no_h_220816.pt`,  https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt

ESM-2 `facebook/esm2_t33_650M_UR50D` and Qwen3-Embedding are downloaded from Hugging Face on first use

### TCM-suite

Unpack the Zenodo release into `./data/hda/tcm_suite`, then:

```bash
python scripts/prepare_hda_data.py --dataset tcm_suite
python scripts/precompute_emb.py --dataset tcm_suite --modality compound
python scripts/precompute_emb.py --dataset tcm_suite --modality target
```

### Ethnobotany

Clone the HGHDA repository to `./data/hda/HGHDA`, then:

```bash
curl -L -o ./data/hda/ethnobotany/mondo_exactmatch_mesh.sssom.tsv \
  https://raw.githubusercontent.com/monarch-initiative/mondo/v2023-08-02/src/ontology/mappings/mondo_exactmatch_mesh.sssom.tsv
python scripts/prepare_hda_data.py --dataset ethnobotany
python scripts/precompute_emb.py --dataset ethnobotany --modality compound
python scripts/precompute_emb.py --dataset ethnobotany --modality target
```

### HTI

Collect 5 files from the HTINet2 release into `./data/hti`.

## Training

HDA, TCM-suite, fold 0:

```bash
torchrun --standalone --nproc_per_node=4 train.py --task hda --seed 42 \
  --data-dir ./data/hda/tcm_suite --hda-cv-fold 0 --output-dir ./runs/tcm_suite/fold_0 \
  --hf-cache-dir ./.cache --text-model Qwen/Qwen3-Embedding-4B \
  --batch-size 16 --gradient-accumulation-steps 4 --lr 2e-5 --wd 1e-5 \
  --train-steps 80000 --warmup-steps 5000 --save-steps 10000 --save-total-limit 1 \
  --temperature 0.07 --bf16 --hda-transport-iters 200 --hda-fusion-weight 2.0
```

HDA, Ethnobotany, fold 0:

```bash
torchrun --standalone --nproc_per_node=4 train.py --task hda --seed 42 \
  --data-dir ./data/hda/ethnobotany --hda-cv-fold 0 --output-dir ./runs/ethnobotany/fold_0 \
  --hf-cache-dir ./.cache --text-model Qwen/Qwen3-Embedding-4B \
  --batch-size 64 --lr 1e-4 --wd 1e-5 --max-grad-norm 5.0 --lr-end 8e-5 \
  --train-steps 30000 --warmup-steps 5000 --save-steps 10000 --save-total-limit 1 \
  --temperature 0.03 --bf16 --hda-transport-iters 50 --hda-target-idf-weight 1.0 \
  --hda-gated-fusion --hda-fusion-detach-language \
  --hda-molecular-view-weight 1.0 --hda-language-view-weight 1.0
```

For each HDA benchmark, repeat with `--hda-cv-fold 1` through `4`, changing
`--output-dir` to the corresponding `fold_1` through `fold_4` directory.

HTI:

```bash
torchrun --standalone --nproc_per_node=4 train.py --task hti --seed 42 \
  --data-dir ./data/hti --output-dir ./runs/hti --hf-cache-dir ./.cache \
  --text-model Qwen/Qwen3-Embedding-4B --batch-size 32 --lr 1e-4 --wd 1e-5 \
  --num-epochs 200 --warmup-ratio 0.06 --save-total-limit 1 --temperature 0.07 --bf16
```

## Evaluation

HDA

```bash
python evaluate.py --task hda --checkpoint ./runs/tcm_suite/fold_0 \
  --data-dir ./data/hda/tcm_suite --hda-cv-fold 0 --hf-cache-dir ./.cache
```

HTI

```bash
python evaluate.py --task hti --checkpoint ./runs/hti --data-dir ./data/hti --hf-cache-dir ./.cache
```

HDA is reported as the mean over the 5 folds:

```bash
python evaluate.py --aggregate --results-dir ./runs/tcm_suite
```
