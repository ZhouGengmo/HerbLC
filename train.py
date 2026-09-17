#!/usr/bin/env python
import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
from evaluate import evaluate_hda_fold, evaluate_hti, print_hda_results, print_hti_results
from herblc.configuration_herblc import HerbLCConfig
from herblc.dataset import (HDADataCollator, HTIDataCollator, load_bag,
                            load_hda_dataset, load_hti_dataset)
from herblc.modeling_herblc import HerbLCHDAModel, HerbLCHTIModel


def parse_args():
    parser = argparse.ArgumentParser(description="HerbLC training script")

    parser.add_argument("--task", required=True, choices=["hda", "hti"])
    parser.add_argument("--output-dir", required=True,
                        help="directory for checkpoints, logs and the results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-cache-dir",
                        help="where the pretrained encoders are cached. Unset uses the Hugging Face default")

    # data
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--hda-cv-fold", type=int, default=0, help="cross-validation fold, 0-4")

    # encoders
    parser.add_argument("--text-model", help="herb and disease text encoder")
    parser.add_argument("--protein-model", help="target encoder")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--projection-dim", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)

    # contrastive objective
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--true-label", type=int,
                        help="multi-positive contrastive objective (0 = single positive)")

    # HDA measures and partial optimal transport
    parser.add_argument("--hda-compound-emb-dim", type=int, help="compound representation size")
    parser.add_argument("--hda-target-emb-dim", type=int, help="target representation size")
    parser.add_argument("--hda-max-components", type=int,
                        help="cap on how many components one entity contributes")
    parser.add_argument("--hda-mass-budget-rho", type=float, help="transported mass rho")
    parser.add_argument("--hda-transport-eps", type=float, help="entropic regularisation")
    parser.add_argument("--hda-transport-iters", type=int, help="Dykstra iterations")
    parser.add_argument("--hda-target-idf-weight", type=float,
                        help="IDF weighting on the disease-target marginal (0 = uniform)")

    # HDA fusion
    parser.add_argument("--hda-fusion-weight", type=float,
                        help="fixed w, mutually exclusive with --hda-gated-fusion")
    parser.add_argument("--hda-gated-fusion", action="store_true", default=None,
                        help="predict w from the two channel scores")
    parser.add_argument("--hda-fusion-detach-language", action="store_true", default=None,
                        help="stop-gradient on the language score inside the fused loss")
    parser.add_argument("--hda-molecular-view-weight", type=float,
                        help="weight of the auxiliary loss on the molecular channel")
    parser.add_argument("--hda-language-view-weight", type=float,
                        help="weight of the auxiliary loss on the language channel")

    # optimisation
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-end", type=float,
                        help="final LR of the polynomial scheduler")
    parser.add_argument("--bf16", action="store_true")

    # schedule
    parser.add_argument("--train-steps", type=int, default=50, help="HDA")
    parser.add_argument("--warmup-steps", type=int, default=5, help="HDA")
    parser.add_argument("--save-steps", type=int, default=10, help="HDA")
    parser.add_argument("--num-epochs", type=int, default=5, help="HTI")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="HTI")
    parser.add_argument("--save-total-limit", type=int, help="max checkpoints to keep")
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int)

    return parser.parse_args()




def target_idf(all_rating, disease_to_targets, n_targets):
    """Inverse document frequency of each target over this fold's training diseases."""
    diseases = {str(d) for targets in all_rating.values() for d in targets}
    df = np.zeros(n_targets, dtype=np.float64)
    for disease in diseases:
        for t in set(disease_to_targets.get(disease, [])):
            df[t] += 1.0
    idf = np.maximum(np.log((len(diseases) + 1.0) / (df + 1.0)), 1e-6)
    print(f"  target IDF over {len(diseases)} training diseases and {n_targets} targets: "
          f"range [{idf.min():.3f}, {idf.max():.3f}]")
    return torch.from_numpy(idf).float()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config = HerbLCConfig()
    for name, value in vars(args).items():
        if value is not None and hasattr(config, name):
            setattr(config, name, value)
    print(config)

    text_tokenizer = AutoTokenizer.from_pretrained(config.text_model, cache_dir=args.hf_cache_dir)
    text_tokenizer.padding_side = "left"
    protein_tokenizer = None
    eval_dataset = None
    accelerator_config = None

    if args.task == "hda":
        if config.hda_fusion_detach_language and config.hda_language_view_weight <= 0:
            raise ValueError("--hda-fusion-detach-language requires --hda-language-view-weight > 0")
        if not 0.0 < config.hda_mass_budget_rho <= 1.0 or config.hda_transport_eps <= 0 \
                or config.hda_transport_iters < 1:
            raise ValueError(f"invalid partial-OT settings: rho={config.hda_mass_budget_rho} "
                             f"eps={config.hda_transport_eps} iters={config.hda_transport_iters}")
        train_dataset, metadata = load_hda_dataset(args.data_dir, cv_fold=args.hda_cv_fold)
        compound_embs, herb_to_compounds = load_bag(
            args.data_dir, "compound", "herb", config.hda_compound_emb_dim)
        target_embs, disease_to_targets = load_bag(
            args.data_dir, "target", "disease", config.hda_target_emb_dim)
        config.hda_num_targets = int(target_embs.shape[0])

        data_collator = HDADataCollator(
            text_tokenizer=text_tokenizer,
            max_text_length=args.max_text_length,
            compound_emb_lookup=compound_embs,
            herb_to_compound_indices=herb_to_compounds,
            target_emb_lookup=target_embs,
            disease_to_target_indices=disease_to_targets,
            max_components=config.hda_max_components,
            emit_global_idx=config.hda_target_idf_weight > 0,
        )
        metadata.update({
            "compound_emb_lookup": compound_embs,
            "herb_to_compound_indices": herb_to_compounds,
            "target_emb_lookup": target_embs,
            "disease_to_target_indices": disease_to_targets,
            "max_components": config.hda_max_components,
        })

        model = HerbLCHDAModel(config, backbone_cache_dir=args.hf_cache_dir)
        model.all_rating = metadata["all_rating"]
        if config.hda_target_idf_weight > 0:
            model.target_idf = target_idf(
                metadata["all_rating"], disease_to_targets, config.hda_num_targets)
    else:
        print("\nHTI")
        train_dataset, eval_dataset, _, metadata = load_hti_dataset(args.data_dir)

        protein_tokenizer = AutoTokenizer.from_pretrained(config.protein_model,
                                                         cache_dir=args.hf_cache_dir)
        data_collator = HTIDataCollator(
            text_tokenizer=text_tokenizer,
            protein_tokenizer=protein_tokenizer,
            max_text_length=args.max_text_length,
            max_protein_length=args.max_protein_length,
        )
        accelerator_config = {
            "split_batches": True,
            "dispatch_batches": False,
            "even_batches": True,
            "use_seedable_sampler": True,
        }

        model = HerbLCHTIModel(config, backbone_cache_dir=args.hf_cache_dir)
        model.all_rating = metadata["all_rating"]
        model.heldout_rating = metadata["heldout_rating"]

    print(model)

    if args.max_grad_norm <= 0:
        raise ValueError(f"--max-grad-norm must be > 0, got {args.max_grad_norm}")
    lr_scheduler_kwargs = {}
    if args.lr_end is not None:
        if not 0.0 <= args.lr_end < args.lr:
            raise ValueError(f"--lr-end must satisfy 0 <= lr_end < learning_rate, got "
                             f"{args.lr_end} vs {args.lr}")
        lr_scheduler_kwargs = {"lr_end": args.lr_end, "power": 1.0}

    common = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.wd,
        lr_scheduler_type="polynomial",
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers else None,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16,
        # the model returns its own loss
        label_names=[],
        load_best_model_at_end=args.task != "hda",
        greater_is_better=False,
        ddp_find_unused_parameters=True,
        accelerator_config=accelerator_config,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    if args.task == "hda":
        training_args = TrainingArguments(
            max_steps=args.train_steps,
            warmup_steps=args.warmup_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            eval_strategy="no",
            dataloader_pin_memory=True,
            **common,
        )
    else:
        training_args = TrainingArguments(
            num_train_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            save_strategy="epoch",
            eval_strategy="epoch",
            dataloader_pin_memory=False,
            **common,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()

    if not trainer.is_world_process_zero():
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    if args.task == "hda":
        print("\n=== HDA evaluation ===")
        results = evaluate_hda_fold(
            model=model,
            metadata=metadata,
            device=device,
            text_tokenizer=text_tokenizer,
            max_text_length=args.max_text_length,
        )
        print_hda_results(results, fold=args.hda_cv_fold)
        results_path = os.path.join(args.output_dir, f"hda_results_fold{args.hda_cv_fold}.json")
    else:
        print("\n=== HTI retrieval evaluation ===")
        results = evaluate_hti(
            model=model,
            metadata=metadata,
            device=device,
            text_tokenizer=text_tokenizer,
            protein_tokenizer=protein_tokenizer,
            max_text_length=args.max_text_length,
            max_protein_length=args.max_protein_length,
        )
        print_hti_results(results)
        results_path = os.path.join(args.output_dir, "hti_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results saved to: {results_path}")


if __name__ == "__main__":
    main()
