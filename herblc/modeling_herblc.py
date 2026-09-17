import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from transformers import AutoModel, PreTrainedModel
from .configuration_herblc import HerbLCConfig


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last non-padding token pooling."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]


def compute_bidirectional_infonce(
    sim: torch.Tensor,
    label_mask: torch.Tensor,
    normalize_loss: bool = False
) -> torch.Tensor:
    """Shared InfoNCE computation."""
    label_fwd = label_mask / label_mask.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    label_bwd = label_mask.T / label_mask.T.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    log_probs_fwd = F.log_softmax(sim, dim=-1)
    loss_fwd = -(label_fwd * log_probs_fwd).sum(dim=-1)
    log_probs_bwd = F.log_softmax(sim.T, dim=-1)
    loss_bwd = -(label_bwd * log_probs_bwd).sum(dim=-1)

    if normalize_loss:
        pos_count_fwd = label_mask.sum(dim=-1).clamp(min=1)
        pos_count_bwd = label_mask.T.sum(dim=-1).clamp(min=1)
        loss_fwd = loss_fwd / pos_count_fwd
        loss_bwd = loss_bwd / pos_count_bwd

    return (loss_fwd.mean() + loss_bwd.mean()) / 2.0


def _wrap_with_lora(model, lora_config, task: str, name: str):
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"{task} LoRA ({name}): {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")
    return model


# HDA (Herb-Disease Association) Model

def _masked_view_infonce(sim, label_mask, valid_row, valid_col, normalize_loss=False):
    vr = valid_row.bool()
    vc = valid_col.bool()

    def _dir(s, lm, cand_valid, anchor_valid):
        ai = anchor_valid.nonzero(as_tuple=True)[0]
        ci = cand_valid.nonzero(as_tuple=True)[0]
        if ai.numel() == 0 or ci.numel() == 0:
            return s.sum() * 0.0
        sub = s.index_select(0, ai).index_select(1, ci)
        lmsub = lm.index_select(0, ai).index_select(1, ci)
        rp = (lmsub.sum(dim=-1) > 0).nonzero(as_tuple=True)[0]
        if rp.numel() == 0:
            return s.sum() * 0.0
        sub = sub.index_select(0, rp)
        lmsub = lmsub.index_select(0, rp)
        tgt = lmsub / lmsub.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        per = -(tgt * F.log_softmax(sub, dim=-1)).sum(dim=-1)  # (n_rp,)
        if normalize_loss:
            per = per / lmsub.sum(dim=-1).clamp(min=1)
        return per.mean()

    loss_fwd = _dir(sim, label_mask, vc, vr)                     # herb→disease
    loss_bwd = _dir(sim.T, label_mask.T, vr, vc)                 # disease→herb
    return (loss_fwd + loss_bwd) / 2.0


@torch.no_grad()
def _solve_partial_ot(S, mask_a, mask_b, rho, eps, iters, a=None, b=None):
    ma = mask_a.float(); mb = mask_b.float()
    na = ma.sum(-1, keepdim=True); nb = mb.sum(-1, keepdim=True)
    valid = (na.squeeze(-1) > 0) & (nb.squeeze(-1) > 0)
    a = ((ma / na.clamp(min=1)) if a is None else (a * ma)).unsqueeze(-1)
    b = ((mb / nb.clamp(min=1)) if b is None else (b * mb)).unsqueeze(-2)
    neg = (~mask_a.bool()).unsqueeze(-1) | (~mask_b.bool()).unsqueeze(-2)
    tiny = 1e-30
    logK = (S / eps).masked_fill(neg, -float("inf"))
    logK = logK - logK.amax(dim=(-1, -2), keepdim=True).clamp(min=-1e30)
    K = torch.exp(logK).masked_fill(neg, 0.0)
    K = K * (rho / K.sum((-1, -2), keepdim=True).clamp(min=tiny))
    q1 = torch.ones_like(K); q2 = torch.ones_like(K); q3 = torch.ones_like(K)
    for _ in range(iters):
        Kprev = K; K = K * q1
        r = K.sum(-1, keepdim=True).clamp(min=tiny)
        K1 = K * torch.clamp(a / r, max=1.0); q1 = q1 * Kprev / K1.clamp(min=tiny)
        K1prev = K1; K1 = K1 * q2
        c = K1.sum(-2, keepdim=True).clamp(min=tiny)
        K2 = K1 * torch.clamp(b / c, max=1.0); q2 = q2 * K1prev / K2.clamp(min=tiny)
        K2prev = K2; K2 = K2 * q3
        K = K2 * (rho / K2.sum((-1, -2), keepdim=True).clamp(min=tiny))
        q3 = q3 * K2prev / K.clamp(min=tiny)
    return torch.where(valid[:, None, None], K, torch.zeros_like(K))


def partial_transport_score(S, cm, tm, rho=0.05, eps=0.05, iters=200, tw=None):
    """Compound-target similarities -> one molecular score per entity pair,
    s_mol = <pi*, S> / rho, where pi* solves the partial transport problem.

    S   (..., m, n) learned compound-by-target similarities
    cm  (..., m)    valid slots of the compound bag
    tm  (..., n)    valid slots of the target bag
    tw  (..., n)    optional target-side marginal weights (IDF); None = uniform
    A pair whose bag is empty on either side scores 0.
    """
    lead = S.shape[:-2]
    m, n = S.shape[-2:]
    cmb = cm.bool()
    tmb = tm.bool()
    both = cmb.any(-1) & tmb.any(-1)
    P = int(torch.tensor(lead).prod().item()) if lead else 1
    S2 = S.reshape(P, m, n)
    cm2 = cm.reshape(P, m)
    tm2 = tm.reshape(P, n)
    b = None
    if tw is not None:
        w = tw.reshape(P, n).float() * tm2.float()
        b = w / w.sum(-1, keepdim=True).clamp(min=1e-12)
    with torch.autocast(device_type=S.device.type if S.is_cuda else "cpu", enabled=False):
        G = _solve_partial_ot(S2.detach().float(), cm2, tm2, rho, eps, iters, b=b).to(S.dtype)
    out = (G * S2).sum((-1, -2)).reshape(lead) / rho
    return torch.where(both, out, torch.zeros_like(out))


class ResidualProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
        super().__init__()
        hidden_dim = hidden_dim or out_dim * 2
        self.input_proj = nn.Linear(in_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    @property
    def weight(self):
        return self.input_proj.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        return h + self.mlp(self.ln(h))


class HerbLCHDAModel(PreTrainedModel):
    """Herb-Disease Association"""
    config_class = HerbLCConfig

    def __init__(self, config: HerbLCConfig, backbone_cache_dir=None):
        super().__init__(config)
        self.config = config

        # Shared text encoder
        self.text_model = AutoModel.from_pretrained(
            config.text_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            cache_dir=backbone_cache_dir,
        )
        self.text_dim = self.text_model.config.hidden_size

        # Projection heads
        self.herb_projection = nn.Linear(self.text_dim, config.projection_dim)
        self.disease_projection = nn.Linear(self.text_dim, config.projection_dim)
        nn.init.xavier_uniform_(self.herb_projection.weight)
        nn.init.zeros_(self.herb_projection.bias)
        nn.init.xavier_uniform_(self.disease_projection.weight)
        nn.init.zeros_(self.disease_projection.bias)

        self.temperature = config.temperature

        # the two measures
        d = config.projection_dim
        self.compound_proj = ResidualProjection(int(config.hda_compound_emb_dim), d, d * 2)
        self.target_proj = ResidualProjection(int(config.hda_target_emb_dim), d, d * 2)
        self.molecular_view_weight = float(config.hda_molecular_view_weight)
        self.molecular_view_temperature = float(config.temperature)
        self.language_view_weight = float(config.hda_language_view_weight)
        self.language_view_temperature = float(config.temperature)
        self.fusion_detach_language = bool(config.hda_fusion_detach_language)
        self.mass_budget = float(config.hda_mass_budget_rho)
        self.transport_eps = float(config.hda_transport_eps)
        self.transport_iters = int(config.hda_transport_iters)
        self.gated_fusion = bool(config.hda_gated_fusion)
        fw = config.hda_fusion_weight
        self.fusion_weight = None if fw is None else float(fw)
        if (self.fusion_weight is not None) == self.gated_fusion:
            raise ValueError("set exactly one of hda_fusion_weight and hda_gated_fusion")
        if self.gated_fusion:
            self.gate_mlp = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
            self.fusion_beta = nn.Parameter(
                torch.tensor(float(np.log(np.expm1(1.0))), dtype=torch.float32))  # inv_softplus(1): gate starts at w=1
            with torch.no_grad():
                self.gate_mlp[-1].weight.zero_()
                self.gate_mlp[-1].bias.zero_()
        self.target_idf_weight = float(config.hda_target_idf_weight)
        if not 0.0 <= self.target_idf_weight <= 1.0:
            raise ValueError(f"hda_target_idf_weight must be in [0, 1], got {self.target_idf_weight}")
        n_tgt = int(config.hda_num_targets)
        self.register_buffer("target_idf",
                             torch.zeros(n_tgt) if (self.target_idf_weight > 0 and n_tgt > 0) else None,
                             persistent=True)
        self.all_rating = None

        self.text_model = _wrap_with_lora(self.text_model, LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config.lora_dropout, bias="none", task_type="FEATURE_EXTRACTION",
        ), "HDA", "text")

    def encode_herb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """language channel herb"""
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(outputs.last_hidden_state, attention_mask)
        proj = self.herb_projection
        emb = emb.to(proj.weight.dtype)
        emb = proj(emb)
        return F.normalize(emb, dim=-1) if normalize else emb

    def encode_disease(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """language channel disease"""
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(outputs.last_hidden_state, attention_mask)
        proj = self.disease_projection
        emb = emb.to(proj.weight.dtype)
        emb = proj(emb)
        return F.normalize(emb, dim=-1) if normalize else emb

    def _target_marginal_weights(self, targ_gidx, tm):
        if self.target_idf_weight <= 0:
            return None
        if self.target_idf is None or self.target_idf.numel() == 0:
            raise ValueError("hda_target_idf_weight > 0 but target_idf was never injected")
        if targ_gidx is None:
            raise ValueError("hda_target_idf_weight > 0 requires targ_gidx")
        idf = self.target_idf.to(targ_gidx.device)[targ_gidx.clamp(min=0)]   # (...,Nt)
        idf = idf * tm.to(idf.dtype)
        idf = idf / idf.sum(-1, keepdim=True).clamp(min=1e-12)
        uni = tm.to(idf.dtype)
        uni = uni / uni.sum(-1, keepdim=True).clamp(min=1e-12)
        lam = self.target_idf_weight
        return (1.0 - lam) * uni + lam * idf

    def fuse_hda_channels(self, text_sim, mol_sim):
        if self.fusion_weight is not None:
            return text_sim + self.fusion_weight * mol_sim
        feats = torch.stack([text_sim.float(), mol_sim.float()], dim=-1)
        delta = self.gate_mlp(feats).squeeze(-1)
        return text_sim + F.softplus(self.fusion_beta + delta) * mol_sim

    def _dual_channel_score(self, herb_text, comp_embs, comp_mask, disease_text, targ_embs, targ_mask,
                            targ_gidx=None, return_components=False, paired=False):
        comp_pts = self.compound_proj(comp_embs.to(device=herb_text.device, dtype=self.compound_proj.weight.dtype))
        targ_pts = self.target_proj(targ_embs.to(device=disease_text.device, dtype=self.target_proj.weight.dtype))
        text_sim = (herb_text * disease_text).sum(-1) if paired else herb_text @ disease_text.t()
        text_fuse = text_sim.detach() if self.fusion_detach_language else text_sim
        cm = comp_mask.to(herb_text.device).bool()
        tm = targ_mask.to(disease_text.device).bool()
        comp_n = F.normalize(comp_pts, dim=-1).to(herb_text.dtype)
        targ_n = F.normalize(targ_pts, dim=-1).to(disease_text.dtype)
        with torch.autocast(device_type=comp_n.device.type, enabled=False):
            tw = self._target_marginal_weights(targ_gidx, tm)
            if paired:
                S = torch.einsum('bid,bjd->bij', comp_n.float(), targ_n.float())   # (B,Nh,Nt)
                cm_b, tm_b, tw_b = cm, tm, tw
            else:
                S = torch.einsum('hid,tjd->htij', comp_n.float(), targ_n.float())  # (Bh,Bt,Nh,Nt)
                cm_b = cm[:, None, :].expand(-1, tm.shape[0], -1)                  # (Bh,Bt,Nh)
                tm_b = tm[None, :, :].expand(cm.shape[0], -1, -1)                  # (Bh,Bt,Nt)
                tw_b = None if tw is None else tw[None, :, :].expand(cm.shape[0], -1, -1)
            mol_sim_fp32 = partial_transport_score(
                S, cm_b, tm_b, rho=self.mass_budget, eps=self.transport_eps,
                iters=self.transport_iters, tw=tw_b).float()
        mol_sim = mol_sim_fp32.to(text_sim.dtype)
        fused = self.fuse_hda_channels(text_fuse, mol_sim)
        return (fused, mol_sim_fp32) if return_components else fused

    def forward(
        self,
        herb_input_ids: torch.Tensor,
        herb_attention_mask: torch.Tensor,
        disease_input_ids: torch.Tensor,
        disease_attention_mask: torch.Tensor,
        herb_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        herb_compound_embs: torch.Tensor = None,
        herb_compound_mask: torch.Tensor = None,
        disease_target_embs: torch.Tensor = None,
        disease_target_mask: torch.Tensor = None,
        **kwargs,
    ):
        herb_text = self.encode_herb(herb_input_ids, herb_attention_mask, normalize=True)
        disease_text = self.encode_disease(disease_input_ids, disease_attention_mask, normalize=True)
        want_mol_view = self.molecular_view_weight > 0
        mol_sim_mat = mv_comp_mask = mv_targ_mask = None
        if herb_compound_embs is None or disease_target_embs is None:
            raise ValueError(
                "herb_compound_embs/disease_target_embs missing from forward inputs"
            )
        _rc = self._dual_channel_score(
            herb_text, herb_compound_embs, herb_compound_mask,
            disease_text, disease_target_embs, disease_target_mask,
            targ_gidx=kwargs.get("target_global_index"),
            return_components=want_mol_view,
        )
        if want_mol_view:
            _raw, mol_sim_mat = _rc
            sim = _raw / self.temperature
            mv_comp_mask = herb_compound_mask; mv_targ_mask = disease_target_mask
        else:
            sim = _rc / self.temperature
        B = sim.size(0)

        label_mask = torch.eye(B, dtype=torch.float, device=sim.device)
        if self.config.true_label > 0 and self.all_rating is not None:
            herb_ids_list = [str(v) for v in herb_ids]
            disease_ids_list = [str(v) for v in disease_ids]
            for i in range(B):
                herb_positives = self.all_rating.get(herb_ids_list[i], set())
                for j in range(B):
                    if i != j and disease_ids_list[j] in herb_positives:
                        label_mask[i, j] = 1.0

        loss = compute_bidirectional_infonce(
            sim, label_mask,
            normalize_loss=True,
        )

        if want_mol_view and mol_sim_mat is not None:
            vh = mv_comp_mask.to(sim.device).bool().any(-1)
            vd = mv_targ_mask.to(sim.device).bool().any(-1)
            if bool(vh.any()) and bool(vd.any()):
                mol_logits = mol_sim_mat.float() / self.molecular_view_temperature
                l_mol = _masked_view_infonce(mol_logits, label_mask, vh, vd, normalize_loss=True)
                loss = loss + self.molecular_view_weight * l_mol

        if self.language_view_weight > 0:
            text_logits = (herb_text.float() @ disease_text.float().T) / self.language_view_temperature
            l_text = compute_bidirectional_infonce(
                text_logits, label_mask,
                normalize_loss=True)
            loss = loss + self.language_view_weight * l_text

        return {"loss": loss}



# HTI (Herb-Target Interaction) Model

class HerbLCHTIModel(PreTrainedModel):
    """Herb-Target Interaction model."""
    config_class = HerbLCConfig

    def __init__(self, config: HerbLCConfig, backbone_cache_dir=None):
        super().__init__(config)
        self.config = config

        # Text encoder
        self.text_model = AutoModel.from_pretrained(
            config.text_model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            cache_dir=backbone_cache_dir,
        )
        self.text_dim = self.text_model.config.hidden_size

        # Protein encoder
        self.protein_model = AutoModel.from_pretrained(config.protein_model, cache_dir=backbone_cache_dir)
        self.protein_dim = self.protein_model.config.hidden_size

        # Projection heads
        self.text_projection = nn.Linear(self.text_dim, config.projection_dim)
        self.protein_projection = nn.Linear(self.protein_dim, config.projection_dim)

        # Temperature
        self.temperature = config.temperature

        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.zeros_(self.text_projection.bias)
        nn.init.xavier_uniform_(self.protein_projection.weight)
        nn.init.zeros_(self.protein_projection.bias)

        self.all_rating = None
        self.heldout_rating = None

        self.text_model = _wrap_with_lora(self.text_model, LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config.lora_dropout, bias="none", task_type="FEATURE_EXTRACTION",
        ), "HTI", "text")
        self.protein_model = _wrap_with_lora(self.protein_model, LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=config.lora_dropout, bias="none", task_type="FEATURE_EXTRACTION",
        ), "HTI", "protein")

    def encode_text(self, input_ids, attention_mask, normalize=True):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = last_token_pool(outputs.last_hidden_state, attention_mask)
        emb = emb.to(self.text_projection.weight.dtype)
        emb = self.text_projection(emb)
        return F.normalize(emb, dim=-1) if normalize else emb

    def encode_protein(self, input_ids, attention_mask, normalize=True):
        outputs = self.protein_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = emb.to(self.protein_projection.weight.dtype)
        emb = self.protein_projection(emb)
        return F.normalize(emb, dim=-1) if normalize else emb

    def forward(
        self,
        herb_input_ids,
        herb_attention_mask,
        gene_input_ids,
        gene_attention_mask,
        herb_ids,
        gene_ids,
        **kwargs,
    ):
        herb_emb = self.encode_text(herb_input_ids, herb_attention_mask, normalize=True)
        gene_emb = self.encode_protein(gene_input_ids, gene_attention_mask, normalize=True)

        sim = herb_emb @ gene_emb.T / self.temperature
        B = sim.size(0)

        label_mask = torch.eye(B, dtype=torch.float, device=sim.device)
        if self.config.true_label > 0 and self.all_rating is not None:
            herb_ids_list = herb_ids.tolist()
            gene_ids_list = gene_ids.tolist()
            for i in range(B):
                herb_positives = self.all_rating.get(herb_ids_list[i], set())
                for j in range(B):
                    if i != j and gene_ids_list[j] in herb_positives:
                        label_mask[i, j] = 1.0

        if self.heldout_rating:
            herb_ids_list = herb_ids.tolist()
            gene_ids_list = gene_ids.tolist()
            drop = torch.zeros_like(label_mask, dtype=torch.bool)
            for i in range(B):
                heldout_genes = self.heldout_rating.get(herb_ids_list[i], ())
                for j in range(B):
                    if i != j and gene_ids_list[j] in heldout_genes:
                        drop[i, j] = True
            if drop.any():
                sim = sim.masked_fill(drop, torch.finfo(sim.dtype).min)

        loss = compute_bidirectional_infonce(
            sim, label_mask,
            normalize_loss=False,
        )

        return {"loss": loss}
