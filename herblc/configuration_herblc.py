from typing import Optional
from transformers import PretrainedConfig


class HerbLCConfig(PretrainedConfig):
    model_type = "herblc"

    def __init__(
        self,
        # encoders
        protein_model: str = "facebook/esm2_t33_650M_UR50D",
        text_model: str = "Qwen/Qwen3-Embedding-4B",
        projection_dim: int = 768,
        # contrastive loss
        temperature: float = 0.07,
        true_label: int = 1,
        # LoRA adapters
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        # the measures
        hda_compound_emb_dim: int = 512,
        hda_target_emb_dim: int = 1280,
        hda_max_components: int = 128,
        hda_num_targets: int = 0,
        # partial optimal transport
        hda_mass_budget_rho: float = 0.05,
        hda_transport_eps: float = 0.05,
        hda_transport_iters: int = 200,
        hda_target_idf_weight: float = 0.0,
        # combining the language and molecular scores
        hda_fusion_weight: Optional[float] = None,
        hda_gated_fusion: bool = False,
        hda_fusion_detach_language: bool = False,
        # auxiliary losses
        hda_molecular_view_weight: float = 0.0,
        hda_language_view_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # encoders
        self.protein_model = protein_model
        self.text_model = text_model
        self.projection_dim = projection_dim

        # contrastive loss
        self.temperature = temperature
        self.true_label = true_label

        # LoRA adapters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # the measures
        self.hda_compound_emb_dim = hda_compound_emb_dim
        self.hda_target_emb_dim = hda_target_emb_dim
        self.hda_max_components = hda_max_components
        self.hda_num_targets = hda_num_targets          # filled in at run time from target_emb

        # partial optimal transport
        self.hda_mass_budget_rho = hda_mass_budget_rho
        self.hda_transport_eps = hda_transport_eps
        self.hda_transport_iters = hda_transport_iters
        self.hda_target_idf_weight = hda_target_idf_weight

        # combining the language and molecular scores
        self.hda_fusion_weight = hda_fusion_weight
        self.hda_gated_fusion = hda_gated_fusion
        self.hda_fusion_detach_language = hda_fusion_detach_language

        # auxiliary losses
        self.hda_molecular_view_weight = hda_molecular_view_weight
        self.hda_language_view_weight = hda_language_view_weight


if __name__ == "__main__":
    herblc_config = HerbLCConfig()
    herblc_config.save_pretrained("./")
