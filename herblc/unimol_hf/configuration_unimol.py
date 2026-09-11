from transformers import PretrainedConfig


class UnimolConfig(PretrainedConfig):
    """UniMol encoder hyperparameters; the defaults are those the released pretrained weights were trained under."""

    model_type = "unimol"

    def __init__(
        self,
        vocab_size: int = 31,
        mask_token_id: int = 30,
        pad_token_id: int = 0,
        encoder_layers: int = 15,
        encoder_embed_dim: int = 512,
        encoder_ffn_embed_dim: int = 2048,
        encoder_attention_heads: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        pooler_dropout: float = 0.0,
        max_seq_len: int = 512,
        activation_fn: str = "gelu",
        pooler_activation_fn: str = "tanh",
        post_ln: bool = False,
        remove_hs: bool = True,
        max_atoms: int = 256,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        kernel: str = 'gaussian',
        delta_pair_repr_norm_loss: float = -1.0,
        pretrain_path: str = None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, mask_token_id=mask_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, unk_token_id=unk_token_id, **kwargs)
        if activation_fn not in ["relu", "gelu", "tanh", "linear"]:
            raise ValueError(f"`activation_fn` must be 'relu', 'gelu', 'tanh' or 'linear', got {activation_fn}.")
        if pooler_activation_fn not in ["relu", "gelu", "tanh", "linear"]:
            raise ValueError(f"`pooler_activation_fn` must be 'relu', 'gelu', 'tanh' or 'linear', got {pooler_activation_fn}.")
        self.vocab_size = vocab_size
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.emb_dropout = emb_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.pooler_dropout = pooler_dropout
        self.max_seq_len = max_seq_len
        self.activation_fn = activation_fn
        self.pooler_activation_fn = pooler_activation_fn
        self.post_ln = post_ln
        self.remove_hs = remove_hs
        self.max_atoms = max_atoms
        self.kernel = kernel
        self.delta_pair_repr_norm_loss = delta_pair_repr_norm_loss
        self.pretrain_path = pretrain_path


if __name__ == "__main__":
    unimol_config = UnimolConfig()
    unimol_config.save_pretrained("./")