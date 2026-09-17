from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import logging
from .configuration_unimol import UnimolConfig
from typing import Optional, Dict, List, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint
import os
logger = logging.get_logger(__name__)

class UnimolModel(PreTrainedModel):
    config_class = UnimolConfig

    def __init__(self, config: UnimolConfig):
        super().__init__(config)
        self.config = config
        self.remove_hs = config.remove_hs
        self.vocab_size = config.vocab_size
        self.mask_token_id = config.mask_token_id
        self.pad_token_id = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            self.vocab_size, config.encoder_embed_dim, self.pad_token_id
        )
        self.encoder = TransformerEncoderWithPair(
            encoder_layers=config.encoder_layers,
            embed_dim=config.encoder_embed_dim,
            ffn_embed_dim=config.encoder_ffn_embed_dim,
            attention_heads=config.encoder_attention_heads,
            emb_dropout=config.emb_dropout,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            activation_dropout=config.activation_dropout,
            max_seq_len=config.max_seq_len,
            activation_fn=config.activation_fn,
            post_ln=config.post_ln,
            no_final_head_layer_norm=config.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = self.vocab_size * self.vocab_size
        self.gbf_proj = NonLinearHead(
            K, config.encoder_attention_heads, config.activation_fn
        )
        if config.kernel == 'gaussian':
            self.gbf = GaussianLayer(K, n_edge_type)

        self.post_init()
        if config.pretrain_path:
            self.load_pretrained_weights(config.pretrain_path)

    def load_pretrained_weights(self, pretrain_path):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if pretrain_path is not None:
            logger.info("Loading pretrained weights from {}".format(pretrain_path))
            state_dict = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
            errors = self.load_state_dict(state_dict['model'], strict=False)
            if errors.missing_keys:
                logger.warning(
                    "Warning in loading model state, missing_keys "
                    + str(errors.missing_keys)
                )
            if errors.unexpected_keys:
                logger.warning(
                    "Warning in loading model state, unexpected_keys "
                    + str(errors.unexpected_keys)
                )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing = False

    def forward(
        self,
        src_tokens: Optional[torch.LongTensor] = None,
        src_distance: Optional[torch.Tensor] = None,
        src_coord: Optional[torch.Tensor] = None,
        src_edge_type: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        return_repr: Optional[bool] = False,
        return_atomic_reprs: Optional[bool] = False,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        padding_mask = None
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).bool()

        if src_tokens is None:
            raise ValueError("src_tokens (input_ids) cannot be None")

        x = self.embed_tokens(src_tokens)

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

        if src_distance is None or src_edge_type is None:
            raise ValueError("src_distance and src_edge_type are required for UniMol.")

        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_rep[:, 0, :]  # CLS token repr
        all_repr = encoder_rep[:, :, :]  # all token repr

        if return_atomic_reprs:
            atomic_outputs = self._extract_atomic_reprs(src_tokens, all_repr, src_coord, padding_mask)
            output_dict = {
                'cls_repr': cls_repr,
                'atomic_reprs': atomic_outputs["atomic_reprs"],
                'atomic_coords': atomic_outputs["atomic_coords"],
            }
            if not return_dict:
                logger.warning("Returning tuple format for representations is not fully implemented.")
                return (output_dict['cls_repr'], output_dict['atomic_reprs'], output_dict['atomic_coords'])
            else:
                return output_dict

        if return_repr:
            if not return_dict:
                return (cls_repr,)
            else:
                return {'cls_repr': cls_repr}

        # default return
        if not return_dict:
            return (cls_repr,)
        else:
            return BaseModelOutputWithPooling(
                last_hidden_state=all_repr,
                pooler_output=cls_repr
            )

    def _extract_atomic_reprs(self, src_tokens, all_repr, src_coord, padding_mask):
        """Helper to extract atomic representations, coords, and token IDs."""
        batch_atomic_reprs = []
        batch_atomic_coords = []
        batch_atomic_token_ids = [] # save atomic ID

        if padding_mask is None:
            padding_mask = torch.zeros_like(src_tokens, dtype=torch.bool, device=src_tokens.device)

        for i in range(src_tokens.size(0)):
            is_atom = (src_tokens[i] != self.config.bos_token_id) & \
                      (src_tokens[i] != self.config.eos_token_id) & \
                      (src_tokens[i] != self.config.pad_token_id) & \
                      (~padding_mask[i])
            
            atom_indices = torch.where(is_atom)[0]
            
            atomic_reprs = all_repr[i, atom_indices, :]
            atomic_coords = src_coord[i, atom_indices, :] if src_coord is not None else None
            atomic_token_ids = src_tokens[i, atom_indices]
            batch_atomic_reprs.append(atomic_reprs)
            if atomic_coords is not None:
                batch_atomic_coords.append(atomic_coords)
            batch_atomic_token_ids.append(atomic_token_ids)

        return {
            "atomic_reprs": batch_atomic_reprs,
            "atomic_coords": batch_atomic_coords,
            "atomic_token_ids": batch_atomic_token_ids
        }



class NonLinearHead(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network 
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """
    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
        hidden=None,
    ):
        """
        Initializes the NonLinearHead module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        :param hidden: Dimension of the hidden layer; defaults to the same as input_dim if not provided.
        """
        super().__init__()
        hidden = input_dim if not hidden else hidden
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the NonLinearHead.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x

@torch.jit.script
def gaussian(x, mean, std):
    """
    Gaussian function implemented for PyTorch tensors.

    :param x: The input tensor.
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """
    def __init__(self, K=128, edge_types=1024):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type).type_as(x)
        bias = self.bias(edge_type).type_as(x)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    """softmax dropout, and mask, bias are optional.
    Args:
        input (torch.Tensor): input tensor
        dropout_prob (float): dropout probability
        is_training (bool, optional): is in training or not. Defaults to True.
        mask (torch.Tensor, optional): the mask tensor, use as input + mask . Defaults to None.
        bias (torch.Tensor, optional): the bias tensor, use as input + bias . Defaults to None.

    Returns:
        torch.Tensor: the result after softmax
    """
    input = input.contiguous()
    if not inplace:
        # copy a input for non-inplace case
        input = input.clone()
    if mask is not None:
        input += mask
    if bias is not None:
        input += bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)

def get_activation_fn(activation):
    """ Returns the activation function corresponding to `activation` """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

class SelfMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.1,
        bias=True,
        scaling_factor=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Tensor:

        bsz, tgt_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        q = (
            q.view(bsz, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * self.num_heads, -1, self.head_dim)
            * self.scaling
        )
        if k is not None:
            k = (
                k.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )
        if v is not None:
            v = (
                v.view(bsz, -1, self.num_heads, self.head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz * self.num_heads, -1, self.head_dim)
            )

        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if not return_attn:
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, bias=attn_bias,
            )
        else:
            attn_weights += attn_bias
            attn = softmax_dropout(
                attn_weights, self.dropout, self.training, inplace=False,
            )

        o = torch.bmm(attn, v)
        assert list(o.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        o = (
            o.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, embed_dim)
        )
        o = self.out_proj(o)
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn

class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        post_ln = False,
    ) -> None:
        super().__init__()

        # Initialize parameters
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout

        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)

        self.self_attn = SelfMultiheadAttention(
            self.embed_dim,
            attention_heads,
            dropout=attention_dropout,
        )
        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.post_ln = post_ln


    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool=False,
    ) -> torch.Tensor:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        if not self.post_ln:
            x = self.self_attn_layer_norm(x)
        # new added
        x = self.self_attn(
            query=x,
            key_padding_mask=padding_mask,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.self_attn_layer_norm(x)

        residual = x
        if not self.post_ln:
            x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if self.post_ln:
            x = self.final_layer_norm(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

class TransformerEncoderWithPair(nn.Module):
    """
    A custom Transformer Encoder module that extends PyTorch's nn.Module. This encoder is designed for tasks that require understanding pair relationships in sequences. It includes standard transformer encoder layers along with additional normalization and dropout features.

    Attributes:
        - emb_dropout: Dropout rate applied to the embedding layer.
        - max_seq_len: Maximum length of the input sequences.
        - embed_dim: Dimensionality of the embeddings.
        - attention_heads: Number of attention heads in the transformer layers.
        - emb_layer_norm: Layer normalization applied to the embedding layer.
        - final_layer_norm: Optional final layer normalization.
        - final_head_layer_norm: Optional layer normalization for the attention heads.
        - layers: A list of transformer encoder layers.

    Methods:
        forward: Performs the forward pass of the module.
    """
    
    def __init__(
        self,
        encoder_layers: int = 6,
        embed_dim: int = 768,
        ffn_embed_dim: int = 3072,
        attention_heads: int = 8,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 256,
        activation_fn: str = "gelu",
        post_ln: bool = False,
        no_final_head_layer_norm: bool = False,
    ) -> None:
        """
        Initializes and configures the layers and other components of the transformer encoder.
        
        :param encoder_layers: (int) Number of encoder layers in the transformer.
        :param embed_dim: (int) Dimensionality of the input embeddings.
        :param ffn_embed_dim: (int) Dimensionality of the feedforward network model.
        :param attention_heads: (int) Number of attention heads in each encoder layer.
        :param emb_dropout: (float) Dropout rate for the embedding layer.
        :param dropout: (float) Dropout rate for the encoder layers.
        :param attention_dropout: (float) Dropout rate for the attention mechanisms.
        :param activation_dropout: (float) Dropout rate for activations.
        :param max_seq_len: (int) Maximum sequence length the model can handle.
        :param activation_fn: (str) The activation function to use (e.g., "gelu").
        :param post_ln: (bool) If True, applies layer normalization after the feedforward network.
        :param no_final_head_layer_norm: (bool) If True, does not apply layer normalization to the final attention head.

        """
        super().__init__()
        self.emb_dropout = emb_dropout
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.emb_layer_norm = nn.LayerNorm(self.embed_dim)
        if not post_ln:
            self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        else:
            self.final_layer_norm = None

        if not no_final_head_layer_norm:
            self.final_head_layer_norm = nn.LayerNorm(attention_heads)
        else:
            self.final_head_layer_norm = None

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=ffn_embed_dim,
                    attention_heads=attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    post_ln=post_ln,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.gradient_checkpointing = False  # set True to trade compute for activation memory (full FT)

    def forward(
        self,
        emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Conducts the forward pass of the transformer encoder.

        :param emb: (torch.Tensor) The input tensor of embeddings.
        :param attn_mask: (Optional[torch.Tensor]) Attention mask to specify positions to attend to.
        :param padding_mask: (Optional[torch.Tensor]) Mask to indicate padded elements in the input.

        :return: (torch.Tensor) The output tensor after passing through the transformer encoder layers.
                 It also returns tensors related to pair representation and normalization losses.
        """
        bsz = emb.size(0)
        seq_len = emb.size(1)
        x = self.emb_layer_norm(emb)
        x = F.dropout(x, p=self.emb_dropout, training=self.training)
        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        input_attn_mask = attn_mask
        input_padding_mask = padding_mask

        def fill_attn_mask(attn_mask, padding_mask, fill_val=float("-inf")):
            if attn_mask is not None and padding_mask is not None:
                # merge key_padding_mask and attn_mask
                attn_mask = attn_mask.view(x.size(0), -1, seq_len, seq_len)
                attn_mask.masked_fill_(
                    padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    fill_val,
                )
                attn_mask = attn_mask.view(-1, seq_len, seq_len)
                padding_mask = None
            return attn_mask, padding_mask

        assert attn_mask is not None
        attn_mask, padding_mask = fill_attn_mask(attn_mask, padding_mask)
        for i in range(len(self.layers)):
            if self.gradient_checkpointing and self.training:
                # checkpoint: don't store this layer's activations; recompute in backward.
                # positional args follow the layer's signature (x, attn_bias, padding_mask, return_attn)
                x, attn_mask, _ = torch.utils.checkpoint.checkpoint(
                    self.layers[i], x, attn_mask, padding_mask, True, use_reentrant=False
                )
            else:
                x, attn_mask, _ = self.layers[i](
                    x, padding_mask=padding_mask, attn_bias=attn_mask, return_attn=True
                )

        def norm_loss(x, eps=1e-10, tolerance=1.0):
            x = x.float()
            max_norm = x.shape[-1] ** 0.5
            norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            error = torch.nn.functional.relu((norm - max_norm).abs() - tolerance)
            return error

        def masked_mean(mask, value, dim=-1, eps=1e-10):
            return (
                torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))
            ).mean()

        x_norm = norm_loss(x)
        if input_padding_mask is not None:
            token_mask = 1.0 - input_padding_mask.float()
        else:
            token_mask = torch.ones_like(x_norm, device=x_norm.device)
        x_norm = masked_mean(token_mask, x_norm)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        delta_pair_repr = attn_mask - input_attn_mask
        delta_pair_repr, _ = fill_attn_mask(delta_pair_repr, input_padding_mask, 0)
        attn_mask = (
            attn_mask.view(bsz, -1, seq_len, seq_len).permute(0, 2, 3, 1).contiguous()
        )
        delta_pair_repr = (
            delta_pair_repr.view(bsz, -1, seq_len, seq_len)
            .permute(0, 2, 3, 1)
            .contiguous()
        )

        pair_mask = token_mask[..., None] * token_mask[..., None, :]
        delta_pair_repr_norm = norm_loss(delta_pair_repr)
        delta_pair_repr_norm = masked_mean(
            pair_mask, delta_pair_repr_norm, dim=(-1, -2)
        )

        if self.final_head_layer_norm is not None:
            delta_pair_repr = self.final_head_layer_norm(delta_pair_repr)

        return x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
