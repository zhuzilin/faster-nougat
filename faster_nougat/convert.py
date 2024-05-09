import torch
import mlx.core as mx
import mlx.nn as nn
import numpy as np


def deconvert(mlx_module):
    if isinstance(mlx_module, mx.array):
        return torch.from_numpy(np.array(mlx_module))
    else:
        raise NotImplementedError()


def convert(hf_module):
    if isinstance(hf_module, torch.Tensor):
        return mx.array(hf_module.numpy())

    if isinstance(hf_module, torch.nn.Linear):
        has_bias = hf_module.bias is not None
        mlx_module = nn.Linear(
            hf_module.in_features,
            hf_module.out_features,
            bias=has_bias,
        )
        weights = [("weight", convert(hf_module.weight))]
        if has_bias:
            weights.append(("bias", convert(hf_module.bias)))
    elif isinstance(hf_module, torch.nn.LayerNorm):
        has_bias = hf_module.bias is not None
        mlx_module = nn.LayerNorm(
            dims=hf_module.normalized_shape[0],
            eps=hf_module.eps,
            affine=hf_module.elementwise_affine,
            bias=has_bias,
        )
        has_weight = hf_module.weight is not None
        weights = []
        if has_weight:
            weights.append(("weight", convert(hf_module.weight)))
        if has_bias:
            weights.append(("bias", convert(hf_module.bias)))

    elif isinstance(hf_module, torch.nn.Embedding):
        mlx_module = nn.Embedding(
            num_embeddings=hf_module.num_embeddings,
            dims=hf_module.embedding_dim,
        )
        weights = [("weight", convert(hf_module.weight))]
    else:
        raise NotImplementedError(f"{type(hf_module)}")

    mlx_module.load_weights(weights)
    return mlx_module
