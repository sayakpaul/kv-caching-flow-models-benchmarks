import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache


torch.set_num_threads(1)
torch.manual_seed(0)
model = QwenImage21Transformer2DModel(
    patch_size=1,
    in_channels=4,
    out_channels=4,
    num_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    context_in_dim=8,
    mlp_ratio=2,
    axes_dims_rope=(4, 6, 6),
    causal_condition=True,
).eval()
with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
    for references in (0, 1, 2):
        for seed in (42, 43):
            generator = torch.Generator().manual_seed(seed)
            text_len = 4 + references
            context = torch.randn(1, text_len, 8, generator=generator)
            reference = torch.randn(1, 4 * references, 4, generator=generator)
            mask = torch.zeros(1, text_len + 1, dtype=torch.bool)
            mask[:, 1 : 1 + references] = True
            mask[:, -1] = True
            cache = QwenImage21KVCache(2)
            maximum = 0.0
            for step in range(4):
                inputs = dict(
                    hidden_states=torch.cat([reference, torch.randn(1, 4, 4, generator=generator)], dim=1),
                    encoder_hidden_states=context,
                    timestep=torch.tensor([1 - step / 4]),
                    img_shapes=[[(1, 2, 2)] * (references + 1)],
                    img_mask=mask,
                    return_dict=False,
                )
                mode = "extract" if step == 0 else "cached"
                cached = model(**inputs, kv_cache=cache, kv_cache_mode=mode)[0][:, -4:]
                full = model(**inputs)[0][:, -4:]
                torch.testing.assert_close(cached, full, atol=2e-5, rtol=2e-5)
                assert torch.isfinite(cached).all()
                maximum = max(maximum, (cached - full).abs().max().item())
            for layer in cache.layer_caches:
                for tensor in layer.get():
                    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
            print({"references": references, "seed": seed, "max_absolute_difference": maximum}, flush=True)
print("QWEN_TINY_CACHE_VALIDATION_PASS", flush=True)
