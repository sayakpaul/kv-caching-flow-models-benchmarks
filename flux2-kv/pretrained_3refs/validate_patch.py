import json
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel

from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinKVPipeline,
    Flux2Transformer2DModel,
)
from diffusers.models.transformers import transformer_flux2


torch.set_num_threads(1)
torch.manual_seed(0)
transformer = Flux2Transformer2DModel(
    patch_size=1,
    in_channels=4,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=16,
    num_attention_heads=2,
    joint_attention_dim=16,
    timestep_guidance_channels=256,
    axes_dims_rope=[4, 4, 4, 4],
    guidance_embeds=False,
).eval()
vae = AutoencoderKLFlux2(
    sample_size=32,
    in_channels=3,
    out_channels=3,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
    block_out_channels=(4, 4),
    layers_per_block=1,
    latent_channels=1,
    norm_num_groups=1,
    use_quant_conv=False,
    use_post_quant_conv=False,
).eval()
pipe = Flux2KleinKVPipeline(
    scheduler=FlowMatchEulerDiscreteScheduler(),
    text_encoder=None,
    tokenizer=None,
    transformer=transformer,
    vae=vae,
)
pipe.set_progress_bar_config(disable=True)
prompt_embeds = torch.randn(1, 16, 16)
images = [Image.fromarray(np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)) for seed in (0, 1, 2)]
results = []
with sdpa_kernel(SDPBackend.MATH):
    for count in (0, 1, 2, 3):
        outputs = []
        for use_cache in (True, False):
            kwargs = dict(
                image=images[:count] if count else None,
                prompt_embeds=prompt_embeds,
                num_inference_steps=4,
                height=32,
                width=32,
                output_type="latent",
                generator=torch.Generator().manual_seed(42),
                use_kv_cache=use_cache,
            )
            if use_cache:
                outputs.append(pipe(**kwargs).images)
            else:
                with patch.object(transformer_flux2, "Flux2KVCache", side_effect=AssertionError("Cache allocated")):
                    outputs.append(pipe(**kwargs).images)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        results.append({"reference_images": count, "max_abs_difference": (outputs[0] - outputs[1]).abs().max().item()})
print(json.dumps({"tests": results, "uncached_cache_allocations": 0}, indent=2))
