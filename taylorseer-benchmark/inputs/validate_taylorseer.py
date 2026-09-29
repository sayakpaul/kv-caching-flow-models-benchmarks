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
images = [
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)) for seed in (0, 1, 2)
]
kwargs = {
    "image": images[0],
    "prompt_embeds": prompt_embeds,
    "height": 32,
    "width": 32,
    "output_type": "latent",
}

from config import audit_cache, cache_config, check_audit

with sdpa_kernel(SDPBackend.MATH):
    for count in (1, 2):
        kwargs["image"] = images[:count]
        for steps in (4, 8):
            baseline = pipe(**kwargs, num_inference_steps=steps, generator=torch.Generator().manual_seed(42)).images
            transformer.enable_cache(cache_config(disable_before=steps + 1))
            with transformer.cache_context("cond"):
                control = pipe(**kwargs, num_inference_steps=steps, generator=torch.Generator().manual_seed(42)).images
            torch.testing.assert_close(baseline, control, atol=0, rtol=0)
            transformer.disable_cache()
            transformer.enable_cache(cache_config())
            outputs = []
            for repeat in range(2):
                audit = {}
                with transformer.cache_context("cond"), audit_cache(transformer, audit):
                    output = pipe(
                        **kwargs, num_inference_steps=steps, generator=torch.Generator().manual_seed(42)
                    ).images
                assert torch.isfinite(output).all()
                check_audit(audit, steps, 4)
                outputs.append(output)
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            transformer.disable_cache()
            print(
                {
                    "references": count,
                    "steps": steps,
                    "control_exact": True,
                    "repeat_exact": True,
                    "prediction_steps": list(range(4, steps + 1, 2)),
                },
                flush=True,
            )
print("TINY_TAYLORSEER_VALIDATION_PASS", flush=True)
