from contextlib import contextmanager
from unittest.mock import patch

import torch

from diffusers import TaylorSeerCacheConfig
from diffusers.hooks.taylorseer_cache import TaylorSeerCacheHook


def cache_config(disable_before=3):
    return TaylorSeerCacheConfig(
        cache_interval=2,
        disable_cache_before_step=disable_before,
        max_order=1,
        taylor_factors_dtype=torch.bfloat16,
    )


@contextmanager
def audit_cache(transformer, audit):
    measure = TaylorSeerCacheHook._measure_should_compute
    forward = transformer.forward
    names = {
        id(module._diffusers_hook.get_hook("taylorseer_cache")): name
        for name, module in transformer.named_modules()
        if hasattr(module, "_diffusers_hook") and module._diffusers_hook.get_hook("taylorseer_cache") is not None
    }

    def traced_measure(hook):
        compute, state = measure(hook)
        audit.setdefault("modules", []).append(
            {
                "module": names[id(hook)],
                "step": state.current_step + 1,
                "compute": compute,
                "orders": {str(k): list(v) for k, v in state.taylor_factors.items()},
            }
        )
        return compute, state

    def traced_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        record = {"mode": kwargs.get("kv_cache_mode")}
        if record["mode"] == "extract":
            cache = result[1]
            layers = cache.double_block_caches + cache.single_block_caches
            record["populated_reference_layers"] = sum(layer.get()[0] is not None for layer in layers)
            record["reference_tokens"] = cache.num_ref_tokens
            assert record["populated_reference_layers"] == len(layers)
        audit.setdefault("transformer", []).append(record)
        return result

    with (
        patch.object(TaylorSeerCacheHook, "_measure_should_compute", new=traced_measure),
        patch.object(transformer, "forward", new=traced_forward),
    ):
        yield


def check_audit(audit, steps, num_layers):
    assert len(audit["transformer"]) == steps
    assert audit["transformer"][0]["mode"] == "extract"
    assert all(x["mode"] == "cached" for x in audit["transformer"][1:])
    assert len(audit["modules"]) == steps * num_layers
    for name in {x["module"] for x in audit["modules"]}:
        rows = [x for x in audit["modules"] if x["module"] == name]
        assert [x["step"] for x in rows if not x["compute"]] == list(range(4, steps + 1, 2))
        assert all(all(orders == [0, 1] for orders in row["orders"].values()) for row in rows if not row["compute"])
