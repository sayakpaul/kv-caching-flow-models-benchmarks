from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def audit_cache(transformer, records):
    forward = transformer.forward

    def traced_forward(*args, **kwargs):
        output = forward(*args, **kwargs)
        mode = kwargs.get("kv_cache_mode")
        cache = kwargs.get("kv_cache")
        record = {
            "mode": mode,
            "causal_condition": transformer.config.causal_condition,
            "output_tokens": output[0].shape[1],
            "img_shapes": kwargs["img_shapes"],
            "cache_layers": 0,
        }
        if cache is not None:
            record["cache_layers"] = len(cache.layer_caches)
            record["prefix_tokens"] = cache.layer_caches[0].get()[0].shape[1]
            record["cache_payload_bytes"] = sum(
                t.numel() * t.element_size() for layer in cache.layer_caches for t in layer.get()
            )
            assert all(
                t.untyped_storage().nbytes() == t.numel() * t.element_size()
                for layer in cache.layer_caches
                for t in layer.get()
            )
        records.append(record)
        return output

    with patch.object(transformer, "forward", new=traced_forward):
        yield
