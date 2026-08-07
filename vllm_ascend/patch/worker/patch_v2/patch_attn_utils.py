import vllm

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
    init_kv_cache,
)

vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.attn_utils.init_kv_cache = init_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
# model_runner imports init_kv_cache via `from ... import`, so the module-level
# binding in model_runner must be patched too for the override to take effect.
vllm.v1.worker.gpu.model_runner.init_kv_cache = init_kv_cache
