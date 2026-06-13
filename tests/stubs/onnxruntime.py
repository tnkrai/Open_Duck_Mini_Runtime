"""Test stub for onnxruntime (heavy native wheel; only needed on-robot).

mini_bdx_runtime/__init__.py imports OnnxInfer, which imports onnxruntime at
module level — the session is only created when OnnxInfer is instantiated,
which no test does.
"""


class InferenceSession:  # pragma: no cover - never instantiated in tests
    def __init__(self, *args, **kwargs):
        raise RuntimeError("onnxruntime stub: not available in tests")
