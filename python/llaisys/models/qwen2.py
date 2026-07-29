import json
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
from pathlib import Path
from typing import Sequence

import safetensors

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS, LlaisysQwen2Meta


_DTYPE_MAP = {
    "float32": DataType.F32,
    "float16": DataType.F16,
    "bfloat16": DataType.BF16,
}


class Qwen2:
    def __init__(
        self,
        model_path,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
    ):
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as file:
            config = json.load(file)

        dtype_name = str(
            config.get("dtype", config.get("torch_dtype", "bfloat16"))
        ).removeprefix("torch.")
        if dtype_name not in _DTYPE_MAP:
            raise ValueError(f"Unsupported Qwen2 weight dtype: {dtype_name}")

        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config.get("head_dim", hidden_size // num_heads))
        self._max_sequence_length = int(config["max_position_embeddings"])
        self._end_token = int(config["eos_token_id"])
        self._model = None

        meta = LlaisysQwen2Meta(
            dtype=int(_DTYPE_MAP[dtype_name]),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=head_dim,
            di=int(config["intermediate_size"]),
            maxseq=self._max_sequence_length,
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config.get("rope_theta", 10000.0)),
            end_token=self._end_token,
        )
        device_ids = (c_int * 1)(device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as data:
                for name in data.keys():
                    tensor = data.get_tensor(name).contiguous()
                    LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                        self._model,
                        name.encode("utf-8"),
                        c_void_p(tensor.data_ptr()),
                        c_size_t(tensor.numel() * tensor.element_size()),
                    )

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if not inputs:
            raise ValueError("Qwen2 generation requires at least one input token")
        if max_new_tokens is None:
            max_new_tokens = 20
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(inputs) + max_new_tokens > self._max_sequence_length:
            raise ValueError("Requested generation exceeds the model context length")

        output = [int(token) for token in inputs]
        if max_new_tokens == 0:
            return output

        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)
        prompt = (c_int64 * len(output))(*output)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, prompt, len(output))
        )

        for step in range(max_new_tokens):
            output.append(next_token)
            if next_token == self._end_token or step + 1 == max_new_tokens:
                break
            token = c_int64(next_token)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, byref(token), 1)
            )

        return output
