"""Pinned model identity and prompt formatting shared by training and evaluation."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from lap_policy import RoadTelemetry


@dataclass(frozen=True)
class ModelSpec:
    model: str
    revision: str
    dtype: str
    prompt_style: str


LEGACY_SPEC = ModelSpec(
    "unsloth/gemma-3-270m-it",
    "23cf460f6bb16954176b3ddcc8d4f250501458a9",
    "float32",
    "raw",
)
GEMMA4_SPEC = ModelSpec(
    "google/gemma-4-E4B-it",
    "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "bfloat16",
    "gemma4_chat",
)
GEMMA4_NATIVE_SPEC = ModelSpec(
    GEMMA4_SPEC.model, GEMMA4_SPEC.revision, GEMMA4_SPEC.dtype, "gemma4_native_tools"
)
FUNCTIONGEMMA_SPEC = ModelSpec(
    "google/functiongemma-270m-it",
    "39eccb091651513a5dfb56892d3714c1b5b8276c",
    "float32",
    "functiongemma_native_tools",
)
QWEN27B_SPEC = ModelSpec(
    "Qwen/Qwen3.8-27B",
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "bfloat16",
    "qwen_native_tools",
)
QWEN_KEY_MAPPING = {r"^model\.language_model\.": "model."}
SPEC_FILENAME = "model_spec.json"
GEMMA4_KEY_MAPPING = {r"^model\.language_model\.": "model."}
GEMMA4_UNUSED_PREFIXES = (
    "model.audio_tower.",
    "model.vision_tower.",
    "model.embed_audio.",
    "model.embed_vision.",
)
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


QWEN_LORA_TARGET_MODULES = LORA_TARGET_MODULES + (
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)


def _validate(spec: ModelSpec) -> ModelSpec:
    if spec not in (LEGACY_SPEC, GEMMA4_SPEC, GEMMA4_NATIVE_SPEC, FUNCTIONGEMMA_SPEC, QWEN27B_SPEC):
        raise ValueError(f"Unsupported model specification: {spec}")
    return spec


def _check_adapter(adapter: Path, spec: ModelSpec) -> None:
    config = adapter / "adapter_config.json"
    if config.exists():
        base = json.loads(config.read_text()).get("base_model_name_or_path")
        if base != spec.model:
            raise ValueError(
                f"Adapter base model {base!r} does not match {spec.model!r}"
            )


def read_spec(adapter: Path) -> ModelSpec:
    adapter = Path(adapter)
    path = adapter / SPEC_FILENAME
    spec = (
        _validate(ModelSpec(**json.loads(path.read_text())))
        if path.exists()
        else LEGACY_SPEC
    )
    _check_adapter(adapter, spec)
    return spec


def write_spec(adapter: Path, spec: ModelSpec) -> None:
    adapter = Path(adapter)
    _validate(spec)
    _check_adapter(adapter, spec)
    path = adapter / SPEC_FILENAME
    if path.exists() and read_spec(adapter) != spec:
        raise ValueError("Cannot replace a different model specification")
    adapter.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(spec), indent=2) + "\n")


def load_base(spec: ModelSpec, device: str = "cuda", *, dtype: str | None = None):
    """Load only the text model, retaining SDPA and the model's normal KV cache."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Gemma4ForCausalLM, Qwen3_5ForCausalLM

    _validate(spec)
    loader = Gemma4ForCausalLM if spec in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC) else AutoModelForCausalLM
    kwargs = {"key_mapping": GEMMA4_KEY_MAPPING} if spec in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC) else {}
    if spec == QWEN27B_SPEC:
        loader = Qwen3_5ForCausalLM
        config = AutoConfig.from_pretrained(spec.model, revision=spec.revision).text_config
        kwargs = {"key_mapping": QWEN_KEY_MAPPING, "config": config}
        if str(device).startswith("cuda"):
            print(json.dumps({"qwen_fast_kernels": require_qwen_fast_kernels()}), flush=True)
    model, info = loader.from_pretrained(
        spec.model,
        revision=spec.revision,
        dtype=getattr(torch, dtype or spec.dtype),
        attn_implementation="sdpa",
        device_map={"": device},
        output_loading_info=True,
        **kwargs,
    )
    _validate_loading_info(spec, info)
    return model


def _validate_loading_info(spec: ModelSpec, info: dict) -> None:
    unexpected = info.get("unexpected_keys", ())
    if spec in (GEMMA4_SPEC, GEMMA4_NATIVE_SPEC):
        unexpected = [
            key for key in unexpected if not key.startswith(GEMMA4_UNUSED_PREFIXES)
        ]
    if spec == QWEN27B_SPEC:
        unexpected = [key for key in unexpected if not key.startswith(("model.visual.", "mtp."))]
    if unexpected or any(
        info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Pretrained model weights did not load completely: {info}")


def require_qwen_fast_kernels():
    """Fail before allocating weights if Transformers selected reference kernels."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    expected = {
        "causal_conv1d_fn": "causal_conv1d",
        "causal_conv1d_update": "causal_conv1d",
        "torch_chunk_gated_delta_rule": "fla",
        "torch_recurrent_gated_delta_rule": "fla",
    }
    import inspect
    evidence = {}
    for name, package in expected.items():
        function = getattr(module, name)
        seen = set()
        implementations = []
        while callable(function) and id(function) not in seen:
            seen.add(id(function))
            if inspect.isfunction(function):
                implementation = inspect.getclosurevars(function).nonlocals.get("implementation")
                if implementation is not None:
                    implementations.append(implementation)
            function = getattr(function, "__wrapped__", None)
        selected = next((f for f in implementations if getattr(f, "__module__", "").startswith(package)), None)
        if selected is None:
            raise RuntimeError(f"Qwen optimized kernel unavailable: {name}; install compatible {package}")
        evidence[name] = selected.__module__ + "." + selected.__name__
    return evidence


def inference_precision(model):
    """Use the same BF16 compute policy inside and outside Accelerate."""
    import torch

    return torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=model.dtype == torch.bfloat16,
    )


class PolicyRoadTelemetry(RoadTelemetry):
    def __init__(self, spec: ModelSpec, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.spec = _validate(spec)
        self.tokenizer = tokenizer
        self.native_tools = None
        if self.spec == QWEN27B_SPEC:
            from qwen_tools import QwenBikeTools
            self.native_tools = QwenBikeTools(tokenizer)
            tokenizer.eos_token = self.native_tools.tool_stop
        elif self.spec == FUNCTIONGEMMA_SPEC:
            from functiongemma_tools import FunctionGemmaBikeTools
            self.native_tools = FunctionGemmaBikeTools(tokenizer)
            tokenizer.eos_token = self.native_tools.tool_stop
        elif self.spec == GEMMA4_NATIVE_SPEC:
            from native_tools import NativeBikeTools
            self.native_tools = NativeBikeTools(tokenizer)
            tokenizer.eos_token = "<|tool_response>"
        elif self.spec == GEMMA4_SPEC:
            # The pinned chat template ends turns with <turn|> (106). Both
            # rollout generation and TRL completion masking use tokenizer EOS.
            # The base tokenizer's <eos> (1) alone does not end a chat answer.
            if tokenizer.convert_tokens_to_ids("<turn|>") != 106:
                raise ValueError("Gemma4 tokenizer has an unexpected turn terminator")
            tokenizer.eos_token = "<turn|>"

    def prompt_features(self, features: dict) -> str:
        if self.native_tools is not None:
            return self.native_tools.prompt(features)
        raw = super().prompt_features(features)
        if self.spec.prompt_style == "raw":
            return raw
        raw = raw.replace(
            "Ride the track safely.",
            "Race forward and complete the lap as quickly as possible while staying on track.",
            1,
        ).replace(
            "Steer -1000..1000; pedals 0..100.",
            "Use exactly four integers. Steer -1000..1000; pedals 0..100. "
            "Do not use decimal values or negative pedal values.",
            1,
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def parse_completion(self, completion):
        if self.native_tools is not None:
            return self.native_tools.parse_completion(completion)
        from lap_policy import parse_action
        return parse_action(completion)
