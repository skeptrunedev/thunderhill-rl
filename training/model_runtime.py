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


def _validate(spec: ModelSpec) -> ModelSpec:
    if spec not in (LEGACY_SPEC, GEMMA4_SPEC):
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
    from transformers import AutoModelForCausalLM, Gemma4ForCausalLM

    _validate(spec)
    loader = Gemma4ForCausalLM if spec == GEMMA4_SPEC else AutoModelForCausalLM
    kwargs = {"key_mapping": GEMMA4_KEY_MAPPING} if spec == GEMMA4_SPEC else {}
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
    if spec == GEMMA4_SPEC:
        unexpected = [
            key for key in unexpected if not key.startswith(GEMMA4_UNUSED_PREFIXES)
        ]
    if unexpected or any(
        info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Pretrained model weights did not load completely: {info}")


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
        if self.spec == GEMMA4_SPEC:
            # The pinned chat template ends turns with <turn|> (106). Both
            # rollout generation and TRL completion masking use tokenizer EOS.
            # The base tokenizer's <eos> (1) alone does not end a chat answer.
            if tokenizer.convert_tokens_to_ids("<turn|>") != 106:
                raise ValueError("Gemma4 tokenizer has an unexpected turn terminator")
            tokenizer.eos_token = "<turn|>"

    def prompt_features(self, features: dict) -> str:
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
