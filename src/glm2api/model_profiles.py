from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    native_function_calling: bool
    preferred_format: str
    stream_handler_type: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "glm": ModelProfile("glm", False, "bracket", "bracket"),
    "glm-4": ModelProfile("glm-4", False, "bracket", "bracket"),
    "glm-4v": ModelProfile("glm-4v", False, "bracket", "bracket"),
    "glm-zero-preview": ModelProfile("glm-zero-preview", False, "bracket", "bracket"),
    "glm-deep-research": ModelProfile("glm-deep-research", False, "bracket", "bracket"),
    "default": ModelProfile("default", False, "bracket", "bracket"),
}


def get_model_profile(model: str) -> ModelProfile:
    lower_model = (model or "").lower()
    if lower_model in MODEL_PROFILES:
        return MODEL_PROFILES[lower_model]
    for key, profile in MODEL_PROFILES.items():
        if key != "default" and key in lower_model:
            return profile
    return MODEL_PROFILES["default"]
