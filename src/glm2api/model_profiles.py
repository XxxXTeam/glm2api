from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelProfile:
    id: str
    native_function_calling: bool
    preferred_format: str
    stream_handler_type: str


MODEL_PROFILES: dict[str, ModelProfile] = {
    "glm": ModelProfile("glm", False, "xml", "xml"),
    "glm-4": ModelProfile("glm-4", False, "xml", "xml"),
    "glm-4v": ModelProfile("glm-4v", False, "xml", "xml"),
    "glm-zero-preview": ModelProfile("glm-zero-preview", False, "xml", "xml"),
    "glm-deep-research": ModelProfile("glm-deep-research", False, "xml", "xml"),
    "default": ModelProfile("default", False, "xml", "xml"),
}


def get_model_profile(model: str) -> ModelProfile:
    lower_model = (model or "").lower()
    if lower_model in MODEL_PROFILES:
        return MODEL_PROFILES[lower_model]
    for key, profile in MODEL_PROFILES.items():
        if key != "default" and key in lower_model:
            return profile
    return MODEL_PROFILES["default"]
