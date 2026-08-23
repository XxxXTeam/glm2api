from glm2api.config import BUILTIN_MODEL_ALIASES, BUILTIN_EXPOSED_MODELS
from glm2api.model_variants import expand_model_variants, split_model_features
from glm2api.services.translator import resolve_chat_mode, resolve_networking, resolve_upstream_model


class _Config:
    glm_assistant_id = "65940acff94777010aa6b796"
    model_aliases = {"glm-4": "glm-4", "custom": "65940acff94777010aa6b797"}


def test_expand_model_variants_adds_think_search_and_combined_suffixes():
    models = expand_model_variants(("glm-4", "glm-image-1"), excluded_models={"glm-image-1"})

    assert models == [
        "glm-4",
        "glm-4-think",
        "glm-4-search",
        "glm-4-think-search",
        "glm-image-1",
    ]


def test_split_model_features_accepts_think_search_in_either_order():
    assert split_model_features("glm-4-think-search") == ("glm-4", {"think", "search"})
    assert split_model_features("glm-4-search-think") == ("glm-4", {"think", "search"})
    assert split_model_features("glm-deep-research") == ("glm-deep-research", set())


def test_variant_model_resolves_to_base_upstream_model():
    upstream_model, assistant_id = resolve_upstream_model("glm-4-think-search", _Config())
    custom_upstream, custom_assistant_id = resolve_upstream_model("custom-search", _Config())

    assert upstream_model == "glm-4"
    assert assistant_id == _Config.glm_assistant_id
    assert custom_upstream == "65940acff94777010aa6b797"
    assert custom_assistant_id == "65940acff94777010aa6b797"


def test_model_suffixes_resolve_chat_mode_and_networking_matrix():
    assert resolve_chat_mode("glm-4-think", None, None) == "zero"
    assert resolve_networking("glm-4-think", None) is False

    assert resolve_chat_mode("glm-4-search", None, None) == ""
    assert resolve_networking("glm-4-search", None) is True

    assert resolve_chat_mode("glm-4-think-search", None, None) == "zero"
    assert resolve_networking("glm-4-think-search", None) is True

    assert resolve_chat_mode("glm-4", None, None) == ""
    assert resolve_networking("glm-4", None) is False


def test_existing_thinking_model_name_still_enables_chat_mode():
    assert resolve_chat_mode("glm-4.1v-thinking-flashx", None, None) == "zero"


def test_glm_5_2_is_exposed_and_passed_through():
    assert "glm-5.2" in BUILTIN_EXPOSED_MODELS
    assert BUILTIN_MODEL_ALIASES["glm-5.2"] == "glm-5.2"


def test_glm_5_3_is_exposed_and_passed_through():
    assert "glm-5.3" in BUILTIN_EXPOSED_MODELS
    assert BUILTIN_MODEL_ALIASES["glm-5.3"] == "glm-5.3"
