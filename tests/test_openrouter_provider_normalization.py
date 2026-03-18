from token_difr.openrouter_api import _extract_providers, _normalize_provider_slug


def test_normalize_provider_slug_canonicalizes_hyphenated_providers() -> None:
    assert _normalize_provider_slug("Atlas Cloud") == "atlas-cloud"
    assert _normalize_provider_slug("atlascloud") == "atlas-cloud"
    assert _normalize_provider_slug("atlascloud/fp8") == "atlas-cloud/fp8"
    assert _normalize_provider_slug("Google") == "google-vertex"
    assert _normalize_provider_slug("Google Vertex") == "google-vertex"


def test_extract_providers_returns_canonical_provider_ids() -> None:
    model_entry = {
        "providers": [{"name": "Atlas Cloud"}, {"name": "Google"}],
        "top_provider": {"name": "Google"},
        "endpoints": [
            {"name": "Atlas Cloud | FP8"},
            {"name": "Google Vertex"},
            {"provider_name": "deepinfra"},
        ],
    }

    providers = _extract_providers(model_entry)

    assert "atlas-cloud" in providers
    assert "atlas-cloud/fp8" in providers
    assert "google-vertex" in providers
    assert "deepinfra" in providers
    assert "atlascloud" not in providers
    assert "atlascloud/fp8" not in providers
    assert "google" not in providers
