from pathlib import Path

def test_readme_documents_required_provider_and_model_examples():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "--provider openai" in readme
    assert "--provider google" in readme
    assert "--model MODEL" in readme

def test_readme_explains_provider_specific_credentials():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "GOOGLE_API_KEY" in readme
    assert "required for Google generation or Google embeddings" in readme.lower() or "required for google generation" in readme.lower()
    assert "auth.json" in readme
    assert "browser authorization flow" in readme
    assert "gpt-6-luna" in readme
