import sys
from unittest.mock import MagicMock

# Mocking dependencies that are not available in the environment
# to allow importing from meeting_summarizer.main
mock_modules = [
    "dotenv",
    "langchain_core",
    "langchain_core.runnables",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
    "langchain_core.exceptions",
    "langchain_core.documents",
    "langchain_core.caches",
    "langchain_core.callbacks",
    "langchain_text_splitters",
    "langgraph",
    "langgraph.graph",
    "pydantic",
    "tiktoken",
    "langchain_ollama",
    "langchain_ollama.chat_models",
    "langchain_openai",
    "langchain_google_genai",
]

for module_name in mock_modules:
    if module_name not in sys.modules:
        sys.modules[module_name] = MagicMock()

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# We need to make sure yaml is available or mock it too.
try:
    import yaml
except ImportError:
    sys.modules["yaml"] = MagicMock()
    import yaml

from meeting_summarizer.main import Config, DocumentValidator, TranscriptValidationError, TranscriptAnalyzerError, _deduplicate_list

def test_config_defaults():
    config = Config()
    assert config.model_provider == "ollama"
    assert config.model_name == "llama3"
    assert config.output_format == "console"

def test_deduplicate_list():
    assert _deduplicate_list(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert _deduplicate_list([]) == []

def test_validate_transcript_not_found():
    config = Config()
    with pytest.raises(TranscriptValidationError) as excinfo:
        DocumentValidator.validate_transcript("nonexistent.txt", config)
    assert "File not found" in str(excinfo.value)

def test_validate_transcript_unsupported_ext(tmp_path):
    config = Config(allowed_extensions=[".txt"])
    bad_file = tmp_path / "transcript.pdf"
    bad_file.write_text("dummy")
    with pytest.raises(TranscriptValidationError) as excinfo:
        DocumentValidator.validate_transcript(str(bad_file), config)
    assert "Unsupported file type" in str(excinfo.value)

def test_validate_transcript_too_large(tmp_path):
    config = Config(max_file_size_mb=0) # 0 MB max size
    large_file = tmp_path / "large.txt"
    large_file.write_text("a" * 1024)
    with pytest.raises(TranscriptValidationError) as excinfo:
        DocumentValidator.validate_transcript(str(large_file), config)
    assert "File too large" in str(excinfo.value)

@patch("meeting_summarizer.main.IterativeRefiner")
@patch("meeting_summarizer.main.LLMProvider")
def test_generic_extraction_iterative(mock_llm_provider, mock_iterative_refiner, tmp_path):
    from meeting_summarizer.main import _generic_extraction, TranscriptState, Config

    mock_llm = MagicMock()
    mock_llm_provider.create_llm.return_value = mock_llm

    mock_refiner_instance = MagicMock()
    class DummyOutput:
        topics = ["AI", "ML"]
    mock_refiner_instance.process.return_value = DummyOutput()
    mock_iterative_refiner.return_value = mock_refiner_instance

    mock_prompts = MagicMock()
    mock_prompts.get_prompt.return_value = "prompt"

    state = {
        'config': Config(iterative_analysis=True),
        'docs': ["dummy doc"],
        'prompts': mock_prompts,
        'full_transcript': "dummy transcript",
        'progress': 0.0,
        'current_step': ''
    }

    result = _generic_extraction(
        state, "task_name", "progress_msg", 0.1, DummyOutput
    )

    assert result.topics == ["AI", "ML"]
    mock_iterative_refiner.assert_called_once()
    mock_refiner_instance.process.assert_called_once_with(["dummy doc"])
