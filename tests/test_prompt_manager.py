import sys
from unittest.mock import MagicMock, patch

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
import logging

# We need to make sure yaml is available or mock it too.
try:
    import yaml
except ImportError:
    sys.modules["yaml"] = MagicMock()
    import yaml

from meeting_summarizer.main import PromptManager

def test_prompt_manager_valid_yaml():
    valid_yaml = """
task1:
  type1: "Template 1"
  type2: "Template 2"
task2:
  type1: "Template 3"
"""
    pm = PromptManager(valid_yaml)
    assert pm.prompts == {
        "task1": {"type1": "Template 1", "type2": "Template 2"},
        "task2": {"type1": "Template 3"}
    }

def test_prompt_manager_malformed_yaml(caplog):
    malformed_yaml = """
task1:
  type1: "Template 1"
  type2: "Template 2
  unclosed string
"""
    with caplog.at_level(logging.ERROR):
        pm = PromptManager(malformed_yaml)

    assert pm.prompts == {}
    assert "Failed to parse prompts" in caplog.text

def test_prompt_manager_get_prompt():
    valid_yaml = """
task1:
  type1: "Hello {name}"
"""
    pm = PromptManager(valid_yaml)

    # We need to mock PromptTemplate.from_template
    with patch("meeting_summarizer.main.PromptTemplate") as mock_prompt_template:
        mock_instance = MagicMock()
        mock_prompt_template.from_template.return_value = mock_instance

        prompt = pm.get_prompt("task1", "type1")

        mock_prompt_template.from_template.assert_called_once_with("Hello {name}")
        assert prompt == mock_instance

def test_prompt_manager_get_prompt_missing_task():
    valid_yaml = """
task1:
  type1: "Hello {name}"
"""
    pm = PromptManager(valid_yaml)
    with pytest.raises(ValueError) as excinfo:
        pm.get_prompt("task2", "type1")
    assert "Prompt for task 'task2' and type 'type1' not found." in str(excinfo.value)

def test_prompt_manager_get_prompt_missing_type():
    valid_yaml = """
task1:
  type1: "Hello {name}"
"""
    pm = PromptManager(valid_yaml)
    with pytest.raises(ValueError) as excinfo:
        pm.get_prompt("task1", "type2")
    assert "Prompt for task 'task1' and type 'type2' not found." in str(excinfo.value)
