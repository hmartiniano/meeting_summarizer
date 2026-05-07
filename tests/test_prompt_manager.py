import pytest
import logging
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
    prompt = pm.get_prompt("task1", "type1")
    assert prompt.template == "Hello {name}"
    assert prompt.input_variables == ["name"]

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
