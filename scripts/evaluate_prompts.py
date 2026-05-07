import json
import re
from typing import Dict, Any, List

def check_json_validity(output: str) -> bool:
    """Checks if the output contains valid JSON (handles markdown formatting)."""
    # Strip markdown code blocks if present
    cleaned = re.sub(r'```(?:json)?\n(.*?)\n```', r'\1', output, flags=re.DOTALL).strip()
    try:
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, TypeError):
        # Try raw just in case
        try:
            json.loads(output)
            return True
        except json.JSONDecodeError:
            return False

def check_no_duplicates(items: List[str]) -> bool:
    """Checks if a list has no duplicate items."""
    if not items:
        return True
    return len(items) == len(set(items))

def validate_structural_constraints(task_name: str, result_obj: Any) -> Dict[str, Any]:
    """Programmatically validates the result object based on the task type."""
    validation_results = {
        "is_valid": True,
        "issues": []
    }

    # Check for duplicates based on the task type attributes
    if task_name == "identify_topics" and hasattr(result_obj, 'topics'):
        if not check_no_duplicates(result_obj.topics):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Duplicate topics found.")

    elif task_name == "extract_key_insights" and hasattr(result_obj, 'insights'):
        if not check_no_duplicates(result_obj.insights):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Duplicate insights found.")

    elif task_name == "extract_decisions" and hasattr(result_obj, 'decisions'):
        if not check_no_duplicates(result_obj.decisions):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Duplicate decisions found.")

    elif task_name == "extract_action_items" and hasattr(result_obj, 'action_items'):
        if not check_no_duplicates(result_obj.action_items):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Duplicate action items found.")

    return validation_results

from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

class JudgeEvaluation(BaseModel):
    score: int = Field(description="Score from 1 to 5, where 5 is excellent and 1 is terrible.")
    reasoning: str = Field(description="A brief explanation of why this score was given.")

def evaluate_with_llm(task_name: str, transcript: str, original_prompt: str, generated_output: str, judge_model_name: str = "gpt-4o") -> JudgeEvaluation:
    """Uses a strong LLM to evaluate the generated output against the transcript and prompt."""

    # We rely on OPENAI_API_KEY being set in the environment
    try:
        judge_llm = ChatOpenAI(model=judge_model_name, temperature=0.0).with_structured_output(JudgeEvaluation)
    except Exception as e:
        return JudgeEvaluation(score=0, reasoning=f"Failed to initialize Judge LLM. Ensure API keys are set. Error: {e}")

    eval_prompt = PromptTemplate.from_template("""
    You are an expert evaluator grading the performance of an AI assistant on a meeting summarization task.

    TASK NAME: {task_name}

    ORIGINAL TRANSCRIPT CHUNK:
    ---
    {transcript}
    ---

    THE PROMPT GIVEN TO THE ASSISTANT:
    ---
    {original_prompt}
    ---

    THE ASSISTANT'S GENERATED OUTPUT:
    ---
    {generated_output}
    ---

    RUBRIC:
    Score the output from 1 to 5 based on:
    - Accuracy: Does the output accurately reflect the transcript?
    - Completeness: Did it capture all necessary elements requested by the prompt?
    - Instruction Following: Did it strictly adhere to the constraints in the prompt?
    - Hallucinations: Are there any fabricated facts not in the transcript? (Should be heavily penalized).

    Provide your score and a brief reasoning.
    """)

    try:
        chain = eval_prompt | judge_llm
        result = chain.invoke({
            "task_name": task_name,
            "transcript": transcript,
            "original_prompt": original_prompt,
            "generated_output": generated_output
        })
        return result
    except Exception as e:
        return JudgeEvaluation(score=0, reasoning=f"Evaluation failed: {e}")


import sys
sys.path.insert(0, 'src')
from meeting_summarizer.main import PromptManager, Config, LLMProvider, _generic_extraction, TranscriptState
from pydantic import BaseModel, Field
import importlib.resources as resources

# Sample Pydantic models to mimic main app
class TopicsOutput(BaseModel):
    topics: List[str] = Field(description="List of main topics discussed")

class InsightsOutput(BaseModel):
    insights: List[str] = Field(description="List of key insights")

class DecisionsOutput(BaseModel):
    decisions: List[str] = Field(description="List of decisions")

class ActionItemsOutput(BaseModel):
    action_items: List[str] = Field(description="List of action items")

class DefaultOutput(BaseModel):
    text: str = Field(description="Extracted text")

def get_model_and_args_for_task(task_name: str, transcript_text: str):
    if task_name in ["identify_topics", "merge_topics"]:
        return TopicsOutput, {"text": transcript_text, "topics": transcript_text, "format_instructions": ""}
    elif task_name == "extract_key_insights":
        return InsightsOutput, {"text": transcript_text, "format_instructions": ""}
    elif task_name == "extract_decisions":
        return DecisionsOutput, {"text": transcript_text, "format_instructions": ""}
    elif task_name == "extract_action_items":
        return ActionItemsOutput, {"text": transcript_text, "format_instructions": ""}
    elif task_name == "summarize_topics":
        return DefaultOutput, {"text": transcript_text, "topic": "General Updates"}
    else:
        return DefaultOutput, {"text": transcript_text, "format_instructions": ""}

def run_evaluation_comparison(transcript_path: str, task_name: str, target_model: str = "openai/gpt-3.5-turbo"):
    """Runs a comparison between v1 and v2 prompts for a given task."""
    print(f"\n--- Starting Evaluation for '{task_name}' ---\n")

    # Load Prompts
    with resources.open_text("meeting_summarizer", "prompts_meeting.yaml") as f:
        pm_v1 = PromptManager(f.read())
    with resources.open_text("meeting_summarizer", "prompts_meeting_v2.yaml") as f:
        pm_v2 = PromptManager(f.read())

    with open(transcript_path, 'r') as f:
        transcript_text = f.read()

    # Setup mock state
    config = Config(model_provider=target_model.split('/')[0], model_name=target_model.split('/')[1], iterative_analysis=False)

    # Run v1
    print("[RUNNING V1 PROMPTS]")
    state_v1 = {
        "config": config,
        "docs": [type('Doc', (object,), {"page_content": transcript_text})()],
        "prompts": pm_v1,
        "full_transcript": transcript_text,
        "progress": 0.0,
        "current_step": "v1_test"
    }

    try:
        # We'll test identify_topics as an example
        output_model, format_args = get_model_and_args_for_task(task_name, transcript_text)
        v1_result = _generic_extraction(state_v1, task_name, "Evaluating v1", 0.1, output_model)
        v1_raw_output = str(v1_result.model_dump())
        v1_struct_val = validate_structural_constraints(task_name, v1_result)

        prompt_str_v1 = pm_v1.get_prompt(task_name, "initial_prompt").format(**format_args)
        v1_judge = evaluate_with_llm(task_name, transcript_text, prompt_str_v1, v1_raw_output)
    except Exception as e:
        print(f"V1 Evaluation Failed: {e}")
        v1_raw_output, v1_struct_val, v1_judge = "FAIL", {"is_valid": False}, JudgeEvaluation(score=0, reasoning=str(e))

    # Run v2
    print("[RUNNING V2 PROMPTS]")
    state_v2 = {
        "config": config,
        "docs": [type('Doc', (object,), {"page_content": transcript_text})()],
        "prompts": pm_v2,
        "full_transcript": transcript_text,
        "progress": 0.0,
        "current_step": "v2_test"
    }

    try:
        output_model, format_args = get_model_and_args_for_task(task_name, transcript_text)
        v2_result = _generic_extraction(state_v2, task_name, "Evaluating v2", 0.1, output_model)
        v2_raw_output = str(v2_result.model_dump())
        v2_struct_val = validate_structural_constraints(task_name, v2_result)

        prompt_str_v2 = pm_v2.get_prompt(task_name, "initial_prompt").format(**format_args)
        v2_judge = evaluate_with_llm(task_name, transcript_text, prompt_str_v2, v2_raw_output)
    except Exception as e:
        print(f"V2 Evaluation Failed: {e}")
        v2_raw_output, v2_struct_val, v2_judge = "FAIL", {"is_valid": False}, JudgeEvaluation(score=0, reasoning=str(e))

    # Report
    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)

    print("\n--- V1 Performance ---")
    print(f"Structural Validation: {'PASS' if v1_struct_val.get('is_valid') else 'FAIL'}")
    print(f"Judge Score: {v1_judge.score}/5")
    print(f"Judge Reasoning: {v1_judge.reasoning}")
    print(f"Output Preview: {v1_raw_output[:200]}...")

    print("\n--- V2 Performance ---")
    print(f"Structural Validation: {'PASS' if v2_struct_val.get('is_valid') else 'FAIL'}")
    print(f"Judge Score: {v2_judge.score}/5")
    print(f"Judge Reasoning: {v2_judge.reasoning}")
    print(f"Output Preview: {v2_raw_output[:200]}...")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Prompt Evaluations")
    parser.add_argument("--transcript", type=str, default="scripts/sample_transcript.txt", help="Path to sample transcript")
    parser.add_argument("--task", type=str, default="identify_topics", help="Task to evaluate (e.g. identify_topics)")
    parser.add_argument("--model", type=str, default="openai/gpt-3.5-turbo", help="Target model to test the prompts against")
    args = parser.parse_args()

    run_evaluation_comparison(args.transcript, args.task, args.model)
