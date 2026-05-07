#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A LangChain/LangGraph program to analyze meeting transcripts with enhanced
error handling, dependency injection, and configuration management.

This version uses a generic IterativeRefiner class to process the document
chunk-by-chunk for all analysis steps, ensuring focused and accurate results.
The executive summary is generated at the end by summarizing the topics.

Prerequisites:
- pip install langchain langgraph langchain-openai langchain-google-genai langchain-ollama pydantic pyyaml tiktoken

Usage:
    python meeting_analyzer.py <path_to_transcript.txt> --model <ollama|openai|google> [options]
    python meeting_analyzer.py --config config.yaml
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, TypedDict, Optional, Callable
import logging
import yaml
import json
import csv
from json.decoder import JSONDecodeError
from importlib import resources
from dotenv import load_dotenv

from langchain_core.runnables import Runnable
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    BaseOutputParser,
    PydanticOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# --- Dynamic Imports for LLM Providers and Tokenizer ---
try:
    from langchain_ollama.chat_models import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Rebuild Pydantic models after all imports to resolve forward references
if ChatOllama:
    ChatOllama.model_rebuild()
if ChatOpenAI:
    ChatOpenAI.model_rebuild()
if ChatGoogleGenerativeAI:
    ChatGoogleGenerativeAI.model_rebuild()


# --- Setup Logging ---
log_level = logging.INFO
if "--log-level" in sys.argv:
    try:
        level_index = sys.argv.index("--log-level") + 1
        if level_index < len(sys.argv):
            log_level = getattr(logging, sys.argv[level_index].upper(), logging.INFO)
    except (ValueError, IndexError):
        pass  # Use default if argument is malformed
logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Configuration and State Management ---


@dataclass
class Config:
    """Configuration class for the Transcript Analyzer."""

    model_provider: str = "ollama"
    model_name: str = "llama3"  # Unified model name field
    ollama_model_options: Dict[str, Any] = field(
        default_factory=lambda: {"num_ctx": 16384, "num_predict": 8192}
    )
    chunk_size: int = 4000
    chunk_overlap: int = 400
    max_file_size_mb: int = 50
    output_format: str = "console"
    output_file: Optional[str] = None
    max_retries: int = 3
    retry_delay: float = 2.0
    enable_progress_bar: bool = True
    allowed_extensions: List[str] = field(default_factory=lambda: [".txt"])
    interview_mode: bool = (
        False  # New: Use interview prompts instead of meeting prompts
    )
    merge_topics: bool = False  # New: Whether to merge topics
    iterative_analysis: bool = (
        False  # New: Process full transcript at once, bypasses iterative refiner
    )
    separate_topic_summarization: bool = (
        False  # New: Summarize topics from extracted text
    )
    extract_action_items: bool = False  # New: Extract action items
    openai_api_base_url: Optional[str] = None  # New: Base URL for OpenAI API

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        try:
            with open(config_path, "r") as f:
                return cls(**yaml.safe_load(f))
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls()


class PromptManager:
    """Manages loading and accessing prompt templates from a YAML file."""

    def __init__(self, prompts_content: str):
        self.prompts = self._parse_prompts(prompts_content)

    def _parse_prompts(self, prompts_content: str) -> Dict[str, Any]:
        try:
            return yaml.safe_load(prompts_content)
        except Exception as e:
            logger.error(f"Failed to parse prompts: {e}")
            return {}

    def get_prompt(self, task: str, prompt_type: str) -> PromptTemplate:
        try:
            template = self.prompts[task][prompt_type]
            return PromptTemplate.from_template(template)
        except KeyError:
            raise ValueError(
                f"Prompt for task '{task}' and type '{prompt_type}' not found."
            )


class TranscriptState(TypedDict):
    """State representation for the transcript processing workflow."""

    transcript_path: str
    config: Config
    prompts: Any  # New: Store loaded prompts
    full_transcript: Optional[str]
    token_count: int
    docs: Optional[List[Document]]
    current_step: str
    progress: float
    analysis: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    processing_time: Dict[str, float]


class TranscriptAnalyzerError(Exception):
    """Custom exception for analyzer errors."""

    pass


class TranscriptValidationError(TranscriptAnalyzerError):
    """Custom exception for validation errors that should not be retried."""

    pass


# --- Pydantic Models for Structured Output ---
class TopicsOutput(BaseModel):
    topics: List[str] = Field(description="List of identified topics.")


class InsightsOutput(BaseModel):
    insights: List[str] = Field(description="List of key insights.")


class DecisionsOutput(BaseModel):
    decisions: List[str] = Field(description="List of decisions discussed.")


class ActionItemsOutput(BaseModel):
    action_items: List[str] = Field(description="List of extracted action items.")


class MergedTopicsOutput(BaseModel):
    """Pydantic model for the output of the topic merging step."""

    merged_topics: Dict[str, List[str]] = Field(
        description="A dictionary where keys are the new, concise, merged topic titles "
        "and values are the original topic strings that were merged under that key."
    )


# --- Core Components: LLM, Validation, Decorators ---


class LLMProvider:
    """Factory class for creating Chat LLM instances."""

    @staticmethod
    def create_llm(config: Config, json_mode: bool = False):
        try:
            provider_method = getattr(
                LLMProvider, f"_create_{config.model_provider}_llm", None
            )
            if not provider_method:
                raise TranscriptAnalyzerError(
                    f"Unsupported model provider: {config.model_provider}"
                )
            return provider_method(config, json_mode)
        except Exception as e:
            logger.error(f"Failed to create LLM: {e}")
            raise TranscriptAnalyzerError(f"LLM initialization failed: {e}")

    @staticmethod
    def _create_ollama_llm(config: Config, json_mode: bool):
        if not ChatOllama:
            raise ImportError("Ollama not found. Run 'pip install langchain-ollama'")
        logger.info(f"Initializing ChatOllama model: {config.model_name}")
        params = {
            "model": config.model_name,
            "temperature": 0,
            **config.ollama_model_options,
        }
        if json_mode:
            params["format"] = "json"
        return ChatOllama(**params)

    @staticmethod
    def _create_openai_llm(config: Config, json_mode: bool):
        if not ChatOpenAI:
            raise ImportError("OpenAI not found. Run 'pip install langchain-openai'")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")
        logger.info(f"Initializing ChatOpenAI model: {config.model_name}")
        params = {"model": config.model_name, "temperature": 0}
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if config.openai_api_base_url:
            params["base_url"] = config.openai_api_base_url
        return ChatOpenAI(**params)

    @staticmethod
    def _create_google_llm(config: Config, json_mode: bool):
        if not ChatGoogleGenerativeAI:
            raise ImportError(
                "Google GenAI not found. Run 'pip install langchain-google-genai'"
            )
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not set")
        logger.info(f"Initializing ChatGoogleGenerativeAI model: {config.model_name}")
        return ChatGoogleGenerativeAI(model=config.model_name, temperature=0)


class DocumentValidator:
    """Validates transcript files before processing."""

    @staticmethod
    def validate_transcript(transcript_path: str, config: Config) -> None:
        path = Path(transcript_path)
        if not path.exists():
            raise TranscriptValidationError(f"File not found: {transcript_path}")
        if path.suffix.lower() not in config.allowed_extensions:
            raise TranscriptValidationError(f"Unsupported file type: {path.suffix}")
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            raise TranscriptValidationError(
                f"File too large: {file_size_mb:.1f}MB > {config.max_file_size_mb}MB"
            )
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                f.read(1024)
        except Exception as e:
            raise TranscriptAnalyzerError(f"Cannot read file: {e}")
        logger.info(
            f"Transcript validation passed: {transcript_path} ({file_size_mb:.1f}MB)"
        )


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """Decorator for retrying functions on failure."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            config = (args[0] if args else {}).get("config", Config())
            for attempt in range(config.max_retries):
                try:
                    return func(*args, **kwargs)
                except TranscriptValidationError as e:
                    # Do not retry on validation errors
                    logger.error(f"Validation failed for {func.__name__}: {e}")
                    raise
                except exceptions as e:
                    if attempt == config.max_retries - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {config.max_retries} attempts: {e}"
                        )
                        raise
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying..."
                        )
                        time.sleep(config.retry_delay * (2**attempt))
            return None  # Should not be reached

        return wrapper

    return decorator


def update_progress(
    state: TranscriptState, step: str, progress: float
) -> TranscriptState:
    """Update processing progress and step information."""
    state["current_step"] = step
    state["progress"] = progress
    if state["config"].enable_progress_bar:
        progress_bar = "█" * int(progress * 40) + "░" * (40 - int(progress * 40))
        print(f"[{progress_bar}] {progress:.1%} - {step}", end="", flush=True)
    return state


def clean_llm_output(text: str) -> str:
    """Removes common LLM end-of-turn tokens and excessive whitespace."""
    # Specific tokens observed from some models
    text = text.replace("</end_of_turn>", "")
    text = text.replace("<|eot_id|>", "")
    text = text.replace("<|endoftext|>", "")
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    # Remove leading/trailing whitespace and multiple newlines
    text = text.strip()
    text = os.linesep.join([s for s in text.splitlines() if s])  # Remove empty lines
    return text


def measure_time(func):
    """Decorator to measure function execution time."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if args and isinstance(args[0], dict):
            (args[0].setdefault("processing_time", {}))[func.__name__] = (
                end_time - start_time
            )
        return result

    return wrapper


# --- Generic Processing Logic ---


def _deduplicate_list(input_list: List[str]) -> List[str]:
    """Deduplicates a list of strings, preserving order."""
    return list(dict.fromkeys(input_list))


class IterativeRefiner:
    """A generic class to handle iterative refinement over document chunks."""

    def __init__(
        self,
        llm: Runnable,
        initial_prompt: PromptTemplate,
        refine_prompt: PromptTemplate,
        output_parser: BaseOutputParser,
        pydantic_model: Optional[BaseModel] = None,
    ):
        self.output_parser = output_parser
        self.pydantic_model = pydantic_model

        if pydantic_model:
            self.initial_chain = (
                initial_prompt
                | llm
                | PydanticOutputParser(pydantic_object=pydantic_model)
            )
            self.refine_chain = (
                refine_prompt
                | llm
                | PydanticOutputParser(pydantic_object=pydantic_model)
            )
        else:
            self.initial_chain = initial_prompt | llm | output_parser
            self.refine_chain = refine_prompt | llm | output_parser

    def process(self, docs: List[Document], context: Dict[str, Any] = None) -> Any:
        if not docs:
            return None

        initial_context = {"text": docs[0].page_content, **(context or {})}
        if self.pydantic_model:
            initial_context["format_instructions"] = PydanticOutputParser(
                pydantic_object=self.pydantic_model
            ).get_format_instructions()

        result = self.initial_chain.invoke(initial_context)

        for doc in docs[1:]:
            refine_context = {
                "text": doc.page_content,
                "existing_answer": (
                    json.dumps(result.model_dump()) if self.pydantic_model else result
                ),
                **(context or {}),
            }
            if self.pydantic_model:
                refine_context["format_instructions"] = PydanticOutputParser(
                    pydantic_object=self.pydantic_model
                ).get_format_instructions()

            try:
                result = self.refine_chain.invoke(refine_context)
            except OutputParserException as e:
                logger.warning(
                    f"Output parsing failed, continuing with previous result. Error: {e}"
                )
                continue
        return clean_llm_output(result) if isinstance(result, str) else result


# --- Graph Nodes: The Core Logic of the Analyzer ---


@measure_time
@retry_on_failure()
def load_and_split_transcript(state: TranscriptState) -> TranscriptState:
    path = state["transcript_path"]
    config = state["config"]
    update_progress(state, "Validating transcript file", 0.0)
    DocumentValidator.validate_transcript(path, config)
    update_progress(state, "Loading transcript", 0.02)
    with open(path, "r", encoding="utf-8") as f:
        full_text = f.read()

    token_count = 0
    if tiktoken:
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            token_count = len(encoding.encode(full_text))
            logger.info(f"Transcript loaded. Token count: {token_count}")
        except Exception as e:
            logger.warning(f"Could not count tokens: {e}")

    docs = None  # Initialize docs to None
    if config.iterative_analysis:  # Only split if using iterative analysis
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        docs = text_splitter.create_documents([full_text])
        logger.info(f"Split transcript into {len(docs)} chunks.")
    else:
        logger.info("Processing full transcript, skipping chunking.")

    state.update(docs=docs, full_transcript=full_text, token_count=token_count)
    return state


def _generic_extraction(
    state: TranscriptState,
    task_name: str,
    progress_msg: str,
    progress_val: float,
    pydantic_model: type[BaseModel],
):
    config, docs, prompts, full_transcript = (
        state["config"],
        state["docs"],
        state["prompts"],
        state["full_transcript"],
    )
    if not docs and not full_transcript:
        return None

    update_progress(state, progress_msg, progress_val)
    llm = LLMProvider.create_llm(config, json_mode=True)
    output_parser = JsonOutputParser()

    if config.iterative_analysis:
        initial_prompt = prompts.get_prompt(task_name, "initial_prompt")
        refine_prompt = prompts.get_prompt(task_name, "refine_prompt")
        refiner = IterativeRefiner(
            llm,
            initial_prompt,
            refine_prompt,
            output_parser,
            pydantic_model=pydantic_model,
        )
        parsed_output = refiner.process(docs)
    else:
        prompt = prompts.get_prompt(task_name, "initial_prompt")
        context = {"text": full_transcript}
        if pydantic_model:
            context["format_instructions"] = PydanticOutputParser(
                pydantic_object=pydantic_model
            ).get_format_instructions()

        chain = prompt | llm | PydanticOutputParser(pydantic_object=pydantic_model)
        raw_output = chain.invoke(context)
        parsed_output = (
            clean_llm_output(raw_output) if isinstance(raw_output, str) else raw_output
        )

    return parsed_output


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def identify_topics(state: TranscriptState) -> TranscriptState:
    parsed_output = _generic_extraction(
        state, "identify_topics", "Identifying topics", 0.05, TopicsOutput
    )
    if not parsed_output:
        return state

    state.setdefault("analysis", {})["topics"] = parsed_output.topics
    state["analysis"]["original_topics"] = parsed_output.topics
    logger.info(
        f"Identified {len(state['analysis']['topics'])} initial topics: {parsed_output.topics}"
    )
    return state


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def merge_overlapping_topics(state: TranscriptState) -> TranscriptState:
    """Identifies and merges overlapping topics using an LLM call."""
    config, prompts = state["config"], state["prompts"]
    topics = state.get("analysis", {}).get("topics", [])
    if not topics or len(topics) < 2:
        logger.info("Skipping topic merging due to insufficient number of topics.")
        # To maintain a consistent data structure, format the topics as if they were merged
        state["analysis"]["merged_topics"] = {topic: [topic] for topic in topics}
        return state

    update_progress(state, "Merging overlapping topics", 0.20)
    llm = LLMProvider.create_llm(config, json_mode=True)
    pydantic_model = MergedTopicsOutput

    prompt = prompts.get_prompt("merge_topics", "initial_prompt")
    context = {
        "topics": "\n".join(f"- {topic}" for topic in topics),
        "format_instructions": PydanticOutputParser(
            pydantic_object=pydantic_model
        ).get_format_instructions(),
    }

    chain = prompt | llm | JsonOutputParser()  # Use a less strict parser first
    try:
        raw_output = chain.invoke(context)

        # --- Data Sanitization Step ---
        # Ensure all values in the dictionary are lists of strings
        if isinstance(raw_output, dict) and "merged_topics" in raw_output:
            cleaned_topics = {}
            for key, value in raw_output["merged_topics"].items():
                if isinstance(value, list):
                    cleaned_topics[key] = [
                        str(v) for v in value
                    ]  # Ensure all items in list are strings
                elif isinstance(value, str):
                    cleaned_topics[key] = [
                        value
                    ]  # Convert string to list of one string
                else:
                    # If the value is neither a list nor a string, skip it or handle as an error
                    logger.warning(
                        f"Skipping malformed topic '{key}' in merge_topics output: value is not a list or string."
                    )
            raw_output["merged_topics"] = cleaned_topics

        # Now, parse with the strict Pydantic model
        parsed_output = MergedTopicsOutput.model_validate(raw_output)

        # Replace the original topics with the merged ones
        state["analysis"]["topics"] = list(parsed_output.merged_topics.keys())
        state["analysis"]["merged_topics"] = parsed_output.merged_topics
        logger.info(
            f"Merged {len(topics)} topics into {len(state['analysis']['topics'])} unique topics."
        )
        logger.debug(f"Topic merge mapping: {parsed_output.merged_topics}")
    except Exception as e:
        logger.error(f"Failed to merge topics: {e}. Proceeding with original topics.")
        state.setdefault("warnings", []).append(f"Topic merging failed: {e}")
        # Ensure the structure is consistent even on failure
        state["analysis"]["merged_topics"] = {topic: [topic] for topic in topics}

    return state


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def extract_topic_texts(state: TranscriptState) -> TranscriptState:
    """Extracts relevant text for each topic from the full transcript."""
    config, prompts, full_transcript = (
        state["config"],
        state["prompts"],
        state["full_transcript"],
    )
    topics = state.get("analysis", {}).get("topics", [])
    if not topics or not full_transcript:
        logger.info("Skipping topic text extraction due to no topics or transcript.")
        return state

    update_progress(state, "Extracting topic texts", 0.22)
    llm = LLMProvider.create_llm(config)
    output_parser = StrOutputParser()

    topic_texts = {}
    total_topics = len(topics)
    for i, topic in enumerate(topics):
        logger.info(f"Extracting text for topic: '{topic}'")
        progress = 0.22 + (0.23 * ((i + 1) / total_topics))
        update_progress(
            state,
            f"Extracting text for topic {i+1}/{total_topics}: '{topic}'",
            progress,
        )

        prompt = prompts.get_prompt("extract_topic_text", "initial_prompt")
        context = {"text": full_transcript, "topic": topic}
        chain = prompt | llm | output_parser
        extracted_text = chain.invoke(context)
        topic_texts[topic] = clean_llm_output(extracted_text)

    state.setdefault("analysis", {})["topic_texts"] = topic_texts
    logger.info("Finished extracting topic texts.")
    return state


@measure_time
@retry_on_failure()
def summarize_topics(state: TranscriptState) -> TranscriptState:
    config, docs, prompts, full_transcript = (
        state["config"],
        state["docs"],
        state["prompts"],
        state["full_transcript"],
    )
    topics = state.get("analysis", {}).get("topics", [])
    topic_texts = state.get("analysis", {}).get("topic_texts")  # Get extracted texts
    if not topics or (not docs and not full_transcript):
        return state

    update_progress(state, "Summarizing topics", 0.25)
    llm = LLMProvider.create_llm(config)
    output_parser = StrOutputParser()

    summaries = {}
    total_topics = len(topics)

    # Determine the text source for summarization
    text_source = (
        topic_texts
        if config.separate_topic_summarization and topic_texts
        else full_transcript
    )

    if config.iterative_analysis:
        initial_prompt = prompts.get_prompt("summarize_topics", "initial_prompt")
        refine_prompt = prompts.get_prompt("summarize_topics", "refine_prompt")
        refiner = IterativeRefiner(llm, initial_prompt, refine_prompt, output_parser)

        for i, topic in enumerate(topics):
            logger.info(f"Summarizing topic: '{topic}'")
            progress = 0.25 + (0.40 * ((i + 1) / total_topics))
            update_progress(
                state, f"Summarizing topic {i+1}/{total_topics}: '{topic}'", progress
            )

            summary = refiner.process(docs, context={"topic": topic})
            summaries[topic] = summary
    else:
        for i, topic in enumerate(topics):
            logger.info(f"Summarizing topic: '{topic}'")
            progress = 0.25 + (0.40 * ((i + 1) / total_topics))
            update_progress(
                state, f"Summarizing topic {i+1}/{total_topics}: '{topic}'", progress
            )

            # Use topic-specific text if available, otherwise use the full transcript
            text_to_summarize = (
                text_source.get(topic) if isinstance(text_source, dict) else text_source
            )
            if not text_to_summarize:
                logger.warning(
                    f"No text found for topic '{topic}', skipping summarization."
                )
                summaries[topic] = "No relevant text found for this topic."
                continue

            prompt = prompts.get_prompt("summarize_topics", "initial_prompt")
            context = {"text": text_to_summarize, "topic": topic}
            chain = prompt | llm | output_parser
            raw_summary = chain.invoke(context)
            summary = (
                clean_llm_output(raw_summary)
                if isinstance(raw_summary, str)
                else raw_summary
            )
            summaries[topic] = summary

    state["analysis"]["topic_summaries"] = summaries
    return state


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def extract_key_insights(state: TranscriptState) -> TranscriptState:
    parsed_output = _generic_extraction(
        state, "extract_key_insights", "Extracting key insights", 0.65, InsightsOutput
    )
    if not parsed_output:
        return state

    state.setdefault("analysis", {})["key_insights"] = _deduplicate_list(
        parsed_output.insights
    )
    logger.info(f"Extracted {len(state['analysis']['key_insights'])} key insights.")
    return state


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def extract_decisions(state: TranscriptState) -> TranscriptState:
    parsed_output = _generic_extraction(
        state, "extract_decisions", "Extracting decisions", 0.80, DecisionsOutput
    )
    if not parsed_output:
        return state

    state.setdefault("analysis", {})["decisions_discussed"] = _deduplicate_list(
        parsed_output.decisions
    )
    logger.info(f"Extracted {len(state['analysis']['decisions_discussed'])} decisions.")
    return state


@measure_time
@retry_on_failure(exceptions=(Exception, OutputParserException))
def extract_action_items(state: TranscriptState) -> TranscriptState:
    parsed_output = _generic_extraction(
        state,
        "extract_action_items",
        "Extracting action items",
        0.85,
        ActionItemsOutput,
    )
    if not parsed_output:
        return state

    state.setdefault("analysis", {})["action_items"] = _deduplicate_list(
        parsed_output.action_items
    )
    logger.info(f"Extracted {len(state['analysis']['action_items'])} action items.")
    return state


@measure_time
@retry_on_failure()
def generate_executive_summary(state: TranscriptState) -> TranscriptState:
    """Generate a final executive summary by refining topic summaries."""
    config = state["config"]
    prompts = state["prompts"]
    topic_summaries = state.get("analysis", {}).get("topic_summaries", {})
    if not topic_summaries:
        state.setdefault("warnings", []).append(
            "No topic summaries available to generate executive overview."
        )
        return state

    # Treat topic summaries as the documents to be refined
    if config.iterative_analysis:
        # Original iterative refinement for executive summary
        summary_docs = [
            Document(page_content=summary) for summary in topic_summaries.values()
        ]
        initial_prompt = prompts.get_prompt(
            "generate_executive_summary", "initial_prompt"
        )
        refine_prompt = prompts.get_prompt(
            "generate_executive_summary", "refine_prompt"
        )
        update_progress(state, "Generating final executive summary", 0.95)
        refiner = IterativeRefiner(
            LLMProvider.create_llm(config),
            initial_prompt,
            refine_prompt,
            StrOutputParser(),
        )
        summary = refiner.process(summary_docs)
    else:
        # If processing full transcript, the executive summary is generated directly from the full transcript
        # as the previous steps would have already processed the full transcript.
        # The prompt for executive summary should be designed to handle the full transcript.
        initial_prompt = prompts.get_prompt(
            "generate_executive_summary", "initial_prompt"
        )
        llm = LLMProvider.create_llm(config)
        output_parser = StrOutputParser()
        chain = initial_prompt | llm | output_parser
        raw_summary = chain.invoke({"text": state["full_transcript"]})
        summary = (
            clean_llm_output(raw_summary)
            if isinstance(raw_summary, str)
            else raw_summary
        )

    state["analysis"]["executive_overview"] = summary
    logger.info("Successfully generated final executive summary.")
    return state


# --- Output and Main Execution ---


def save_results(state: TranscriptState) -> None:
    config = state["config"]
    if not config.output_file:
        return
    try:
        output_path = Path(config.output_file)
        analysis_data = state.get("analysis", {})
        output_content = {
            "analysis": analysis_data,
            "processing_time": state.get("processing_time", {}),
            "token_count": state.get("token_count", 0),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }
        if config.output_format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_content, f, indent=2)
        elif config.output_format == "yaml":
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(output_content, f, default_flow_style=False, sort_keys=False)
        elif config.output_format in ["csv", "tsv"]:
            delimiter = "," if config.output_format == "csv" else "\t"
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=delimiter)
                writer.writerow(["category", "item"])
                for insight in analysis_data.get("key_insights", []):
                    writer.writerow(["insight", insight])
                for decision in analysis_data.get("decisions_discussed", []):
                    writer.writerow(["decision", decision])
                for action_item in analysis_data.get("action_items", []):
                    writer.writerow(["action_item", action_item])
        elif config.output_format == "markdown":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(
                    f"# Meeting Transcript Analysis\n\n**File:** {state['transcript_path']}\n\n"
                )
                if "executive_overview" in analysis_data:
                    f.write(
                        f"## 1. Executive Overview\n\n{analysis_data['executive_overview']}\n\n"
                    )
                f.write(f"## 2. Topics Discussed\n\n")

                original_topics = analysis_data.get("original_topics", [])
                merged_topics_mapping = analysis_data.get("merged_topics", {})
                final_topics = analysis_data.get("topics", [])

                if original_topics:
                    f.write(f"### Original Identified Topics\n\n")
                    for topic in original_topics:
                        f.write(f"- {topic}\n")
                    f.write("\n")

                if merged_topics_mapping and config.merge_topics:
                    f.write(f"### Merged Topics\n\n")
                    for merged, original in merged_topics_mapping.items():
                        f.write(f"- **{merged}** (from: {', '.join(original)})\n")
                    f.write("\n")

                f.write(f"### Topic Summaries\n\n")
                topic_summaries = analysis_data.get("topic_summaries", {})
                if not final_topics:
                    f.write("No topics were identified.\n\n")
                else:
                    for topic in final_topics:
                        f.write(
                            f"<details>\n<summary><h4>{topic}</h4></summary>\n\n{topic_summaries.get(topic, 'No summary generated.')}\n\n</details>\n\n"
                        )
                key_insights = analysis_data.get("key_insights", [])
                f.write(f"## 3. Overall Key Insights from the Expert\n\n")
                if key_insights:
                    for insight in key_insights:
                        f.write(f"- {insight}\n")
                else:
                    f.write(
                        "All key insights are integrated within the topic-specific summaries above.\n"
                    )
                f.write("\n")
                decisions = analysis_data.get("decisions_discussed", [])
                f.write(f"## 4. Decisions Advised or Discussed\n\n")
                if decisions:
                    for decision in decisions:
                        f.write(f"- {decision}\n")
                else:
                    f.write(
                        "No explicit decisions related to the expert's advice were discussed.\n"
                    )
                if state.get("errors"):
                    f.write(f"\n## Errors\n\n")
                    for error in state["errors"]:
                        f.write(f"- {error}\n")

                if state.get("warnings"):
                    f.write(f"\n## Warnings\n\n")
                    for warning in state["warnings"]:
                        f.write(f"- {warning}\n")

                if state.get("processing_time"):
                    f.write(f"\n## Performance Metrics\n\n")
                    total_time = sum(state["processing_time"].values())
                    f.write(f"- **Total processing time:** {total_time:.2f} seconds\n")
                    f.write(f"- **Token count:** {state.get('token_count', 'N/A')}\n")
                    f.write("\n### Step-by-step processing time:\n\n")
                    for step, time_taken in state["processing_time"].items():
                        f.write(f"- **{step}:** {time_taken:.2f}s\n")
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def print_results(state: TranscriptState) -> None:
    analysis = state.get("analysis", {})
    print("\n" + "=" * 80 + "\nMEETING TRANSCRIPT ANALYSIS RESULTS\n" + "=" * 80)
    print(
        f"Analyzed transcript: {state['transcript_path']} ({state.get('token_count', 'N/A')} tokens)"
    )
    if "executive_overview" in analysis:
        print(f"\n## 1. Executive Overview\n\n{analysis['executive_overview']}")

    print("\n## 2. Topics Discussed\n")
    original_topics = analysis.get("original_topics", [])
    merged_topics_mapping = analysis.get("merged_topics", {})
    final_topics = analysis.get("topics", [])
    config = state["config"]

    if original_topics:
        print("\n### Original Identified Topics\n")
        for topic in original_topics:
            print(f"- {topic}")

    if merged_topics_mapping and config.merge_topics:
        print("\n### Merged Topics\n")
        for merged, original in merged_topics_mapping.items():
            print(f"- {merged} (from: {', '.join(original)})")

    print("\n### Topic Summaries\n")
    topic_summaries = analysis.get("topic_summaries", {})
    if not final_topics:
        print("No topics were identified.")
    else:
        for topic in final_topics:
            print(
                f"\n#### {topic}\n\n{topic_summaries.get(topic, 'No summary generated.')}"
            )
    print("\n\n## 3. Overall Key Insights from the Expert\n")
    key_insights = analysis.get("key_insights", [])
    if key_insights:
        for insight in key_insights:
            print(f"- {insight}")
    else:
        print(
            "All key insights are integrated within the topic-specific summaries above."
        )
    print("\n## 4. Decisions Advised or Discussed\n")
    decisions = analysis.get("decisions_discussed", [])
    if decisions:
        for decision in decisions:
            print(f"- {decision}")
    else:
        print("No explicit decisions related to the expert's advice were discussed.")
    if state.get("processing_time"):
        print("\n" + "-" * 40 + "\n⏱️  PERFORMANCE METRICS")
        total_time = sum(state["processing_time"].values())
        print(f"Total processing time: {total_time:.2f} seconds")
        for step, time_taken in state["processing_time"].items():
            print(f"  - {step}: {time_taken:.2f}s")
    if state.get("errors"):
        print("\n" + "-" * 40 + "\n❌ ERRORS")
        for error in state["errors"]:
            print(f"  - {error}")
    if state.get("warnings"):
        print("\n" + "-" * 40 + "\n⚠️  WARNINGS")
        for warning in state["warnings"]:
            print(f"  - {warning}")
    print("\n" + "=" * 80)


def build_analyzer_graph(config: Config):
    """Build the LangGraph workflow for transcript analysis."""
    workflow = StateGraph(TranscriptState)
    workflow.add_node("load_and_split", load_and_split_transcript)
    workflow.add_node("identify_topics", identify_topics)
    workflow.add_node("merge_overlapping_topics", merge_overlapping_topics)
    workflow.add_node("extract_topic_texts", extract_topic_texts)
    workflow.add_node("summarize_topics", summarize_topics)
    workflow.add_node("extract_key_insights", extract_key_insights)
    workflow.add_node("extract_decisions", extract_decisions)
    workflow.add_node("extract_action_items", extract_action_items)
    workflow.add_node("generate_executive_summary", generate_executive_summary)

    workflow.set_entry_point("load_and_split")
    workflow.add_edge("load_and_split", "identify_topics")

    previous_node = "identify_topics"
    if config.merge_topics:
        workflow.add_edge("identify_topics", "merge_overlapping_topics")
        previous_node = "merge_overlapping_topics"

    if config.separate_topic_summarization:
        workflow.add_edge(previous_node, "extract_topic_texts")
        workflow.add_edge("extract_topic_texts", "summarize_topics")
    else:
        workflow.add_edge(previous_node, "summarize_topics")

    workflow.add_edge("summarize_topics", "extract_key_insights")
    workflow.add_edge("extract_key_insights", "extract_decisions")

    previous_node = "extract_decisions"
    if config.extract_action_items:
        workflow.add_edge("extract_decisions", "extract_action_items")
        previous_node = "extract_action_items"

    workflow.add_edge(previous_node, "generate_executive_summary")
    workflow.add_edge("generate_executive_summary", END)

    return workflow.compile()


def main():
    """Main function with argument parsing and workflow execution."""
    load_dotenv()  # Load environment variables from .env file
    parser = argparse.ArgumentParser(
        description="Analyze a meeting transcript with LangGraph"
    )

    # Input/Output Options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "transcript_path", nargs="?", help="Path to transcript file (.txt)"
    )
    io_group.add_argument("--output", help="Output file path")
    io_group.add_argument(
        "--format",
        choices=["console", "json", "markdown", "yaml", "csv", "tsv"],
        help="Output format",
    )

    # Configuration Options
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument("--config", help="Path to YAML configuration file")
    config_group.add_argument(
        "--interview",
        action="store_true",
        help="Use interview-focused prompts instead of meeting prompts.",
    )
    config_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    # LLM Model Options
    llm_group = parser.add_argument_group("LLM Model Options")
    llm_group.add_argument(
        "--model",
        help="Model to use, in 'provider/model_name' format (e.g., openai/gpt-4o). Defaults to 'ollama' if no provider is specified.",
    )
    llm_group.add_argument(
        "--ollama-options",
        type=json.loads,
        help='JSON string of options for Ollama (e.g., "{"num_ctx": 8192}")',
    )
    llm_group.add_argument(
        "--openai-api-base-url",
        help="Alternative base URL for OpenAI API (e.g., for local LLMs)",
    )
    config_group.add_argument(
        "--iterative-analysis",
        action="store_true",
        help="Process the transcript iteratively, instead of all at once.",
    )
    config_group.add_argument(
        "--merge-topics", action="store_true", help="Enable the topic merging step."
    )
    config_group.add_argument(
        "--separate-topic-summarization",
        action="store_true",
        help="Summarize topics from extracted relevant text instead of the full transcript.",
    )
    config_group.add_argument(
        "--extract-action-items",
        action="store_true",
        help="Extract action items from the transcript.",
    )

    args = parser.parse_args()

    try:
        # --- Configuration Loading Hierarchy ---
        # 1. Load from YAML file if specified, otherwise use defaults
        config = (
            Config.from_yaml(args.config)
            if args.config and Path(args.config).exists()
            else Config()
        )

        # 2. Override with Environment Variables
        if os.getenv("MODEL_PROVIDER"):
            config.model_provider = os.getenv("MODEL_PROVIDER")
        if os.getenv("MODEL_NAME"):
            config.model_name = os.getenv("MODEL_NAME")
        if os.getenv("OPENAI_API_BASE_URL"):
            config.openai_api_base_url = os.getenv("OPENAI_API_BASE_URL")
        if os.getenv("OLLAMA_MODEL_OPTIONS"):
            config.ollama_model_options = json.loads(os.getenv("OLLAMA_MODEL_OPTIONS"))
        if os.getenv("OUTPUT_FORMAT"):
            config.output_format = os.getenv("OUTPUT_FORMAT")
        if os.getenv("OUTPUT_FILE"):
            config.output_file = os.getenv("OUTPUT_FILE")
        if os.getenv("LOG_LEVEL"):
            global log_level
            log_level = getattr(logging, os.getenv("LOG_LEVEL").upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                force=True,
            )

        # 3. Override with Command-Line Arguments
        if args.model:
            if "/" in args.model:
                provider, model_name = args.model.split("/", 1)
                config.model_provider = provider
                config.model_name = model_name
            else:
                config.model_name = args.model

        if args.output:
            config.output_file = args.output
        if args.format:
            config.output_format = args.format
        if args.interview:
            config.interview_mode = True
        if args.merge_topics:
            config.merge_topics = True
        if args.iterative_analysis:
            config.iterative_analysis = True
        if args.separate_topic_summarization:
            config.separate_topic_summarization = True
        if args.extract_action_items:
            config.extract_action_items = True
        if args.openai_api_base_url:
            config.openai_api_base_url = args.openai_api_base_url
        if args.log_level:
            log_level = getattr(logging, args.log_level.upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format="%(asctime)s - %(levelname)s - %(message)s",
                force=True,
            )

        prompts_file = (
            "prompts_interview.yaml"
            if config.interview_mode
            else "prompts_meeting.yaml"
        )
        with resources.open_text("meeting_summarizer", prompts_file) as f:
            prompts_content = f.read()
        prompt_manager = PromptManager(prompts_content)

        if not args.transcript_path:
            parser.error("Transcript path is required")

        initial_state: TranscriptState = {
            "transcript_path": args.transcript_path,
            "config": config,
            "full_transcript": None,
            "docs": None,
            "current_step": "initialized",
            "progress": 0.0,
            "token_count": 0,
            "analysis": {},
            "errors": [],
            "warnings": [],
            "processing_time": {},
            "prompts": prompt_manager,
        }

        logger.info("Starting meeting transcript analysis workflow...")
        app = build_analyzer_graph(config)
        final_state = app.invoke(initial_state)

        print()  # New line after progress bar
        if config.output_format == "console":
            print_results(final_state)

        save_results(final_state)

        if final_state.get("errors"):
            logger.error("Workflow completed with errors.")
            sys.exit(1)
        else:
            logger.info("Workflow completed successfully.")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"A fatal error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
