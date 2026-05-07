import streamlit as st
import os
import sys
from pathlib import Path
from importlib import resources
import logging
from typing import Dict, Any

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent))

from meeting_summarizer.main import (
    Config,
    PromptManager,
    build_analyzer_graph,
    TranscriptState
)

# Setup basic logging for the webapp
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Meeting Summarizer", page_icon="📝")

def run_analysis(transcript_text: str, config: Config, prompt_manager: PromptManager):
    """Executes the analysis workflow."""
    initial_state: TranscriptState = {
        "transcript_path": "uploaded_file.txt", # Placeholder
        "config": config,
        "full_transcript": transcript_text,
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

    app = build_analyzer_graph(config)

    # Progress placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Note: Streamlit doesn't support easy real-time progress updates from within
    # the LangGraph run without complex callbacks or event emitters.
    # For now, we run it and then display results.
    # Real-time progress would require wrapping the graph nodes.

    status_text.text("Running analysis workflow...")
    progress_bar.progress(50) # Arbitrary middle step

    final_state = app.invoke(initial_state)

    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    return final_state

def main():
    st.title("📝 Meeting Summarizer")
    st.markdown("Analyze meeting transcripts using LangGraph and LLMs.")

    # Sidebar: Configuration
    st.sidebar.header("Configuration")

    provider = st.sidebar.selectbox("Model Provider", ["ollama", "openai", "google"], index=0)
    model_name = st.sidebar.text_input("Model Name", value="llama3")

    mode = st.sidebar.radio("Analysis Mode", ["Meeting", "Interview"])

    st.sidebar.subheader("Advanced Options")
    iterative = st.sidebar.checkbox("Iterative Analysis", value=False)
    merge_topics = st.sidebar.checkbox("Merge Overlapping Topics", value=False)
    separate_summary = st.sidebar.checkbox("Separate Topic Summarization", value=False)
    extract_action = st.sidebar.checkbox("Extract Action Items", value=False)

    # Main Area: File Upload
    uploaded_file = st.file_uploader("Upload a meeting transcript (.txt)", type=["txt"])

    if uploaded_file is not None:
        transcript_text = uploaded_file.getvalue().decode("utf-8")
        st.success("File uploaded successfully!")

        if st.button("Generate Summary"):
            config = Config(
                model_provider=provider,
                model_name=model_name,
                interview_mode=(mode == "Interview"),
                iterative_analysis=iterative,
                merge_topics=merge_topics,
                separate_topic_summarization=separate_summary,
                extract_action_items=extract_action,
                enable_progress_bar=False # Disable CLI progress bar
            )

            # Load prompts
            prompts_file = "prompts_interview_v2.yaml" if config.interview_mode else "prompts_meeting_v2.yaml"
            try:
                with resources.open_text("meeting_summarizer", prompts_file) as f:
                    prompts_content = f.read()
                prompt_manager = PromptManager(prompts_content)

                with st.spinner("Analyzing..."):
                    results = run_analysis(transcript_text, config, prompt_manager)

                # Display Results
                analysis = results.get("analysis", {})

                st.header("1. Executive Overview")
                st.write(analysis.get("executive_overview", "No summary generated."))

                st.header("2. Topics Discussed")
                topic_summaries = analysis.get("topic_summaries", {})
                for topic, summary in topic_summaries.items():
                    with st.expander(topic):
                        st.write(summary)

                st.header("3. Key Insights")
                for insight in analysis.get("key_insights", []):
                    st.markdown(f"- {insight}")

                st.header("4. Decisions")
                for decision in analysis.get("decisions_discussed", []):
                    st.markdown(f"- {decision}")

                if config.extract_action_items:
                    st.header("5. Action Items")
                    for item in analysis.get("action_items", []):
                        st.markdown(f"- {item}")

                if results.get("warnings"):
                    st.warning("Warnings: " + ", ".join(results["warnings"]))

                if results.get("errors"):
                    st.error("Errors: " + ", ".join(results["errors"]))

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception(e)

if __name__ == "__main__":
    main()
