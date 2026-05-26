"""
LLM-based failure analysis agent for carbon intensity forecasting.

This module implements an agentic workflow that:
    1. Receives model failure data and evaluation metrics
    2. Uses tools to investigate feature/error relationships
    3. Generates evidence-based hypotheses about failure causes
    4. Recommends specific improvements
    5. Designs validation experiments
"""

import json
import logging
import google.generativeai as genai
from carbon_forecast.agent.tools import AnalysisTools
from carbon_forecast.agent.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class FailureAnalyst:
    """
    Agentic failure analysis system.

    Uses an LLM with tool access to investigate why a forecasting
    model fails under specific conditions. The agent can call
    analytical tools to examine correlations, compare failure
    regimes, and deep-dive into specific episodes.

    Architecture:
        - Tools (AnalysisTools): functions the agent can call
        - LLM (Gemini): reasons about data and decides which tools to use
        - Automatic function calling: Gemini executes the tool loop internally
    """

    def __init__(self, tools: AnalysisTools, api_key: str):
        self.tools = tools
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )

    def _build_tool_functions(self) -> list:
        """
        Wrap AnalysisTools methods as standalone functions
        that Gemini can introspect and call directly.

        Logging is added inside each wrapper so we can track
        which tools the agent decides to use.
        """

        tools = self.tools

        def get_error_correlation(feature_name: str) -> dict:
            """Check how a feature correlates with prediction errors.
            Returns correlation coefficient and mean error in low vs high quartiles."""
            logger.info(f"  Tool call: get_error_correlation({feature_name})")
            return tools.get_error_correlation(feature_name)

        def compare_failure_vs_success(feature_name: str) -> dict:
            """Compare feature distribution during model failures vs normal operation.
            Returns mean and std for both regimes plus the difference."""
            logger.info(f"  Tool call: compare_failure_vs_success({feature_name})")
            return tools.compare_failure_vs_success(feature_name)

        def get_episode_deep_dive(start: str, end: str) -> dict:
            """Get detailed statistics for a specific time window.
            Returns duration, errors, and feature summaries for the period."""
            logger.info(f"  Tool call: get_episode_deep_dive({start}, {end})")
            return tools.get_episode_deep_dive(start, end)

        def get_feature_at_extremes(feature_name: str, percentile: float) -> dict:
            """Get model performance when a feature is at extreme values.
            Returns threshold, mean error, and sample count at the extreme."""
            logger.info(f"  Tool call: get_feature_at_extremes({feature_name}, {percentile})")
            return tools.get_feature_at_extremes(feature_name, percentile)

        return [
            get_error_correlation,
            compare_failure_vs_success,
            get_episode_deep_dive,
            get_feature_at_extremes,
        ]

    def analyse(
        self,
        failure_report: list[dict],
        stratified_report: list[dict],
        feature_importances: dict,
        available_features: list[str],
    ) -> dict:
        """
        Run the full agent analysis loop.

        Args:
            failure_report:
                Top failure episodes from FailureDetector.
            stratified_report:
                Stratified evaluation results from StratifiedEvaluator.
            feature_importances:
                Model feature importance scores.
            available_features:
                List of column names available for investigation.

        Returns:
            Structured analysis dict with hypotheses, recommendations,
            and proposed validation experiments.
        """

        logger.info("Starting agent analysis...")

        tool_functions = self._build_tool_functions()

        user_message = (
            "Analyse the following model failure data and investigate root causes.\n\n"
            f"## Top Failure Episodes\n{json.dumps(failure_report, indent=2, default=str)}\n\n"
            f"## Stratified Evaluation\n{json.dumps(stratified_report, indent=2, default=str)}\n\n"
            f"## Feature Importances\n{json.dumps(feature_importances, indent=2)}\n\n"
            f"## Available Features\n{json.dumps(available_features)}\n\n"
            "Use your tools to investigate the failure patterns. "
            "Return your final analysis as valid JSON matching the schema in your instructions."
        )

        # Gemini handles the tool loop automatically:
        # it calls functions, reads results, and continues
        # until it has enough information to respond
        chat = self.model.start_chat(enable_automatic_function_calling=True)

        logger.info("Sending data to agent...")
        response = chat.send_message(user_message, tools=tool_functions)

        logger.info("Agent analysis complete.")

        # Parse JSON from response
        raw_text = response.text

        # Clean markdown fences if present
        clean_text = raw_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        if clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        try:
            analysis = json.loads(clean_text)
            logger.info(f"Parsed {len(analysis.get('hypotheses', []))} hypotheses")
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from agent response")
            analysis = {"raw_response": raw_text}

        return analysis

    def report(
        self,
        failure_report: list[dict],
        stratified_report: list[dict],
        feature_importances: dict,
        available_features: list[str],
    ) -> dict:
        """
        Run analysis and log a summary of findings.
        """

        analysis = self.analyse(
            failure_report=failure_report,
            stratified_report=stratified_report,
            feature_importances=feature_importances,
            available_features=available_features,
        )

        logger.info(f"\n{'='*60}")
        logger.info("AGENT FAILURE ANALYSIS REPORT")
        logger.info(f"{'='*60}")

        if "hypotheses" in analysis:
            for h in analysis["hypotheses"]:
                rank = h.get("rank", "?")
                hypothesis = h.get("hypothesis", "")
                confidence = h.get("confidence", "")
                action = h.get("recommended_action", "")

                logger.info(f"\n[#{rank}] {hypothesis}")
                logger.info(f"  Confidence: {confidence}")
                logger.info(f"  Action: {action}")

        if "priority_actions" in analysis:
            logger.info(f"\nPriority Actions:")
            for action in analysis["priority_actions"]:
                logger.info(f"  - {action}")

        logger.info(f"\n{'='*60}")

        return analysis