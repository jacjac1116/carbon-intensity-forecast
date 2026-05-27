from carbon_forecast.evaluation.failure import FailureDetector
from carbon_forecast.evaluation.stratified import StratifiedEvaluator
from carbon_forecast.agent.tools import AnalysisTools
import anthropic
import json
from carbon_forecast.agent.prompt import SYSTEM_PROMPT
from carbon_forecast.agent.tools_schema import TOOLS
import logging
import re

logger = logging.getLogger(__name__)

class FailureAnalyst:
    """
    LLM-based agent that analyses model failure patterns,
    generates hypotheses, and recommends improvements.

    Workflow:
        1. Receives failure episodees + stratified metrics + feature importances
        2. Uses tools to investigate the data
        3. Generates ranked hypothesis about failure causes
        4. Proposes specific corrective actions
        5. Designs targetted test scenarios to validate hypotheses
    """

    def __init__(self, tools: AnalysisTools, api_key: str):
        self.tools = tools
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyse (self, failure_report: dict, stratified_report: dict, feature_importances: dict, available_features: list[str], quantile_metrics: dict) -> dict:
        """
        Run the full analysis loop
        """

        user_message = f"""
        Failure episodes: {json.dumps(failure_report, default=str)}
        Stratified evaluation: {json.dumps(stratified_report, default=str)}
        Feature importances: {json.dumps(feature_importances, default=str)}
        Available features for investigation: {available_features}
        f"## Quantile Model Metrics\n{json.dumps(quantile_metrics, indent=2, default=str)}\n\n"
        """

        # Map tool names to actual functions
        tool_functions = {
            'get_error_correlation': self.tools.get_error_correlation,
            'compare_failure_vs_success': self.tools.compare_failure_vs_success,
            'get_episode_deep_dive': self.tools.get_episode_deep_dive,
            'get_features_at_extreme': self.tools.get_feature_at_extremes
        }

        messages = [{'role': 'user',
                     'content': user_message}]
        
        while True:
            response = self.client.messages.create(
                model = 'claude-haiku-4-5-20251001',
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                max_tokens=8192,
            )

            # Case 1: Claude wants to use a tool
            if response.stop_reason == 'tool_use':

                # Add Claude's full response (including its reasoning) as assistant message
                messages.append({'role': 'assistant',
                                 'content': response.content})
                
                # Process each tool call in the response
                tool_results = []

                for block in response.content:
                    if block.type == 'tool_use':

                        logger.info(f"Agent calling: {block.name}({block.input})")

                        # Execute the actual function
                        try:
                            result = tool_functions[block.name](**block.input)
                        except Exception as e:
                            result = {'error': str(e)}

                        
                        # Package the result for Claude
                        tool_results.append({
                            'type': 'tool_result',
                            'tool_use_id': block.id,
                            'content': json.dumps(result),
                        })
                
                # Send all tool results back to Claude
                messages.append({'role': 'user',
                                 'content': tool_results})
            
            # Cse 2: Claude is done
            else:
                # Extract the final response
                final_text = ''
                for block in response.content:
                    if hasattr(block, 'text'):
                        final_text += block.text
                
                # Parse JSON from Claude's response
                json_match = re.search(r'```json\s*(.*?)\s*```', final_text, re.DOTALL)
                if json_match:
                    clean_text = json_match.group(1).strip()
                else:
                    clean_text = final_text.strip()
                
                try:
                    analysis = json.loads(clean_text)
                except json.JSONDecodeError:
                    logger.warning("Could not parse JSON, returning raw text")
                    analysis = {"raw_response": final_text}

                return analysis


