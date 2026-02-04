"""
Multi-Handoff Agent: Wrapper for OpenAI agents to coordinate multiple handoffs

This module provides a MultiHandoffAgent class that wraps an OpenAI agent and
enables it to coordinate between multiple handoff agents to achieve an objective.
The agent intelligently decides which handoff agent to call next, analyzes responses,
and continues looping until the objective is complete.
"""

from __future__ import annotations

import asyncio
import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic

# Import from agents package
from agents import Agent, Runner
from agents.agent_output import AgentOutputSchema, AgentOutputSchemaBase

from agents.handoffs import Handoff
from agents.model_settings import ModelSettings
from agents.run import RunConfig
from agents.result import RunResult
from agents.run_context import RunContextWrapper
from agents.items import ItemHelpers, ModelResponse, RunItem
from agents.tool import Tool
from agents.usage import Usage
from agents.logger import logger


# Create a generic type variable for the context
TContext = TypeVar("TContext")


@dataclass
class HandoffResponse:
    """Represents the response from a handoff agent."""

    agent_name: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffDecision:
    """Decision about which agent to call next or whether to finish."""

    next_agent: Optional[str]
    message: str
    objective_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiHandoffAgent(Generic[TContext]):
    """
    A wrapper for OpenAI agents that enables dynamic coordination between multiple handoff agents.

    This agent wrapper intelligently decides which handoff agent to call next based on the
    objective and the responses received so far. It continues looping through handoff agents
    until the objective is complete.
    """

    def __init__(
        self,
        base_agent: Agent[TContext],
        max_iterations: int = 10,
        objective_template: str = "Determine if the following objective is complete based on the conversation history: {objective}. If complete, respond with 'OBJECTIVE COMPLETE: <reason>'. If not complete, respond with 'NEXT STEP: Use the {next_agent} agent to <action>'.",
    ):
        """
        Initialize the MultiHandoffAgent.

        Args:
            base_agent: The base OpenAI agent to wrap.
            max_iterations: Maximum number of iterations/handoffs to perform.
            objective_template: Template for asking the LLM about objective completion.
        """
        self.base_agent = base_agent
        self.max_iterations = max_iterations
        self.objective_template = objective_template
        self.conversation_history: List[Dict[str, str]] = []
        self.handoff_history: List[HandoffResponse] = []

    async def run(
        self,
        initial_prompt: str,
        context: Optional[TContext] = None,
        run_config: Optional[RunConfig] = None,
    ) -> RunResult:
        """
        Run the multi-handoff agent workflow.

        This method executes the core loop:
        1. Process the initial prompt with the base agent
        2. Decide which handoff agent to call next
        3. Process the handoff response
        4. Loop until objective is complete or max iterations reached
        5. Return the final result

        Args:
            initial_prompt: The initial prompt/objective.
            context: Optional context for the agent.
            run_config: Optional configuration for running the agent.

        Returns:
            The final run result.
        """
        logger.info(
            f"Starting multi-handoff agent workflow with objective: {initial_prompt}"
        )

        # Add the initial prompt to conversation history
        self.conversation_history.append(
            {"role": "user", "content": initial_prompt})

        # Initialize context
        if context is None:
            context = None  # Replace with appropriate default context if needed

        # Initialize result with the first run of the base agent
        result = await Runner.run(
            starting_agent=self.base_agent,
            input=initial_prompt,
            context=context,
            run_config=run_config,
        )

        # Add the initial response to conversation history
        initial_response = self._extract_response_text(result)
        self.conversation_history.append(
            {"role": "assistant", "content": initial_response}
        )

        # Start the handoff loop
        iteration = 0
        objective_complete = False

        while not objective_complete and iteration < self.max_iterations:
            # Increment the iteration counter
            iteration += 1
            logger.info(
                f"Starting handoff iteration {iteration}/{self.max_iterations}")

            # Decide which handoff agent to call next
            decision = await self._decide_next_handoff(initial_prompt, result)

            # Check if the objective is complete
            if decision.objective_complete:
                logger.info(
                    f"Objective complete after {iteration} iterations: {decision.message}"
                )
                objective_complete = True
                # Add final summary to conversation history
                self.conversation_history.append(
                    {"role": "assistant", "content": decision.message}
                )
                break

            # Get the next handoff agent
            next_agent = self._get_handoff_agent_by_name(decision.next_agent)
            if next_agent is None:
                logger.warning(
                    f"Could not find handoff agent '{decision.next_agent}'. Ending workflow."
                )
                break

            logger.info(f"Selected next handoff agent: {decision.next_agent}")
            logger.info(f"Handoff message: {decision.message}")

            # Call the handoff agent
            handoff_result = await Runner.run(
                starting_agent=next_agent,
                input=decision.message,
                context=context,
                run_config=run_config,
            )

            # Extract the handoff response
            handoff_response = self._extract_response_text(handoff_result)

            # Store the handoff response
            self.handoff_history.append(
                HandoffResponse(
                    agent_name=decision.next_agent, response=handoff_response
                )
            )

            # Add the handoff response to conversation history
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": f"[{decision.next_agent}] Query: {decision.message}",
                }
            )
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": f"[{decision.next_agent}] Response: {handoff_response}",
                }
            )

            # Ask the base agent to analyze the handoff response
            analysis_prompt = f'I received this response from the {decision.next_agent} agent: "{handoff_response}". Please analyze this response in the context of our objective: "{initial_prompt}". What should we do next?'

            analysis_result = await Runner.run(
                starting_agent=self.base_agent,
                input=analysis_prompt,
                context=context,
                run_config=run_config,
            )

            # Extract the analysis response
            analysis_response = self._extract_response_text(analysis_result)
            self.conversation_history.append(
                {"role": "assistant", "content": analysis_response}
            )

            # Update the result with the latest response
            result = analysis_result

        # If we reached max iterations without completing the objective
        if not objective_complete and iteration >= self.max_iterations:
            logger.warning(
                f"Reached maximum number of iterations ({self.max_iterations}) without completing the objective."
            )

            # Generate a final summary
            summary_prompt = f"We've reached the maximum number of iterations ({self.max_iterations}) trying to achieve this objective: \"{initial_prompt}\". Please summarize what we've learned and the current status."

            summary_result = await Runner.run(
                starting_agent=self.base_agent,
                input=summary_prompt,
                context=context,
                run_config=run_config,
            )

            # Update the final result
            result = summary_result

        logger.info(
            f"Multi-handoff agent workflow completed after {iteration} iterations."
        )
        return result

    async def _decide_next_handoff(
        self, objective: str, last_result: RunResult
    ) -> HandoffDecision:
        """
        Decide which handoff agent to call next based on the objective and conversation history.

        Args:
            objective: The original objective/prompt.
            last_result: The last result from the base agent.

        Returns:
            A HandoffDecision containing the next agent to call or indicating the objective is complete.
        """
        # Prepare conversation summary from recent history
        conversation_summary = "\n".join(
            [
                (
                    f"[{msg['role']}]: {msg['content'][:100]}..."
                    if len(msg["content"]) > 100
                    else f"[{msg['role']}]: {msg['content']}"
                )
                for msg in self.conversation_history[
                    -5:
                ]  # Include only the last 5 exchanges
            ]
        )

        # Get available handoff agent names
        available_agents = [
            handoff.name for handoff in self.base_agent.handoffs]
        available_agents_str = ", ".join(available_agents)

        decision_prompt = (
            f'Based on the objective: "{objective}"\n\n'
            f"And this conversation history:\n{conversation_summary}\n\n"
            f"Available handoff agents: {available_agents_str}\n\n"
            "Is the objective complete? If yes, explain why. If not, which handoff agent should we call next and what message should we send to them?\n\n"
            "Respond with EITHER:\n"
            '1. "OBJECTIVE COMPLETE: [reason the objective is complete]"\n'
            '2. "NEXT AGENT: [agent name]\nMESSAGE: [message to send to the agent]"'
        )

        # Call the base agent to decide
        decision_result = await Runner.run(
            starting_agent=self.base_agent,
            input=decision_prompt,
            context=None,
        )

        # Extract the decision
        decision_text = self._extract_response_text(decision_result)

        # Parse the decision text
        if "OBJECTIVE COMPLETE:" in decision_text:
            reason = decision_text.split("OBJECTIVE COMPLETE:", 1)[1].strip()
            return HandoffDecision(
                next_agent=None, message=reason, objective_complete=True
            )
        elif "NEXT AGENT:" in decision_text:
            # Extract the next agent and message
            lines = decision_text.split("\n")
            next_agent = None
            message_lines = []
            message_started = False

            for line in lines:
                if line.startswith("NEXT AGENT:"):
                    next_agent = line.split("NEXT AGENT:", 1)[1].strip()
                elif line.startswith("MESSAGE:"):
                    message_part = line.split("MESSAGE:", 1)[1].strip()
                    if message_part:
                        message_lines.append(message_part)
                    message_started = True
                elif message_started:
                    message_lines.append(line)

            message = "\n".join(message_lines).strip()

            if next_agent and message:
                return HandoffDecision(
                    next_agent=next_agent, message=message, objective_complete=False
                )

        # Default decision if parsing fails
        logger.warning(
            f"Could not parse decision text. Using default next agent.")
        default_agent = self._get_default_handoff_agent()
        return HandoffDecision(
            next_agent=default_agent.name if default_agent else None,
            message=f"I need assistance with this objective: {objective}",
            objective_complete=False,
        )

    def _extract_response_text(self, result: RunResult) -> str:
        """
        Extract the text response from a RunResult object.

        Args:
            result: The RunResult object.

        Returns:
            The extracted text response.
        """
        # Check if there is a final output
        if result.final_output is not None:
            if isinstance(result.final_output, str):
                return result.final_output
            else:
                try:
                    # Try to convert to string representation
                    return str(result.final_output)
                except:
                    pass

        # Get text from message output items
        for item in result.new_items:
            if hasattr(item, "type") and item.type == "message_output_item":
                return ItemHelpers.text_message_output(item)

        # If no text found, return empty string
        return ""

    def _get_handoff_agent_by_name(self, name: str) -> Optional[Agent[Any]]:
        """
        Get a handoff agent by name.

        Args:
            name: The name of the handoff agent to find.

        Returns:
            The handoff agent or None if not found.
        """
        if not name:
            return None

        for handoff in self.base_agent.handoffs:
            if handoff.name == name:
                return handoff
            elif (
                handoff.name.replace(" ", "_").lower(
                ) in name.replace(" ", "_").lower()
            ):
                # Handle case where name might be formatted differently
                # e.g., "EMR Writeback" vs "emr_writeback"
                # or "EMR Writeback" vs "emr writeback"
                return handoff
        return None

    def _get_default_handoff_agent(self) -> Optional[Agent[Any]]:
        """
        Get the default handoff agent (the first one).

        Returns:
            The default handoff or None if there are no handoffs.
        """
        if self.base_agent.handoffs:
            # Make sure we have at least one valid handoff
            for handoff in self.base_agent.handoffs:
                return handoff

        return None

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            The conversation history as a list of role/content dictionaries.
        """
        return self.conversation_history

    def get_handoff_history(self) -> List[HandoffResponse]:
        """
        Get the handoff history.

        Returns:
            The history of handoff responses.
        """
        return self.handoff_history
