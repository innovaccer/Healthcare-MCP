"""
HMCP Agent: Agent implementation for Healthcare Model Context Protocol servers

This module provides an Agent class that works with HMCPServerHelper to create
agents that can be used for handoffs in multi-agent workflows. It integrates with
the existing agents framework to enable seamless interactions between OpenAI agents
and HMCP servers.
"""

from __future__ import annotations

import dataclasses
import inspect
import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)
from typing_extensions import NotRequired, TypedDict, Awaitable

# Import from local hmcp package
from hmcp.client.hmcp_client import HMCPClient

# Import from external agents package (assuming agents package is installed)
# try:
from agents import Agent
from agents.agent import ToolsToFinalOutputResult
from agents.agent_output import AgentOutputSchemaBase

from agents.handoffs import Handoff
from agents.model_settings import ModelSettings
from agents import Model, ModelResponse, Usage
from agents.run_context import RunContextWrapper
from agents.tool import Tool, FunctionToolResult
from agents.util._types import MaybeAwaitable
from agents.items import ItemHelpers
from agents.result import RunResult
from agents.logger import logger
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

# Import HMCPServerHelper
from hmcp.client.client_connector import HMCPClientConnector

# Create a generic type variable for the context
TContext = TypeVar("TContext")


class StopAtTools(TypedDict):
    """A configuration to stop agent execution at specific tools."""

    stop_at_tool_names: List[str]
    """A list of tool names, any of which will stop the agent from running further."""


class HMCPAgentSettings(TypedDict):
    """Settings for the HMCP Agent."""

    max_tokens: NotRequired[int]
    """Maximum number of tokens to generate in a response."""

    temperature: NotRequired[float]
    """Temperature for controlling randomness (0-1)."""

    top_p: NotRequired[float]
    """Top-p for nucleus sampling (0-1)."""

    stop: NotRequired[List[str]]
    """List of stop sequences that will halt generation."""

    metadata: NotRequired[Dict[str, Any]]
    """Additional metadata to include with sampling requests."""


class HMCPModel(Model):
    """A model implementation that uses an HMCP server helper for sampling."""

    def __init__(
        self,
        hmcp_helper: HMCPClientConnector,
        settings: Optional[HMCPAgentSettings] = None,
    ):
        """Initialize the HMCP Model.

        Args:
            hmcp_helper: The HMCP Server Helper to use for sampling.
            settings: Optional settings for the HMCP sampling.
        """
        self.hmcp_helper = hmcp_helper
        self.settings = settings or {}

    async def get_response(
        self,
        system_instructions: Optional[str],
        input: List[Any],
        model_settings: ModelSettings,
        tools: Optional[List[Tool]] = None,
        output_schema: Optional[Any] = None,
        handoffs: Optional[List[Any]] = None,
        tracing: Optional[Any] = None,
        previous_response_id: Optional[str] = None,
    ):
        """Get a response from the HMCP server."""
        # Connect to the HMCP server if not already connected
        if not self.hmcp_helper.connected:
            await self.hmcp_helper.connect(transport="streamable-http")

        # Convert input items to message history format for HMCP
        messages_history = []

        # Add the input items to the message history
        for item in input:
            if hasattr(item, "role") and hasattr(item, "content"):
                messages_history.append(
                    {"role": item.role, "content": item.content})
            elif isinstance(item, dict) and "role" in item and "content" in item:
                messages_history.append(
                    {"role": item["role"], "content": item["content"]}
                )

        # Get user message from the last input item
        user_message = ""
        if input and hasattr(input[-1], "content"):
            user_message = input[-1].content

        # Combine model settings with HMCP settings
        hmcp_model_params = {**self.settings}
        if model_settings.temperature is not None:
            hmcp_model_params["temperature"] = model_settings.temperature
        if model_settings.max_tokens is not None:
            hmcp_model_params["max_tokens"] = model_settings.max_tokens

        try:
            response = await self.hmcp_helper.create_message(
                message=user_message,
                role="user",
                messages_history=messages_history,
                model_params=hmcp_model_params,
            )

            # Create model response
            items = []
            message_item = ResponseOutputMessage(
                id="__fake_id__",
                content=[],
                role=response.get("role", "assistant"),
                type="message",
                status="completed",
            )

            message_item.content.append(
                ResponseOutputText(
                    text=response.get("content", ""), type="output_text", annotations=[]
                )
            )

            items.append(message_item)

            return ModelResponse(
                output=items,
                usage=Usage(),
                response_id=None,
            )

        except Exception as e:
            logger.error(f"Error getting response from HMCP server: {e}")
            return ModelResponse(
                output=[
                    ResponseOutputMessage(
                        id="__fake_id__",
                        role="assistant",
                        content=[
                            ResponseOutputText(
                                text=f"Error: Could not get response from HMCP server: {str(e)}",
                                type="output_text",
                                annotations=[],
                            )
                        ],
                        type="message",
                        status="incomplete",
                    )
                ],
                usage=Usage(),
                response_id=None,
            )

    async def stream_response(
        self,
        system_instructions: Optional[str],
        input: List[Any],
        model_settings: ModelSettings,
        tools: Optional[List[Tool]] = None,
        output_schema: Optional[Any] = None,
        handoffs: Optional[List[Any]] = None,
        tracing: Optional[Any] = None,
        previous_response_id: Optional[str] = None,
    ):
        """
        Stream a response from the HMCP server.

        Note: Since HMCP server helper doesn't support streaming yet,
        this method just yields the final response.

        Args:
            system_instructions: Optional system instructions for the model.
            input: The input items for the model.
            model_settings: Settings for the model.
            tools: Optional list of tools the model can use.
            output_schema: Optional schema for the model output.
            handoffs: Optional list of handoffs the model can use.
            tracing: Optional tracing configuration.
            previous_response_id: Optional ID of the previous response.

        Yields:
            The stream of responses from the model.
        """
        response = await self.get_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
        )

        # Since we don't have streaming, yield a single "completed" event
        from openai.types.responses import ResponseCompletedEvent

        yield ResponseCompletedEvent(response=response)


@dataclass
class HMCPAgent(Generic[TContext]):
    """
    An agent that wraps an HMCPServerHelper to integrate with the agents framework.

    This agent allows HMCP servers with sampling capability to be used within multi-agent
    workflows, including handoffs between different agents.
    """

    name: str
    """The name of the agent."""

    hmcp_helper: HMCPClientConnector
    """The HMCP Server Helper instance used to communicate with the HMCP server."""

    instructions: Union[
        str,
        Callable[
            [RunContextWrapper[TContext], "HMCPAgent[TContext]"],
            MaybeAwaitable[str],
        ],
        None,
    ] = None
    """The instructions for the agent. This may not be used directly by the HMCP server,
    but is useful for documentation and when this agent is used as a handoff."""

    handoff_description: Optional[str] = None
    """A description of the agent. This is used when the agent is used as a handoff."""

    settings: HMCPAgentSettings = field(default_factory=dict)
    """Configures HMCP server-specific settings for sampling."""

    model: Union[str, Model] = field(default_factory=lambda: None)
    """The model implementation. For HMCPAgent, this will be automatically created using HMCPModel."""

    handoffs: List[Union[Agent[Any], Handoff[TContext]]
                   ] = field(default_factory=list)
    """Handoffs are sub-agents that the agent can delegate to."""

    tools: List[Tool] = field(default_factory=list)
    """A list of tools that the agent can use."""

    output_type: Optional[Union[type[Any], AgentOutputSchemaBase]] = None
    """The type of the output object."""

    tool_use_behavior: Union[
        Literal["run_llm_again", "stop_on_first_tool"],
        StopAtTools,
        Callable[
            [RunContextWrapper[TContext], List[FunctionToolResult]],
            MaybeAwaitable[ToolsToFinalOutputResult],
        ],
    ] = "run_llm_again"
    """Configures the tool use behavior."""

    message_history: List[Dict[str, str]] = field(default_factory=list)
    """Keeps track of the message history for the agent."""

    # Additional properties to match Agent interface
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    """Configures model-specific tuning parameters (e.g. temperature, top_p)."""

    mcp_servers: List[Any] = field(default_factory=list)
    """A list of MCP servers that the agent can use."""

    reset_tool_choice: bool = True
    """Whether to reset tool choice after tool usage."""

    def __post_init__(self):
        """
        Initialize the agent after the dataclass constructor runs.
        This ensures the HMCP helper is properly connected and the model is set up.
        """
        # Ensure the HMCP helper is properly initialized
        if not hasattr(self.hmcp_helper, "connected") or not self.hmcp_helper.connected:
            logger.warning(
                f"HMCP helper for agent '{self.name}' is not connected. "
                "You should call connect() before using this agent."
            )

        # Set up the model if not already provided
        if self.model is None:
            self.model = HMCPModel(self.hmcp_helper, self.settings)

    async def process_message(
        self,
        message: str,
        context: Optional[TContext] = None,
    ) -> Dict[str, Any]:
        """
        Process a message using the HMCP server.

        This is the main method for interacting with the HMCP agent. It sends a message
        to the HMCP server and returns the response.

        Args:
            message: The message to send to the HMCP agent.
            context: Optional context object for the agent.

        Returns:
            The response from the HMCP agent.
        """
        # Add the message to the history
        self.message_history.append({"role": "user", "content": message})

        # Connect to the HMCP server if not already connected
        if not self.hmcp_helper.connected:
            await self.hmcp_helper.connect(transport="streamable-http")

        try:
            # Send the message to the HMCP server
            response = await self.hmcp_helper.create_message(
                message=message,
                role="user",
                messages_history=self.message_history[
                    :-1
                ],  # Exclude the current message
                model_params=self.settings,
            )

            # Add the response to the message history
            self.message_history.append(
                {
                    "role": response.get("role", "assistant"),
                    "content": response.get("content", ""),
                }
            )

            return response

        except Exception as e:
            logger.error(f"Error processing message with HMCP agent: {e}")
            error_response = {
                "role": "assistant",
                "content": f"Error: Could not get response from HMCP agent: {str(e)}",
                "error": str(e),
            }
            self.message_history.append(error_response)
            return error_response

    def to_agent(self) -> Agent[TContext]:
        """
        Convert this HMCPAgent to a standard Agent object.

        This method creates an Agent instance that wraps this HMCPAgent and allows
        it to be used with the standard agents framework for handoffs.

        Returns:
            An Agent instance that wraps this HMCPAgent.
        """
        # Create a new Agent with the same properties as this HMCPAgent
        agent = Agent(
            name=self.name,
            instructions=self.instructions,
            handoff_description=self.handoff_description,
            handoffs=self.handoffs,
            model=self.model,
            model_settings=self.model_settings,
            tools=self.tools,

            output_type=self.output_type,
            tool_use_behavior=self.tool_use_behavior,
        )
        return agent

    async def get_system_prompt(
        self, run_context: RunContextWrapper[TContext]
    ) -> Optional[str]:
        """Get the system prompt for the agent."""
        if isinstance(self.instructions, str):
            return self.instructions
        elif callable(self.instructions):
            if inspect.iscoroutinefunction(self.instructions):
                return await cast(Awaitable[str], self.instructions(run_context, self))
            else:
                return cast(str, self.instructions(run_context, self))

        return None

    def clone(self, **kwargs: Any) -> "HMCPAgent[TContext]":
        """
        Make a copy of the agent, with the given arguments changed.

        Args:
            **kwargs: The arguments to change in the cloned agent.

        Returns:
            A new HMCPAgent with the specified changes.
        """
        return dataclasses.replace(self, **kwargs)

    def as_tool(
        self,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        custom_output_extractor: Optional[Callable[[
            RunResult], Awaitable[str]]] = None,
    ) -> Tool:
        """
        Transform this agent into a tool, callable by other agents.
        """

        from agents.tool import function_tool
        from agents.util import _transforms

        name = tool_name or _transforms.transform_string_function_style(
            self.name)
        description = (
            tool_description or f"Use the {self.name} HMCP Agent to process information"
        )

        @function_tool(name_override=name, description_override=description)
        async def run_hmcp_agent(context: RunContextWrapper, input: str) -> str:
            """Run the HMCP agent with the given input."""
            # Add the input to the message history
            self.message_history.append({"role": "user", "content": input})

            # Connect if needed
            if not self.hmcp_helper.connected:
                await self.hmcp_helper.connect(transport="streamable-http")

            try:
                # Process the message
                response = await self.hmcp_helper.create_message(
                    message=input,
                    role="user",
                    messages_history=self.message_history[:-1],
                    model_params=self.settings,
                )

                # Update history and return
                content = response.get("content", "")
                self.message_history.append(
                    {"role": response.get(
                        "role", "assistant"), "content": content}
                )

                return content

            except Exception as e:
                logger.error(f"Error running HMCP agent: {e}")
                return f"Error: Could not get response from HMCP agent: {str(e)}"

        return run_hmcp_agent

    async def cleanup(self) -> None:
        """
        Clean up resources used by the agent.

        This should be called when the agent is no longer needed to clean up
        any resources and close connections.
        """
        if hasattr(self.hmcp_helper, "cleanup"):
            await self.hmcp_helper.cleanup()

    async def __aenter__(self) -> "HMCPAgent[TContext]":
        """Enter the async context manager."""
        if not self.hmcp_helper.connected:
            await self.hmcp_helper.connect(transport="streamable-http")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager."""
        await self.cleanup()
