import os
import pytest
from unittest.mock import Mock, patch, AsyncMock, PropertyMock
from typing import Any
import mcp.types as types
from hmcp.server.hmcp_server import HMCPServer, SamplingFnT, CreateMessageRequest

from contextvars import ContextVar
from mcp.shared.context import RequestContext
from mcp.server.lowlevel.server import request_ctx


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key environment variable"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def basic_server():
    return HMCPServer(
        name="test_server"
    )


# Create a mock request context for testing
@pytest.fixture
def mock_request_context():
    ctx = Mock(spec=RequestContext)
    token = request_ctx.set(ctx)
    yield ctx
    request_ctx.reset(token)


def test_server_initialization(basic_server):
    """Test basic server initialization"""
    assert basic_server.name == "test_server"
    assert "hmcp" in basic_server.experimentalCapabilities
    assert basic_server.experimentalCapabilities["hmcp"]["sampling"] is True


@pytest.mark.asyncio
async def test_sampling_callback_registration(basic_server):
    """Test registering a custom sampling callback"""

    @basic_server.sampling()
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            message=types.SamplingMessage(
                role="assistant", content=types.TextContent(text="Test response")
            )
        )

    # Verify the callback was registered
    assert basic_server._samplingCallback == mock_sampling


@pytest.mark.asyncio
async def test_default_sampling_callback(basic_server):
    """Test the default sampling callback behavior"""
    params = types.CreateMessageRequestParams(
        messages=[], maxTokens=100  # Add required field
    )
    context = Mock()

    result = await basic_server._samplingCallback(context, params)
    assert isinstance(result, types.ErrorData)
    assert result.code == types.INVALID_REQUEST
    assert result.message == "Sampling not supported"


def test_sse_app_creation(basic_server):
    """Test SSE app creation with authentication middleware"""
    app = basic_server.sse_app()
    assert app is not None
    # Verify routes are properly configured
    routes = [route for route in app.routes]
    assert len(routes) == 2  # Should have SSE path and message path routes


def test_streamable_http_app_creation(basic_server):
    """Test streamable-http app creation"""
    app = basic_server.streamable_http_app()
    assert app is not None
    # Verify routes are properly configured
    routes = [route for route in app.routes]
    assert len(routes) >= 1  # Should have at least the streamable-http path
    # Verify session manager was created
    assert basic_server.session_manager is not None


@pytest.mark.asyncio
async def test_streamable_http_session_manager(basic_server):
    """Test that session manager is properly initialized for streamable-http"""
    # First access should create session manager
    assert basic_server._session_manager is None
    app = basic_server.streamable_http_app()
    assert basic_server._session_manager is not None
    
    # Second access should reuse the same session manager
    session_manager_1 = basic_server._session_manager
    app2 = basic_server.streamable_http_app()
    session_manager_2 = basic_server._session_manager
    assert session_manager_1 is session_manager_2


def test_streamable_http_settings(basic_server):
    """Test streamable-http specific settings"""
    assert basic_server.settings.streamable_http_path == "/mcp"
    assert basic_server.settings.stateless_http is False  # Default is False for stateful sessions
    assert basic_server.settings.json_response is False


def test_patched_get_capabilities(basic_server):
    """Test the patched get_capabilities method"""
    # Create a proper mock for notification options with required attributes
    notification_options = Mock()
    notification_options.prompts_changed = False
    notification_options.resources_changed = False
    notification_options.tools_changed = False
    notification_options.logging_changed = False

    experimental_capabilities = {}

    capabilities = basic_server.patched_get_capabilities(
        notification_options, experimental_capabilities
    )

    assert capabilities.experimental is not None
    assert "hmcp" in capabilities.experimental
    assert capabilities.experimental["hmcp"]["sampling"] is True


@pytest.mark.asyncio
async def test_sampling_handler_with_valid_message(basic_server, mock_request_context):
    """Test sampling handler with a valid message"""

    # Create a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    basic_server._samplingCallback = mock_sampling

    # Create test message and request
    message = types.SamplingMessage(
        role="user",
        content=types.TextContent(
            type="text", text="Test message"  # Add required field
        ),
    )
    params = types.CreateMessageRequestParams(
        messages=[message], maxTokens=100  # Add required field
    )
    request = CreateMessageRequest(
        method="sampling/createMessage", params=params  # Add required method field
    )

    # Get the handler and execute it
    handler = basic_server._mcp_server.request_handlers[CreateMessageRequest]
    result = await handler(request)

    assert isinstance(result, types.ServerResult)
    assert result.root.role == "assistant"
    assert result.root.content.text == "Test response"
    assert result.root.model == "test-model"


@pytest.mark.asyncio
async def test_sampling_callback_error(basic_server, mock_request_context):
    """Test handling of errors from sampling callback"""

    # Create a mock sampling callback that raises an exception
    async def mock_sampling(context, params):
        raise ValueError("Test error")

    basic_server._samplingCallback = mock_sampling

    # Create test message and request
    message = types.SamplingMessage(
        role="user", content=types.TextContent(type="text", text="Test message")
    )
    params = types.CreateMessageRequestParams(
        messages=[message], maxTokens=100)
    request = CreateMessageRequest(
        method="sampling/createMessage", params=params)

    # Get the handler and execute it
    handler = basic_server._mcp_server.request_handlers[CreateMessageRequest]
    result = await handler(request)

    assert isinstance(result, types.ErrorData)
    assert "Test error" in result.message
