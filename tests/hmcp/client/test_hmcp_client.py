import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any
import mcp.types as types
from mcp import ClientSession
from hmcp.client.hmcp_client import HMCPClient
from hmcp.client.client_connector import HMCPClientConnector


@pytest.fixture
def mock_session():
    """Mock ClientSession"""
    session = Mock(spec=ClientSession)
    session.send_request = AsyncMock()
    return session


@pytest.fixture
def hmcp_client(mock_session):
    """Create HMCPClient instance with mocked session"""
    return HMCPClient(mock_session)


@pytest.fixture(autouse=True)
def patch_client_request():
    with patch("mcp.types.ClientRequest", side_effect=lambda x: x), \
            patch("hmcp.client.hmcp_client.types.ClientRequest", side_effect=lambda x: x):
        yield


@pytest.mark.asyncio
async def test_create_message_basic(hmcp_client, mock_session):
    """Test basic create_message functionality"""
    # Setup test data
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        ),
        types.SamplingMessage(
            role="assistant", content=types.TextContent(type="text", text="Hi there!")
        ),
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    # Call the method
    result = await hmcp_client.create_message(messages)

    # Verify the result
    assert result == expected_result

    # Verify the request was sent correctly
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages
    assert request.params.maxTokens == 1000
    assert request.params.temperature is None
    assert request.params.topP is None
    assert request.params.stop is None
    assert request.params.metadata is None


@pytest.mark.asyncio
async def test_create_message_with_all_params(hmcp_client, mock_session):
    """Test create_message with all optional parameters"""
    # Setup test data
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    max_tokens = 500
    temperature = 0.7
    top_p = 0.9
    stop = ["\n", "Human:"]
    metadata = {"key": "value"}

    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    # Call the method
    result = await hmcp_client.create_message(
        messages=messages,
        maxTokens=max_tokens,
        temperature=temperature,
        topP=top_p,
        stop=stop,
        metadata=metadata,
    )

    # Verify the result
    assert result == expected_result

    # Verify the request was sent correctly
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages
    assert request.params.maxTokens == max_tokens
    assert request.params.temperature == temperature
    assert request.params.topP == top_p
    assert request.params.stop == stop
    assert request.params.metadata == metadata


@pytest.mark.asyncio
async def test_create_message_with_empty_messages(hmcp_client, mock_session):
    """Test create_message with empty message list"""
    messages: List[types.SamplingMessage] = []
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_error_response(hmcp_client, mock_session):
    """Test create_message with error response from server"""
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    error_response = types.ErrorData(code=400, message="Invalid request")
    mock_session.send_request.return_value = error_response

    result = await hmcp_client.create_message(messages)

    assert result == error_response
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_long_messages(hmcp_client, mock_session):
    """Test create_message with long message content"""
    messages = [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(
                type="text", text="This is a very long message " * 100
            ),
        )
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_special_characters(hmcp_client, mock_session):
    """Test create_message with special characters in content"""
    messages = [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(
                type="text", text="Special chars: !@#$%^&*()_+{}|:\"<>?[]\\;',./"
            ),
        )
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_metadata_types(hmcp_client, mock_session):
    """Test create_message with different metadata value types"""
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    metadata = {
        "string": "value",
        "number": 123,
        "boolean": True,
        "null": None,
        "array": [1, 2, 3],
        "object": {"nested": "value"},
    }
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages, metadata=metadata)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert hasattr(request, "method")
    assert request.method == "sampling/createMessage"
    assert hasattr(request, "params")
    assert request.params.messages == messages
    assert request.params.metadata == metadata


# Tests for HMCPClientConnector with streamable-http transport


@pytest.fixture
def mock_streamable_http_client():
    """Mock streamable_http_client as async context manager"""
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_get_session_id = Mock(return_value="test-session-123")
    
    class MockStreamableHTTP:
        async def __aenter__(self):
            return (mock_read, mock_write, mock_get_session_id)
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def mock_client(*args, **kwargs):
        return MockStreamableHTTP()
    
    return mock_client


@pytest.fixture
def mock_sse_client():
    """Mock SSE client as async context manager"""
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    
    class MockSSE:
        async def __aenter__(self):
            return (mock_read, mock_write)
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    def mock_client(*args, **kwargs):
        return MockSSE()
    
    return mock_client


@pytest.mark.asyncio
async def test_client_connector_streamable_http_initialization():
    """Test HMCPClientConnector initialization with streamable-http"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=True
    )
    
    assert connector.server_url == "http://localhost:8050"
    assert connector.debug is True
    assert connector.connected is False
    assert connector.session is None
    assert connector.client is None


@pytest.mark.asyncio
async def test_client_connector_streamable_http_connect(mock_streamable_http_client):
    """Test connecting with streamable-http transport"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=False
    )
    
    with patch('hmcp.client.client_connector.streamable_http_client', mock_streamable_http_client), \
         patch('hmcp.client.client_connector.ClientSession') as mock_session_class, \
         patch('hmcp.client.client_connector.httpx.AsyncClient'):
        
        # Mock session initialization
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=types.InitializeResult(
            protocolVersion="2024-11-05",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="test-server", version="1.0.0")
        ))
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Connect using streamable-http
        result = await connector.connect(transport="streamable-http")
        
        # Verify connection details
        assert connector.connected is True
        assert connector.session is not None
        assert connector.client is not None
        assert result["name"] == "test-server"
        assert result["version"] == "1.0.0"
        assert result["protocolVersion"] == "2024-11-05"


@pytest.mark.asyncio
async def test_client_connector_sse_connect(mock_sse_client):
    """Test connecting with SSE transport"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=False
    )
    
    with patch('hmcp.client.client_connector.sse_client', mock_sse_client), \
         patch('hmcp.client.client_connector.ClientSession') as mock_session_class:
        
        # Mock session initialization
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=types.InitializeResult(
            protocolVersion="2024-11-05",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="test-server", version="1.0.0")
        ))
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Connect using SSE
        result = await connector.connect(transport="sse")
        
        # Verify connection details
        assert connector.connected is True
        assert result["name"] == "test-server"


@pytest.mark.asyncio
async def test_client_connector_invalid_transport():
    """Test connecting with invalid transport"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=False
    )
    
    with patch('hmcp.client.client_connector.ClientSession'):
        with pytest.raises(ValueError, match="Invalid transport"):
            await connector.connect(transport="invalid-transport")


@pytest.mark.asyncio
async def test_client_connector_streamable_http_with_auth():
    """Test streamable-http connection with OAuth authentication"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        client_id="test-client",
        client_secret="test-secret",
        scopes=["read", "write"],
        debug=False
    )
    
    assert connector.oauth_client is not None
    
    with patch('hmcp.client.client_connector.streamable_http_client') as mock_http_client, \
         patch('hmcp.client.client_connector.ClientSession') as mock_session_class, \
         patch('hmcp.client.client_connector.httpx.AsyncClient'), \
         patch.object(connector.oauth_client, 'set_client_credentials_token', new_callable=AsyncMock) as mock_set_token, \
         patch.object(connector.oauth_client, 'get_auth_header', return_value={"Authorization": "Bearer test-token"}), \
         patch.object(connector.oauth_client, '__aenter__', new_callable=AsyncMock), \
         patch.object(connector.oauth_client, '__aexit__', new_callable=AsyncMock):
        
        # Mock streamable http client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_get_session_id = Mock(return_value="test-session-123")
        mock_http_client.return_value.__aenter__.return_value = (mock_read, mock_write, mock_get_session_id)
        
        # Mock session initialization
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=types.InitializeResult(
            protocolVersion="2024-11-05",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="test-server", version="1.0.0")
        ))
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Connect with auth
        result = await connector.connect(transport="streamable-http")
        
        # Verify auth was set up
        mock_set_token.assert_called_once()
        assert connector.connected is True


@pytest.mark.asyncio
async def test_client_connector_streamable_http_cleanup():
    """Test proper cleanup of streamable-http connection"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=False
    )
    
    with patch('hmcp.client.client_connector.streamable_http_client') as mock_http_client, \
         patch('hmcp.client.client_connector.ClientSession') as mock_session_class, \
         patch('hmcp.client.client_connector.httpx.AsyncClient'):
        
        # Mock streamable http client
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_get_session_id = Mock(return_value="test-session-123")
        mock_http_client.return_value.__aenter__.return_value = (mock_read, mock_write, mock_get_session_id)
        
        # Mock session
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock(return_value=types.InitializeResult(
            protocolVersion="2024-11-05",
            capabilities=types.ServerCapabilities(),
            serverInfo=types.Implementation(name="test-server", version="1.0.0")
        ))
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        # Connect and cleanup
        await connector.connect(transport="streamable-http")
        assert connector.connected is True
        
        await connector.cleanup()
        assert connector.connected is False
        assert connector.session is None
        assert connector.client is None


@pytest.mark.asyncio
async def test_client_connector_streamable_http_connection_failure():
    """Test handling of connection failures with streamable-http"""
    connector = HMCPClientConnector(
        url="http://localhost:8050",
        debug=False
    )
    
    with patch('hmcp.client.client_connector.streamable_http_client') as mock_http_client, \
         patch('hmcp.client.client_connector.httpx.AsyncClient'):
        
        # Make connection fail
        mock_http_client.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception, match="Connection failed"):
            await connector.connect(transport="streamable-http")
        
        # Verify cleanup occurred
        assert connector.connected is False
