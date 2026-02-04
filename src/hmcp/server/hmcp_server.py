from __future__ import annotations
import logging
from typing import Any, Literal, Protocol, Optional
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.shared.context import RequestContext
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import (
    BearerAuthBackend,
    RequireAuthMiddleware,
)
from mcp.server.auth.provider import OAuthAuthorizationServerProvider
from mcp.server.streamable_http import EventStore
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from starlette.applications import Starlette
from pydantic import RootModel, BaseModel

# Configure logging for the HMCP server module
logger = logging.getLogger(__name__)

# Use the MCP SDK's CreateMessageRequest instead of defining a custom one
# This ensures compatibility with the standard MCP protocol
CreateMessageRequest = types.CreateMessageRequest

# Extended server result type to include CreateMessageResult
# This dynamically extends the MCP ServerResult type to support HMCP's extended functionality
new_server_result_base = RootModel[
    types.EmptyResult
    | types.InitializeResult
    | types.CompleteResult
    | types.GetPromptResult
    | types.ListPromptsResult
    | types.ListResourcesResult
    | types.ListResourceTemplatesResult
    | types.ReadResourceResult
    | types.CallToolResult
    | types.ListToolsResult
    | types.CreateMessageResult
]

# Dynamically create a new class that extends the base ServerResult with our additions
types.ServerResult = type("ServerResult", (new_server_result_base,), {})

# Update the ClientRequest type to use our custom CreateMessageRequest
types.ClientRequest = RootModel[
    types.PingRequest
    | types.InitializeRequest
    | types.CompleteRequest
    | types.SetLevelRequest
    | types.GetPromptRequest
    | types.ListPromptsRequest
    | types.ListResourcesRequest
    | types.ListResourceTemplatesRequest
    | types.ReadResourceRequest
    | types.SubscribeRequest
    | types.UnsubscribeRequest
    | types.CallToolRequest
    | types.ListToolsRequest
    | CreateMessageRequest
]

types.StopReason = Literal["endTurn", "stopSequence", "maxTokens", "infoNeeded"] | str


class SamplingFnT(Protocol):
    """
    Protocol defining the signature for sampling callback functions.

    Sampling callbacks must accept a context and parameters, and return either
    a message result or an error. This protocol ensures type safety when implementing
    custom sampling handlers.

    Args:
        context: The request context containing metadata about the client request
        params: Parameters for the sampling operation including messages and generation settings

    Returns:
        Either a successful CreateMessageResult or an ErrorData object
    """

    async def __call__(
        self,
        context: RequestContext[Any, Any],
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData: ...


async def _default_sampling_callback(
    context: RequestContext[Any, Any],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.ErrorData:
    """
    Default sampling callback that returns an error indicating sampling is not supported.

    This is used when no custom sampling callback is provided to the server.
    It ensures that clients receive a meaningful error message rather than
    a method-not-implemented error.

    Args:
        context: The request context (unused in the default implementation)
        params: The sampling parameters (unused in the default implementation)

    Returns:
        An ErrorData object with an appropriate error message
    """
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="Sampling not supported",
    )


class HMCPServer(FastMCP):
    """
    HMCP Server extends the FastMCP Server with sampling capability.

    This class provides an implementation of the HMCP (Healthcare Model Context Protocol)
    server that adds text generation capabilities on top of the standard MCP protocol.
    HMCP servers can handle CreateMessageRequest messages from clients and generate
    text completions based on provided prompts and context.

    The server advertises its sampling capabilities to clients during initialization
    and provides a flexible API for implementing custom sampling logic through
    callback functions.
    """

    def __init__(
        self,
        name: str,
        host: str = "127.0.0.1",
        port: int = 8060,
        debug: bool = False,
        log_level: str = "INFO",
        version: str | None = None,
        instructions: str | None = None,
        auth_server_provider: (
            OAuthAuthorizationServerProvider[Any, Any, Any] | None
        ) = None,
        event_store: EventStore | None = None,
        samplingCallback: SamplingFnT | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the HMCP Server.

        Args:
            name: The name of the server that will be reported to clients
            version: The version of the server (optional), used for compatibility checks
            instructions: Human-readable instructions for using the server (optional)
            samplingCallback: A callback function to handle sampling requests (optional)
                              If not provided, the server will return errors for sampling requests
            **kwargs: Additional settings to pass to the underlying FastMCP implementation
                       These can include configuration for logging, transports, etc.
        """
        # Initialize the parent FastMCP server with standard settings
        super().__init__(
            name=name,
            instructions=instructions,
            auth_server_provider=auth_server_provider,
            event_store=event_store,
            host=host,
            port=port,
            debug=debug,
            log_level=log_level,
            *args,
            **kwargs,
        )

        # Define experimental capabilities with sampling for advertisement to clients
        # This allows clients to detect that this server supports HMCP sampling features
        self.experimentalCapabilities = {
            "hmcp": {
                "sampling": True,
                "version": "0.1.0",
            }
        }

        # Store the sampling callback or use the default one if none provided
        # The callback will be invoked whenever a CreateMessageRequest is received
        self._samplingCallback = samplingCallback or _default_sampling_callback

        # Register the handler for processing CreateMessageRequest messages
        self._registerSamplingHandler()

        # Override the get_capabilities method to include our custom capabilities
        # This is necessary to add sampling capabilities to the server's advertised features
        self._original_get_capabilities = self._mcp_server.get_capabilities
        self._mcp_server.get_capabilities = self.patched_get_capabilities

    # TODO: // Need to ask kuldeep if we can remove this as its same as fast-mcp method
    def sse_app(self, mount_path: str | None = None) -> Starlette:
        """Return an instance of the SSE server app."""

        # Update mount_path in settings if provided
        if mount_path is not None:
            self.settings.mount_path = mount_path

        # Create normalized endpoint considering the mount path
        normalized_message_endpoint = self._normalize_path(
            self.settings.mount_path, self.settings.message_path
        )

        # Set up auth context and dependencies

        sse = SseServerTransport(
            normalized_message_endpoint,
        )

        async def handle_sse(scope: Scope, receive: Receive, send: Send):
            # Add client ID from auth context into request context if available

            async with sse.connect_sse(
                scope,
                receive,
                send,
            ) as streams:
                await self._mcp_server.run(
                    streams[0],
                    streams[1],
                    self._mcp_server.create_initialization_options(),
                )
            return Response()

        # Create routes
        routes: list[Route | Mount] = []
        middleware: list[Middleware] = []
        required_scopes = []

        # Add auth endpoints if auth provider is configured
        if self._auth_server_provider:
            assert self.settings.auth
            from mcp.server.auth.routes import create_auth_routes

            required_scopes = self.settings.auth.required_scopes or []

            middleware = [
                # extract auth info from request (but do not require it)
                Middleware(
                    AuthenticationMiddleware,
                    backend=BearerAuthBackend(
                        provider=self._auth_server_provider,
                    ),
                ),
                # Add the auth context middleware to store
                # authenticated user in a contextvar
                Middleware(AuthContextMiddleware),
            ]
            routes.extend(
                create_auth_routes(
                    provider=self._auth_server_provider,
                    issuer_url=self.settings.auth.issuer_url,
                    service_documentation_url=self.settings.auth.service_documentation_url,
                    client_registration_options=self.settings.auth.client_registration_options,
                    revocation_options=self.settings.auth.revocation_options,
                )
            )

        # When auth is not configured, we shouldn't require auth
        if self._auth_server_provider:
            # Auth is enabled, wrap the endpoints with RequireAuthMiddleware
            routes.append(
                Route(
                    self.settings.sse_path,
                    endpoint=RequireAuthMiddleware(handle_sse, required_scopes),
                    methods=["GET"],
                )
            )
            routes.append(
                Mount(
                    self.settings.message_path,
                    app=RequireAuthMiddleware(sse.handle_post_message, required_scopes),
                )
            )
        else:
            # Auth is disabled, no need for RequireAuthMiddleware
            # Since handle_sse is an ASGI app, we need to create a compatible endpoint
            async def sse_endpoint(request: Request) -> Response:
                # Convert the Starlette request to ASGI parameters
                return await handle_sse(request.scope, request.receive, request._send)  # type: ignore[reportPrivateUsage]

            routes.append(
                Route(
                    self.settings.sse_path,
                    endpoint=sse_endpoint,
                    methods=["GET"],
                )
            )
            routes.append(
                Mount(
                    self.settings.message_path,
                    app=sse.handle_post_message,
                )
            )
        # mount these routes last, so they have the lowest route matching precedence
        routes.extend(self._custom_starlette_routes)

        # Create Starlette app with routes and middleware
        return Starlette(
            debug=self.settings.debug, routes=routes, middleware=middleware
        )

    def streamable_http_app(self) -> Starlette:
        """Return an instance of the StreamableHTTP server app."""
        from starlette.middleware import Middleware
        from starlette.routing import Mount

        # Create session manager on first call (lazy initialization)
        if self._session_manager is None:
            self._session_manager = StreamableHTTPSessionManager(
                app=self._mcp_server,
                event_store=self._event_store,
                json_response=self.settings.json_response,
                stateless=self.settings.stateless_http,  # Use the stateless setting
            )

        # Create the ASGI handler
        async def handle_streamable_http(
            scope: Scope, receive: Receive, send: Send
        ) -> None:
            await self.session_manager.handle_request(scope, receive, send)

        # Create routes
        routes: list[Route | Mount] = []
        middleware: list[Middleware] = []
        required_scopes = []

        # Add auth endpoints if auth provider is configured
        if self._auth_server_provider:
            assert self.settings.auth
            from mcp.server.auth.routes import create_auth_routes

            required_scopes = self.settings.auth.required_scopes or []

            middleware = [
                Middleware(
                    AuthenticationMiddleware,
                    backend=BearerAuthBackend(
                        provider=self._auth_server_provider,
                    ),
                ),
                Middleware(AuthContextMiddleware),
            ]
            routes.extend(
                create_auth_routes(
                    provider=self._auth_server_provider,
                    issuer_url=self.settings.auth.issuer_url,
                    service_documentation_url=self.settings.auth.service_documentation_url,
                    client_registration_options=self.settings.auth.client_registration_options,
                    revocation_options=self.settings.auth.revocation_options,
                )
            )
            routes.append(
                Mount(
                    self.settings.streamable_http_path,
                    app=RequireAuthMiddleware(handle_streamable_http, required_scopes),
                )
            )
        else:
            # Auth is disabled, no wrapper needed
            routes.append(
                Mount(
                    self.settings.streamable_http_path,
                    app=handle_streamable_http,
                )
            )

        routes.extend(self._custom_starlette_routes)

        return Starlette(
            debug=self.settings.debug,
            routes=routes,
            middleware=middleware,
            lifespan=lambda app: self.session_manager.run(),
        )

    def _registerSamplingHandler(self):
        """Register the handler for CreateMessageRequest."""

        async def samplingHandler(req: types.CreateMessageRequest):
            """Handle sampling/createMessage requests."""

            try:
                # Get the current request context
                ctx = self._mcp_server.request_context

                # Process the request using the registered sampling callback
                response = await self._samplingCallback(ctx, req.params)

                # Return the response directly if it's an error, otherwise wrap it in a ServerResult
                if isinstance(response, types.ErrorData):
                    return response
                else:
                    return types.ServerResult(response)

            except Exception as e:
                logger.error(f"Error in sampling handler: {str(e)}")
                return types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Error processing sampling request: {str(e)}",
                )

        # Register our handler for CreateMessageRequest type
        self._mcp_server.request_handlers[CreateMessageRequest] = samplingHandler

    def sampling(self):
        """
        Decorator to register a sampling callback function.

        This provides a convenient way to define the sampling logic for the server.
        The decorated function will be called whenever the server receives a
        CreateMessageRequest.

        Returns:
            A decorator function that registers the decorated function as the sampling callback

        Example:
            @hmcp_server.sampling()
            async def handle_sampling(context, params):
                # Generate a response based on the messages in params.messages
                return types.CreateMessageResult(
                    message=types.SamplingMessage(
                        role="assistant",
                        content="Generated response here"
                    )
                )
        """

        def decorator(func: SamplingFnT):
            logger.debug("Registering sampling handler")
            self._samplingCallback = func
            return func

        return decorator

    def patched_get_capabilities(
        self,
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> types.ServerCapabilities:
        """
        Override the get_capabilities method to provide custom HMCP capabilities.

        This method extends the standard MCP capabilities with HMCP-specific
        capabilities, allowing clients to discover the server's sampling features.

        Args:
            notification_options: Options for configuring server notifications
            experimental_capabilities: Additional experimental capabilities to include

        Returns:
            ServerCapabilities object with HMCP sampling capabilities included
        """
        # Call the parent class method to get standard capabilities
        capabilities = self._original_get_capabilities(
            notification_options, experimental_capabilities
        )

        # Add custom HMCP-specific capabilities to the experimental section
        # This advertises the server's sampling functionality to clients
        capabilities.experimental = {
            **capabilities.experimental,
            **self.experimentalCapabilities,
        }

        logger.debug("Custom HMCP capabilities added: %s", capabilities.experimental)

        return capabilities
