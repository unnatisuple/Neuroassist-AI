"""
NeuroAssist AI v2 — Global Error Handler
Every failure path returns a specific, actionable message with a trace ID.
No unhandled exceptions should ever surface as "Something went wrong."
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from backend.dependencies import generate_trace_id
from datetime import datetime, timezone


class GlobalErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Catches ALL unhandled exceptions and returns structured error responses
    with trace IDs for support purposes. Per spec: no generic "Something went wrong."
    """

    async def dispatch(self, request: Request, call_next):
        trace_id = generate_trace_id()
        request.state.trace_id = trace_id

        try:
            response = await call_next(request)
            return response

        except FileNotFoundError as e:
            logger.error(f"[{trace_id}] File not found: {e}")
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "Resource Not Found",
                    "detail": f"The requested file or resource was not found: {str(e)}",
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except PermissionError as e:
            logger.error(f"[{trace_id}] Permission denied: {e}")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Access Denied",
                    "detail": f"Insufficient permissions for this operation: {str(e)}",
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except ConnectionError as e:
            logger.error(f"[{trace_id}] Connection error: {e}")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service Unavailable",
                    "detail": (
                        "A downstream service (database or external API) is currently unreachable. "
                        "Please retry in a few moments."
                    ),
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except TimeoutError as e:
            logger.error(f"[{trace_id}] Timeout: {e}")
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "error": "Request Timeout",
                    "detail": (
                        "The operation timed out. For model inference, this may indicate "
                        "the server is under heavy load. Please retry."
                    ),
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except ValueError as e:
            logger.warning(f"[{trace_id}] Validation error: {e}")
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": "Validation Error",
                    "detail": f"Invalid input: {str(e)}",
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            # Log full traceback for debugging, return sanitized message to user
            logger.exception(f"[{trace_id}] Unhandled exception: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal Server Error",
                    "detail": (
                        f"An unexpected error occurred ({type(e).__name__}). "
                        f"Please contact support with trace ID: {trace_id}"
                    ),
                    "trace_id": trace_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
