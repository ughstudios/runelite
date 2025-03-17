
"""
WebSocket client for communicating with the RuneLite client.

This module defines a WebSocketClient class that handles the communication
with the RuneLite plugin via WebSocket.
"""

import asyncio
import json
import threading
import time
from typing import Any, Dict, Optional, Union
import concurrent.futures

import nest_asyncio  
import websockets
from websockets.legacy.client import WebSocketClientProtocol
from .logging_utils import get_logger


class WebSocketClient:
    """Client for communicating with the RuneLite client via WebSocket."""

    def __init__(
        self, websocket_url: str = "ws://localhost:43595"
    ):
        """Initialize the WebSocket client.

        Args:
            debug: Whether to enable debug logging
            websocket_url: The URL of the WebSocket server to connect to
        """
        
        self.logger = get_logger()
        self.websocket_url = websocket_url
        self.ws: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.connection_event = threading.Event()
        self.ws_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[Union[str, bytes]] = asyncio.Queue()
        self.response_futures: Dict[int, Any] = {}
        self.next_request_id = 0
        self.state: Optional[Dict[str, Any]] = None
        self.last_state_update = 0.0
        self.rate_limit = 1.5  
        self.last_request_time = 0.0
        self.loop = asyncio.new_event_loop()
        nest_asyncio.apply(self.loop)
        self.ws_thread = threading.Thread(
            target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()

    def _run_websocket_loop(self) -> None:
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())

    async def _websocket_client(self) -> None:
        """Handle websocket connection and message processing."""
        async with websockets.connect(
            self.websocket_url,
            ping_interval=None,
            ping_timeout=None,
            close_timeout=5
        ) as websocket:
            self.ws = websocket
            self.connected = True
            self.connection_event.set()
            self.logger.info("WebSocket connection established")
            asyncio.create_task(self._process_messages())
            while True:
                message = await websocket.recv()
                await self.message_queue.put(message)

    async def _process_messages(self) -> None:
        """Process incoming messages from the websocket."""
        while True:
            message = await self.message_queue.get()
            if isinstance(message, bytes):
                self.logger.debug(
                    f"Received binary message of {
                        len(message)} bytes")
                message_str = message.decode("utf-8", errors="replace")
            else:
                message_str = message
            data = self._parse_json(message_str)
            if not data:
                self.logger.warning("Received invalid JSON message")
                self.message_queue.task_done()
                continue
            self._handle_message(data)
            self.message_queue.task_done()

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON safely without exceptions."""
        if not text:
            return None

        result = json.loads(text)
        if not isinstance(result, dict):
            self.logger.warning(f"Received non-dict data: {type(result)}")
            return None

        return result

    def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle an incoming message from the websocket."""
        request_id = data.get("request_id")
        if request_id is not None and request_id in self.response_futures:
            future = self.response_futures.pop(request_id)
            if not future.done():
                future.set_result(data)
            return
        if "error" in data:
            self.state = data
            self.last_state_update = time.time()
            self.logger.error(f"Received error from server: {data['error']}")
            
            if "rate limit" in data["error"].lower():
                self.rate_limit = min(
                    self.rate_limit * 1.5, 5.0
                )  
                self.logger.warning(
                    f"Increasing rate limit to {self.rate_limit} seconds"
                )
        elif self._is_valid_state(data):
            self.state = data
            self.last_state_update = time.time()
            self.logger.debug("Received valid state update")
        elif not data:
            self.logger.debug("Received empty response from server")
            self.state = {
                "error": "Empty response from server",
                "timestamp": time.time(),
            }
            self.last_state_update = time.time()
        else:
            self.logger.debug(
                f"Received unhandled message type: {
                    data.get(
                        'type', 'unknown')}"
            )
            self.state = data
            self.last_state_update = time.time()

    def _is_valid_state(self, data: Dict[str, Any]) -> bool:
        """Check if the data is a valid game state."""
        if "error" in data:
            return False
        if data.get("success"):
            return True
        required_keys = ["player", "npcs", "objects", "groundItems"]
        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data["player"], dict):
            return False
        return True

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for the WebSocket connection to be established.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if connected, False if timed out
        """
        return self.connection_event.wait(timeout=timeout)

    def send_command(self, command: Dict[str, Any]) -> bool:
        """Send a command to the RuneLite plugin.

        Args:
            command: The command to send

        Returns:
            bool: True if the command was sent, False otherwise
        """
        if not self.connected or not self.ws:
            self.logger.warning("Cannot send command: WebSocket not connected")
            return False
        command_json = json.dumps(command)
        future = asyncio.run_coroutine_threadsafe(
            self._send_command_async(command_json), self.loop
        )
        return future.result(
            timeout=5.0
        )

    async def _send_command_async(self, command_json: str) -> bool:
        """Send a command to the RuneLite plugin asynchronously.

        Args:
            command_json: The JSON-encoded command to send

        Returns:
            bool: True if the command was sent, False otherwise
        """
        if not self.connected or not self.ws:
            return False
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last_request)

        self.last_request_time = time.time()
        async with self.ws_lock:
            await self.ws.send(command_json)
        return True

    def close(self) -> None:
        """Close the WebSocket connection."""
        if not self.connected or not self.ws:
            self.logger.debug("WebSocket already closed or never connected")
            return
        self.logger.info("Closing WebSocket connection...")
        self.connected = False
        self.connection_event.clear()
        for request_id, future in list(self.response_futures.items()):
            if not future.done():
                try:
                    future.set_exception(
                        ConnectionError("WebSocket connection closed"))
                except Exception:
                    pass
        self.response_futures.clear()

        try:
            close_task = asyncio.run_coroutine_threadsafe(
                self._close_async(), self.loop
            )
            close_task.result(timeout=2.0)
        except concurrent.futures.TimeoutError:
            self.logger.warning("Timeout during WebSocket close operation")
        except Exception as e:
            self.logger.error(f"Error during WebSocket close: {e}")
        self.state = None
        self.ws = None
        self.logger.info("WebSocket connection cleanup completed")

    async def _close_async(self) -> None:
        """Close the WebSocket connection asynchronously and clean up tasks."""
        try:
            current_task = asyncio.current_task(self.loop)
            tasks_to_cancel = [
                task
                for task in asyncio.all_tasks(self.loop)
                if task is not current_task
                and not task.done()
            ]

            if tasks_to_cancel:
                self.logger.debug(
                    f"Cancelling {len(tasks_to_cancel)} pending tasks")
                for task in tasks_to_cancel:
                    task_name = (
                        task.get_name() if hasattr(task, "get_name") else str(task)
                    )
                    self.logger.debug(f"Cancelling task: {task_name}")
                    task.cancel()
                await asyncio.wait(
                    tasks_to_cancel, timeout=1.0, return_when=asyncio.ALL_COMPLETED
                )
            
            if self.ws and hasattr(self.ws, "open") and self.ws.open:
                try:
                    await asyncio.wait_for(self.ws.close(), timeout=1.0)
                    self.logger.info("WebSocket connection closed successfully")
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout during WebSocket close")
                except Exception as e:
                    self.logger.error(f"Error closing WebSocket: {e}")

        except Exception as e:
            self.logger.error(f"Error in _close_async: {e}")
            
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.connected = False
        self.connection_event.clear()
