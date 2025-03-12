#!/usr/bin/env python3
"""
WebSocket client for communicating with the RuneLite client.

This module defines a WebSocketClient class that handles the communication
with the RuneLite plugin via WebSocket.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, Optional, Union, Callable

import nest_asyncio
import websockets
from websockets.legacy.client import WebSocketClientProtocol


class WebSocketClient:
    """Client for communicating with the RuneLite client via WebSocket."""

    def __init__(self, debug: bool = False, websocket_url: str = "ws://localhost:43595"):
        """Initialize the WebSocket client.
        
        Args:
            debug: Whether to enable debug logging
            websocket_url: The URL of the WebSocket server to connect to
        """
        # Configure logging
        self.logger = logging.getLogger("WebSocketClient")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # WebSocket connection info
        self.websocket_url = websocket_url
        self.ws: Optional[WebSocketClientProtocol] = None
        self.connected = False
        self.connection_event = threading.Event()
        
        # Synchronization and message handling
        self.ws_lock = asyncio.Lock()
        self.message_queue: asyncio.Queue[Union[str, bytes]] = asyncio.Queue()
        self.response_futures: Dict[int, asyncio.Future[Dict[str, Any]]] = {}
        self.next_request_id = 0
        
        # State tracking
        self.state: Optional[Dict[str, Any]] = None
        self.last_state_update = 0.0
        self.rate_limit = 0.6  # Minimum time between requests in seconds
        self.last_request_time = 0.0
        
        # Set up asyncio
        self.loop = asyncio.new_event_loop()
        nest_asyncio.apply(self.loop)
        
        # Start the WebSocket client
        self.ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        self.ws_thread.start()
    
    def _run_websocket_loop(self) -> None:
        """Run the websocket event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._websocket_client())
    
    async def _websocket_client(self) -> None:
        """Handle websocket connection and message processing."""
        # Connect to the WebSocket server
        async with websockets.connect(
            self.websocket_url, 
            ping_interval=None,  # Disable automatic pings
            ping_timeout=None,   # Disable ping timeout
            close_timeout=5      # Shorter close timeout
        ) as websocket:
            self.ws = websocket
            self.connected = True
            self.connection_event.set()
            self.logger.info("WebSocket connection established")
            
            # Start message processor
            message_processor = asyncio.create_task(self._process_messages())
            
            # Keep receiving messages until connection is closed
            while True:
                try:
                    # Receive message
                    message = await websocket.recv()
                    # Add to queue for processing
                    await self.message_queue.put(message)
                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("WebSocket connection closed")
                    break
    
    async def _process_messages(self) -> None:
        """Process incoming messages from the websocket."""
        while True:
            # Get message from queue
            message = await self.message_queue.get()
            
            # Decode binary messages
            if isinstance(message, bytes):
                self.logger.debug(f"Received binary message of {len(message)} bytes")
                message_str = message.decode('utf-8', errors='replace')
            else:
                message_str = message
            
            # Parse message
            data = self._parse_json(message_str)
            if data is None:
                self.logger.warning("Received invalid JSON message")
                self.message_queue.task_done()
                continue
            
            # Handle message
            self._handle_message(data)
            
            # Mark as processed
            self.message_queue.task_done()
    
    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON safely without exceptions."""
        if not text:
            return None
        
        # Parse JSON without try/except
        result = json.loads(text)
        if not isinstance(result, dict):
            self.logger.warning(f"Received non-dict data: {type(result)}")
            return None
        
        return result
    
    def _handle_message(self, data: Dict[str, Any]) -> None:
        """Handle an incoming message from the websocket."""
        # Check if this is a response to a specific request
        request_id = data.get("request_id")
        if request_id is not None and request_id in self.response_futures:
            # Complete the future for this request
            future = self.response_futures.pop(request_id)
            if not future.done():
                future.set_result(data)
            return
        
        # Handle state updates
        if self._is_valid_state(data):
            # Update state
            self.state = data
            self.last_state_update = time.time()
            self.logger.debug("Received valid state update")
        elif "error" in data:
            self.logger.error(f"Received error from server: {data['error']}")
        else:
            self.logger.debug(f"Received unhandled message type: {data.get('type', 'unknown')}")
    
    def _is_valid_state(self, data: Dict[str, Any]) -> bool:
        """Check if the data is a valid game state."""
        # Basic validation
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
        
        # Convert command to JSON
        command_json = json.dumps(command)
        
        # Send command in the event loop
        future = asyncio.run_coroutine_threadsafe(
            self._send_command_async(command_json),
            self.loop
        )
        
        # Wait for the result
        try:
            return future.result(timeout=2.0)
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False
    
    async def _send_command_async(self, command_json: str) -> bool:
        """Send a command to the RuneLite plugin asynchronously.
        
        Args:
            command_json: The JSON-encoded command to send
            
        Returns:
            bool: True if the command was sent, False otherwise
        """
        if not self.connected or not self.ws:
            return False
        
        # Apply rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.rate_limit:
            # Sleep to enforce rate limit
            await asyncio.sleep(self.rate_limit - time_since_last_request)
        
        # Update last request time
        self.last_request_time = time.time()
        
        # Send with lock to prevent concurrent sends
        async with self.ws_lock:
            await self.ws.send(command_json)
        
        return True
    
    def close(self) -> None:
        """Close the WebSocket connection."""
        if self.connected and self.ws:
            # Close the connection
            asyncio.run_coroutine_threadsafe(
                self._close_async(),
                self.loop
            )
            
            # Wait for the connection to close
            self.connection_event.clear()
            self.connected = False
            
            # Clean up the thread
            if hasattr(self, 'ws_thread') and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=1.0)
    
    async def _close_async(self) -> None:
        """Close the WebSocket connection asynchronously."""
        if self.ws:
            await self.ws.close() 