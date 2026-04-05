import asyncio
import json
import threading
import websockets


class StateServer:
    def __init__(self, port: int = 8765):
        self.port = port
        self._clients = set()
        self._loop = None
        self._thread = None
        self._server = None

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._server = await websockets.serve(
            self._handler, "0.0.0.0", self.port
        )
        await self._server.wait_closed()

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass  # We only broadcast, not receive
        finally:
            self._clients.discard(websocket)

    def broadcast(self, state: dict):
        if self._loop is None or self._loop.is_closed():
            return
        message = json.dumps(state)
        asyncio.run_coroutine_threadsafe(
            self._broadcast_async(message), self._loop
        )

    async def _broadcast_async(self, message: str):
        if not self._clients:
            return
        dead = set()
        for ws in self._clients.copy():
            try:
                await ws.send(message)
            except websockets.ConnectionClosed:
                dead.add(ws)
        self._clients -= dead

    def stop(self):
        if self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None
        self._server = None
