"""
CloudCommandReceiver — Supabase Realtime Broadcast subscriber for robot commands.

Receives walk commands from the TNKR dashboard via Supabase Broadcast channel
and writes them to /dev/shm/tnkr_remote_commands.json, where RemoteController
reads them. This enables keyboard control from HTTPS pages where direct HTTP
to the robot is blocked by browser security policies.

Mirrors CloudPublisher architecture: background async thread, never blocks
the 50 Hz walk loop.
"""

import asyncio
import json
import os
import time
import threading

COMMAND_FILE = "/dev/shm/tnkr_remote_commands.json"


class CloudCommandReceiver:
    """Subscribes to a Supabase Broadcast channel and writes commands to /dev/shm."""

    def __init__(self, channel_name: str, supabase_url: str, supabase_key: str):
        self.channel_name = channel_name
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self._channel = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._running = False

    def start(self):
        """Start the background async event loop and subscribe to Supabase."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            print("[CloudCommandReceiver] Warning: Supabase connection timed out")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:
            print(f"[CloudCommandReceiver] Connection failed: {e}")

    async def _connect(self):
        from supabase import acreate_client
        from realtime.types import RealtimeSubscribeStates

        supabase = await acreate_client(self.supabase_url, self.supabase_key)
        self._channel = supabase.channel(self.channel_name)

        def on_subscribe(status, err):
            if status == RealtimeSubscribeStates.SUBSCRIBED:
                print(f"[CloudCommandReceiver] Subscribed to {self.channel_name}")
                self._ready.set()
            elif err:
                print(f"[CloudCommandReceiver] Subscribe error: {err}")

        def on_command(payload):
            """Write received command to /dev/shm for RemoteController to read."""
            try:
                data = payload.get("payload", {})
                # Add local timestamp for staleness detection
                data["timestamp"] = time.time()
                # Atomic write: temp file + rename
                tmp_path = COMMAND_FILE + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(data, f)
                os.replace(tmp_path, COMMAND_FILE)
            except Exception:
                pass  # Never crash — robot safety first

        self._channel.on_broadcast("command", on_command)
        await self._channel.subscribe(on_subscribe)

        # Keep the event loop alive while running
        while self._running:
            await asyncio.sleep(1)

    def stop(self):
        """Clean up the Supabase connection and background thread."""
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3)
        self._channel = None
        self._loop = None
        self._thread = None
