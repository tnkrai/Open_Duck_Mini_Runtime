"""
CloudPublisher — Supabase Realtime Broadcast publisher for robot telemetry.

Publishes walk-loop state to a Supabase Broadcast channel so the TNKR
dashboard can receive telemetry over WSS (avoiding browser mixed-content
blocking on HTTPS pages).

Runs the async Supabase client in a background thread so the synchronous
50 Hz walk loop is never blocked.
"""

import asyncio
import os
import time
import threading

# Default Supabase credentials — the anon key is public by design (same as
# embedding in frontend JS).  Override via env vars for self-hosted setups.
_DEFAULT_SUPABASE_URL = "https://ahmocgtnahrabokouiki.supabase.co"
_DEFAULT_SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFobW9jZ3RuYWhyYWJva291aWtpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM1MTcxNTEsImV4cCI6MjA3OTA5MzE1MX0._KjbMvf10TlyuDvHLfaXDTejX1tXXy9G-Uvcxn6IEus"

SUPABASE_URL = os.environ.get("TNKR_SUPABASE_URL", _DEFAULT_SUPABASE_URL)
SUPABASE_KEY = os.environ.get("TNKR_SUPABASE_KEY", _DEFAULT_SUPABASE_KEY)


class CloudPublisher:
    """Publishes robot state to Supabase Broadcast at a throttled rate."""

    def __init__(self, channel_name: str, target_hz: float = 10):
        self.channel_name = channel_name
        self.min_interval = 1.0 / target_hz
        self.last_publish = 0.0
        self._channel = None
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._running = False
        self._publish_count = 0

    def start(self):
        """Start the background async event loop and connect to Supabase."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=15):
            print("[CloudPublisher] Warning: Supabase connection timed out")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception as e:
            print(f"[CloudPublisher] Connection failed: {e}")

    async def _connect(self):
        from supabase import acreate_client
        from realtime.types import RealtimeSubscribeStates

        supabase = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
        self._channel = supabase.channel(self.channel_name)

        def on_subscribe(status, err):
            if status == RealtimeSubscribeStates.SUBSCRIBED:
                print(f"[CloudPublisher] ✓ Subscribed to channel: {self.channel_name}")
                print(f"[CloudPublisher] ✓ Ready to broadcast at {1.0/self.min_interval:.0f} Hz")
                self._ready.set()
            elif err:
                print(f"[CloudPublisher] ✗ Subscribe error: {err}")

        await self._channel.subscribe(on_subscribe)

        # Keep the event loop alive while running
        while self._running:
            await asyncio.sleep(1)

    def publish(self, state: dict):
        """
        Called from the synchronous walk loop on every tick (50 Hz).
        Throttles to target_hz and schedules the async send.
        Fire-and-forget — never blocks the walk loop.
        """
        now = time.time()
        if now - self.last_publish < self.min_interval:
            return
        self.last_publish = now

        if self._channel and self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._channel.send_broadcast("state", state),
                    self._loop,
                )
                self._publish_count += 1
                if self._publish_count % 50 == 1:
                    print(f"[CloudPublisher] Sent frame #{self._publish_count} to {self.channel_name}")
            except Exception:
                pass  # fire-and-forget — robot must never stall on network

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
