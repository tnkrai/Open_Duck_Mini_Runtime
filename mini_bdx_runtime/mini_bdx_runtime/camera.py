"""
Camera module for the Open Duck Mini robot.

Uses picamzero (Raspberry Pi Camera Zero library) to capture images from the
robot's onboard camera. The camera is mounted in the robot's head and rotated
90° clockwise relative to the frame, so captured images are rotated to correct
for this.

Currently used by:
  - fc_test.py: GPT-4o-mini vision agent that uses the camera to navigate
                and find objects in the environment.

Not yet integrated with:
  - v2_rl_walk_mujoco.py: The walk script does not stream camera frames.
  - tnkr_server.py: No HTTP/WebSocket endpoint exposes the camera feed.
  - Telemetry pipeline: Camera frames are not included in the Supabase
    broadcast, so the TNKR dashboard cannot display a live video feed.

Hardware:
  - Raspberry Pi Camera (via picamzero library)
  - Mounted in the duck's head, rotated 90° clockwise
  - Output: 512x512 JPEG images

Configuration:
  - Enabled via duck_config.json -> expression_features.camera (bool)
  - Default: False (camera is off unless explicitly enabled)
  - Note: duck_config reads this flag but the walk script doesn't act on it yet.
"""

from picamzero import Camera
import cv2
import base64
import os


class Cam:
    """Captures and encodes images from the robot's onboard Pi camera."""

    def __init__(self):
        # picamzero auto-detects the connected Pi camera
        self.cam = Camera()

    def get_encoded_image(self) -> str:
        """Capture a frame, correct orientation, and return as base64 JPEG.

        The pipeline:
          1. Capture raw frame from the Pi camera as a numpy array
          2. Resize to 512x512 (smaller for faster transfer/inference)
          3. Convert BGR -> RGB (OpenCV uses BGR by default)
          4. Rotate 90° clockwise to correct for the camera's physical mounting
          5. Save to disk as JPEG (needed for base64 encoding)
          6. Read back and encode as base64 string

        Returns:
            Base64-encoded JPEG string, suitable for embedding in JSON
            or passing to vision APIs (e.g., OpenAI GPT-4o image input).
        """
        im = self.cam.capture_array()
        im = cv2.resize(im, (512, 512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)

        # Write to a temp file then re-read for base64 encoding
        # TODO: encode directly from numpy array to avoid disk I/O
        cv2.imwrite("/home/bdxv2/aze.jpg", im)

        return self.encode_image("/home/bdxv2/aze.jpg")

    def encode_image(self, image_path: str) -> str:
        """Read an image file from disk and return its base64 encoding.

        Args:
            image_path: Absolute path to a JPEG/PNG image file.

        Returns:
            Base64-encoded string of the image bytes.

        Raises:
            FileNotFoundError: If the image file doesn't exist at the path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
