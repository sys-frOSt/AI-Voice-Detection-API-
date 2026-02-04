"""
Hugging Face Inference API Detector
Uses external API for deepfake detection (No local GPU required).
Uses the huggingface_hub InferenceClient for proper API access.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env to get HF_TOKEN
load_dotenv()


class HFDetector:
	"""
	Detects AI voices using Hugging Face Inference API.
	Free, no-GPU (serverless) solution.
	"""

	# Model for audio classification / deepfake detection
	MODEL_ID = "mo-thecreator/Deepfake-audio-detection"

	def __init__(self, api_token: str = None):
		"""
		Initialize with Hugging Face API token.
		If not provided, tries to read HF_TOKEN from env.
		"""
		self.api_token = api_token or os.getenv("HF_TOKEN")
		self.is_available = bool(self.api_token)
		self.client = None

		if self.is_available:
			try:
				from huggingface_hub import InferenceClient
				self.client = InferenceClient(token=self.api_token)
			except ImportError:
				print("huggingface_hub not installed. Run: pip install huggingface_hub")
				self.is_available = False

	def detect(self, audio_bytes: bytes) -> Dict[str, Any]:
		"""
		Send audio to Hugging Face API for detection.

		Args:
			audio_bytes: Raw audio bytes (MP3/WAV)

		Returns:
			Detection result dictionary
		"""
		if not self.is_available or not self.client:
			return {
				"classification": "UNKNOWN",
				"confidenceScore": 0.0,
				"explanation": "Hugging Face Token (HF_TOKEN) missing in .env or library not installed",
				"method": "hf_api_failed",
			}

		try:
			# Use audio_classification endpoint
			result = self.client.audio_classification(
				audio=audio_bytes,
				model=self.MODEL_ID,
			)

			# Result format: [{'label': 'real', 'score': 0.99}, {'label': 'fake', 'score': 0.01}]
			if not result:
				return {
					"classification": "UNKNOWN",
					"confidenceScore": 0.0,
					"explanation": "No result from HF API",
					"method": "hf_api_failed",
				}

			# Get top prediction
			top_result = result[0]
			label = (
				top_result.label.lower()
				if hasattr(top_result, "label")
				else str(top_result.get("label", "")).lower()
			)
			score = float(
				top_result.score if hasattr(top_result, "score") else top_result.get("score", 0)
			)

			# Map to our schema
			is_ai = label in ["fake", "spoof", "ai", "deepfake"]

			if is_ai:
				classification = "AI_GENERATED"
				reasons = "Deep learning model detected synthetic speech patterns"
			else:
				classification = "HUMAN"
				reasons = "Deep learning model confirmed natural human voice"

			return {
				"classification": classification,
				"confidenceScore": score,
				"explanation": reasons,
				"method": "hf_inference_api",
				"raw_label": label,
			}

		except Exception as e:
			error_msg = str(e)
			print(f"HF Detector Error: {error_msg}")

			# Check for model loading
			if "loading" in error_msg.lower():
				return {
					"classification": "UNKNOWN",
					"confidenceScore": 0.0,
					"explanation": "Model is loading, please retry in a few seconds",
					"method": "hf_api_loading",
				}

			return {
				"classification": "UNKNOWN",
				"confidenceScore": 0.0,
				"explanation": f"API Error: {error_msg[:100]}",
				"method": "hf_api_error",
			}


# Singleton
hf_detector = HFDetector()
