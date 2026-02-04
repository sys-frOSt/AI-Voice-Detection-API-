"""
CNN-Based Voice Detector using MFCC features
Trained on Human vs Non-Human (AI) voice classification
"""
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Dict, Any, Optional


# Model parameters (must match training)
SR = 16000  # Sample rate
DURATION = 22.0  # seconds
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 256
INCLUDE_DELTAS = True
MAX_LEN_SAMPLES = int(SR * DURATION)


class SmallCNN(nn.Module):
	"""CNN architecture matching indibert
	."""

	def __init__(self, n_feats: int = 120):  # 40 MFCC * 3 (with deltas)
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((2, 2)),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.fc = nn.Linear(64, 2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.net(x)
		x = x.view(x.size(0), -1)
		return self.fc(x)


class CNNVoiceDetector:
	"""
	CNN-based voice detector using MFCC features.
	Trained on human-nonhuman dataset from Kaggle.
	"""

	def __init__(self, model_path: str = None):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model = None
		self.is_loaded = False

		# Default model path from .env or fallback to models/model_best.pt
		if model_path is None:
			model_path = os.getenv("CNN_MODEL_PATH", "models/model_best.pt")
			# Make path relative to project root if not absolute
			if not os.path.isabs(model_path):
				model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
		self.model_path = model_path

	def load_model(self) -> bool:
		"""Load the trained CNN model."""
		if self.is_loaded:
			return True

		if not os.path.exists(self.model_path):
			print(f"CNN model not found at {self.model_path}")
			return False

		try:
			print(f"Loading CNN MFCC model from {self.model_path}...")
			self.model = SmallCNN(n_feats=120)  # 40 * 3 = 120 features

			# Load weights
			state_dict = torch.load(self.model_path, map_location=self.device)
			self.model.load_state_dict(state_dict)
			self.model.to(self.device)
			self.model.eval()

			self.is_loaded = True
			print(f"✓ CNN model loaded on {self.device}")
			return True

		except Exception as e:
			print(f"Failed to load CNN model: {e}")
			return False

	def extract_mfcc_features(self, audio: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
		"""
		Extract MFCC + delta + delta2 features from audio.

		Args:
			audio: Audio samples as numpy array
			sr: Sample rate

		Returns:
			MFCC features of shape (120, time_frames)
		"""
		try:
			# Resample if needed
			if sr != SR:
				audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)

			# Pad/trim to fixed length
			if len(audio) < MAX_LEN_SAMPLES:
				audio = np.pad(audio, (0, MAX_LEN_SAMPLES - len(audio)), mode="constant")
			else:
				audio = audio[:MAX_LEN_SAMPLES]

			# Extract MFCCs
			mfcc = librosa.feature.mfcc(
				y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
			)

			if INCLUDE_DELTAS:
				delta = librosa.feature.delta(mfcc)
				delta2 = librosa.feature.delta(mfcc, order=2)
				feat = np.concatenate([mfcc, delta, delta2], axis=0)
			else:
				feat = mfcc

			return feat.astype(np.float32)

		except Exception as e:
			print(f"MFCC extraction failed: {e}")
			return None

	def detect(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
		"""
		Detect if audio is AI-generated or human using CNN on MFCC features.

		Args:
			audio: Audio samples as numpy array
			sr: Sample rate

		Returns:
			Detection result with classification, confidence, and explanation
		"""
		if not self.is_loaded:
			if not self.load_model():
				return {
					"classification": "UNKNOWN",
					"confidenceScore": 0.5,
					"explanation": "CNN model not available",
					"method": "cnn_unavailable",
				}

		# Extract MFCC features
		features = self.extract_mfcc_features(audio, sr)
		if features is None:
			return {
				"classification": "UNKNOWN",
				"confidenceScore": 0.5,
				"explanation": "Feature extraction failed",
				"method": "cnn_error",
			}

		try:
			# Prepare tensor: (C, T) -> (1, 1, C, T)
			tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(self.device)

			# Inference
			with torch.no_grad():
				logits = self.model(tensor)
				probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
				pred = np.argmax(probs)

			# pred: 0 = nonhuman (AI), 1 = human
			if pred == 0:
				classification = "AI_GENERATED"
				confidence = float(probs[0])
				explanation = "CNN detected synthetic voice patterns in MFCC features"
			else:
				classification = "HUMAN"
				confidence = float(probs[1])
				explanation = "CNN confirmed natural voice patterns in MFCC features"

			return {
				"classification": classification,
				"confidenceScore": round(confidence, 3),
				"explanation": explanation,
				"method": "cnn_mfcc",
				"probs": {"ai": float(probs[0]), "human": float(probs[1])},
			}

		except Exception as e:
			print(f"CNN inference failed: {e}")
			return {
				"classification": "UNKNOWN",
				"confidenceScore": 0.5,
				"explanation": f"CNN inference error: {str(e)}",
				"method": "cnn_error",
			}


# Lazy-loaded singleton
_cnn_detector = None


def get_cnn_detector() -> CNNVoiceDetector:
	"""Get or create the CNN voice detector singleton."""
	global _cnn_detector
	if _cnn_detector is None:
		_cnn_detector = CNNVoiceDetector()
	return _cnn_detector
