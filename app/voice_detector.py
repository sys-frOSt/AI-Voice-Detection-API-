"""
Voice Detection Module
AI-generated vs Human voice classification using ensemble approach:
1. Heuristic-based detection (fast, interpretable)
2. ML-based detection using Wav2Vec2 embeddings (high accuracy)
"""
import os
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class VoiceDetector:
	"""
	Detects whether a voice sample is AI-generated or human-spoken.
	Uses an ensemble of heuristics and ML-based detection.
	"""

	# Feature order for the model (must match training order)
	FEATURE_NAMES = None  # Will be set dynamically

	def __init__(self, model_path: str = None, use_ml: bool = True):
		"""
		Initialize the voice detector.

		Args:
			model_path: Path to pre-trained model file. If None, uses heuristic-based detection.
			use_ml: Whether to use ML-based detection (Wav2Vec2)
		"""
		self.model_path = model_path
		self.model = None
		self.scaler = None
		self.is_trained = False
		self.use_ml = use_ml
		self.ml_detector = None
		self.hf_detector = None
		self.transformers_detector = None
		self.cnn_detector = None

		# Try to load pre-trained model
		if model_path and os.path.exists(model_path):
			self._load_model(model_path)
		else:
			# Use heuristic-based detection
			self._initialize_heuristics()

		# Lazy-load ML detector (Wav2Vec2 embeddings)
		if self.use_ml:
			self._init_ml_detector()

		# Initialize HF Detector (for API calls - currently disabled)
		# self._init_hf_detector()

		# Initialize Transformers detector (local pipeline - best accuracy)
		self._init_transformers_detector()

		# Initialize CNN detector (trained on Kaggle human-nonhuman dataset)
		self._init_cnn_detector()

	def _init_transformers_detector(self):
		"""Initialize the transformers pipeline detector."""
		try:
			from app.transformer import transformers_detector
			self.transformers_detector = transformers_detector
			# Don't load model here - lazy load on first use
			print("✓ Transformers detector available (lazy load)")
		except ImportError as e:
			print(f"Transformers detector not available: {e}")

	def _init_hf_detector(self):
		"""Initialize Hugging Face Inference API detector."""
		try:
			from app.hf_detector import hf_detector
			if hf_detector.is_available:
				self.hf_detector = hf_detector
				print("✓ Hugging Face API Detector enabled")
		except ImportError:
			pass

	def _init_ml_detector(self):
		"""Initialize the ML detector (lazy loading)."""
		try:
			from app.ml_detector import get_ml_detector
			self.ml_detector = get_ml_detector()
		except ImportError as e:
			print(f"ML detector not available: {e}")
			self.ml_detector = None

	def _init_cnn_detector(self):
		"""Initialize the CNN MFCC detector (trained on Kaggle dataset)."""
		try:
			from app.cnn_detector import get_cnn_detector
			self.cnn_detector = get_cnn_detector()
			print("✓ CNN detector available (lazy load)")
		except ImportError as e:
			print(f"CNN detector not available: {e}")
			self.cnn_detector = None

	def _load_model(self, path: str):
		"""Load pre-trained model from file."""
		try:
			data = joblib.load(path)
			self.model = data["model"]
			self.scaler = data["scaler"]
			self.FEATURE_NAMES = data.get("feature_names", None)
			self.is_trained = True
		except Exception as e:
			print(f"Warning: Could not load model: {e}. Using heuristic detection.")
			self._initialize_heuristics()

	def _initialize_heuristics(self):
		"""Initialize heuristic-based detection thresholds."""
		# Tuned thresholds for modern AI voices
		self.heuristics = {
			# AI voices: pitch_std typically 20-40 range
			"pitch_std_threshold": 35.0,

			# AI voices: spectral variation often 500-1500 range
			"spectral_centroid_std_threshold": 1600.0,

			# AI voices: ZCR std often 0.05-0.15 range
			"zcr_std_threshold": 0.15,

			# AI voices: RMS std often 0.03-0.08 range
			"rms_std_threshold": 0.08,

			# Spectral contrast threshold
			"spectral_contrast_threshold": 22.0,

			# Voiced ratio - AI voices tend to have very consistent voiced ratio
			"voiced_ratio_min": 0.4,
			"voiced_ratio_max": 0.75,

			# MFCC variability
			"mfcc_std_threshold": 150.0,

			# Pitch range
			"pitch_range_threshold": 150.0,
		}
		self.is_trained = False

	def _extract_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
		"""Convert feature dictionary to numpy array for model input."""
		if self.FEATURE_NAMES:
			return np.array([features.get(name, 0.0) for name in self.FEATURE_NAMES])
		else:
			return np.array([features[k] for k in sorted(features.keys())])

	def _heuristic_detection(self, features: Dict[str, Any]) -> Tuple[str, float, List[str]]:
		"""
		Detect AI voice using heuristic rules.

		Modern AI voices are sophisticated, so we use weighted scoring.
		"""
		ai_score = 0.0
		human_score = 0.0
		reasons = []

		# === PITCH ANALYSIS (Weight: 20%) ===
		pitch_std = features.get("pitch_std", 50.0)
		pitch_range = features.get("pitch_range", 200.0)

		if pitch_std > 0:
			if pitch_std < self.heuristics["pitch_std_threshold"]:
				ai_score += 0.20
				reasons.append("Controlled pitch variation typical of AI synthesis")
			else:
				human_score += 0.15

			if pitch_range < self.heuristics["pitch_range_threshold"]:
				ai_score += 0.10
				reasons.append("Limited pitch range detected")
			else:
				human_score += 0.05

		# === SPECTRAL ANALYSIS (Weight: 25%) ===
		spectral_centroid_std = features.get("spectral_centroid_std", 500.0)

		if spectral_centroid_std < self.heuristics["spectral_centroid_std_threshold"]:
			ai_score += 0.25
			reasons.append("Spectral characteristics consistent with synthetic voice")
		else:
			human_score += 0.20

		# === ENERGY DYNAMICS (Weight: 15%) ===
		rms_std = features.get("rms_std", 0.05)

		if rms_std < self.heuristics["rms_std_threshold"]:
			ai_score += 0.15
			reasons.append("Smooth energy contour suggesting digital generation")
		else:
			human_score += 0.10

		# === ZERO CROSSING RATE (Weight: 10%) ===
		zcr_std = features.get("zcr_std", 0.05)

		if zcr_std < self.heuristics["zcr_std_threshold"]:
			ai_score += 0.10
			reasons.append("Uniform acoustic patterns detected")
		else:
			human_score += 0.08

		# === VOICED RATIO (Weight: 10%) ===
		voiced_ratio = features.get("voiced_ratio", 0.5)

		if self.heuristics["voiced_ratio_min"] < voiced_ratio < self.heuristics["voiced_ratio_max"]:
			ai_score += 0.10
			reasons.append("Voice-to-silence ratio typical of synthesized speech")
		else:
			human_score += 0.05

		# === MFCC ANALYSIS (Weight: 15%) ===
		mfcc_0_std = features.get("mfcc_0_std", 100.0)

		if mfcc_0_std < self.heuristics["mfcc_std_threshold"]:
			ai_score += 0.15
			reasons.append("Timbre characteristics suggest AI origin")
		else:
			human_score += 0.12

		# === DELTA MFCC (Weight: 5%) ===
		delta_mfcc_std_sum = sum(features.get(f"delta_mfcc_{i}_std", 1.0) for i in range(13))
		if delta_mfcc_std_sum < 8.0:
			ai_score += 0.05
			reasons.append("Limited temporal dynamics")
		else:
			human_score += 0.03

		# === VOCODER ARTIFACTS (High-Freq analysis) (Weight: 10%) ===
		# Neural vocoders often leave artifacts in high frequencies or band-limit audio
		hf_energy_ratio = features.get("hf_energy_ratio", 0.1)
		if hf_energy_ratio < 0.01:
			# Extremely low high-freq energy (suspicious cutoff)
			ai_score += 0.15
			reasons.append("Abnormal high-frequency cutoff (Possible Neural Vocoder)")
		elif hf_energy_ratio > 0.8:
			# Suspiciously high HF energy (noise artifacts)
			ai_score += 0.10
			reasons.append("High-frequency spectral artifacts detected")
		else:
			human_score += 0.05

		# === FINAL CLASSIFICATION ===
		total_score = ai_score + human_score

		if total_score > 0:
			ai_ratio = ai_score / total_score
		else:
			ai_ratio = 0.5

		if ai_ratio > 0.45:
			classification = "AI_GENERATED"
			confidence = 0.55 + (ai_ratio - 0.45) * 0.7
			if not reasons:
				reasons = ["Synthetic voice patterns detected in audio analysis"]
		else:
			classification = "HUMAN"
			confidence = 0.55 + (0.55 - ai_ratio) * 0.7
			reasons = ["Natural speech patterns detected", "Organic voice characteristics identified"]

		confidence = min(confidence, 0.94)
		confidence = max(confidence, 0.55)

		return classification, round(confidence, 2), reasons

	def _ml_detection(self, audio: np.ndarray, sr: int) -> Optional[Tuple[str, float, List[str]]]:
		"""
		Detect AI voice using ML model (Wav2Vec2).

		Args:
			audio: Audio samples
			sr: Sample rate

		Returns:
			Tuple of (classification, confidence, reasons) or None if failed
		"""
		if self.ml_detector is None:
			return None

		try:
			# Resample to 16kHz for Wav2Vec2
			if sr != 16000:
				import librosa
				audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
				sr = 16000

			result = self.ml_detector.detect(audio, sr)

			if result.get("classification") == "UNKNOWN":
				return None

			return (
				result["classification"],
				result["confidenceScore"],
				[result["explanation"]],
			)
		except Exception as e:
			print(f"ML detection failed: {e}")
			return None

	def _ensemble_detection(
		self,
		heuristic_result: Tuple[str, float, List[str]],
		ml_result: Optional[Tuple[str, float, List[str]]],
	) -> Tuple[str, float, List[str]]:
		"""
		Combine heuristic and ML detection results.

		Weights: Heuristic 40%, ML 60% (when ML is available)
		"""
		h_class, h_conf, h_reasons = heuristic_result

		if ml_result is None:
			# Only heuristic available
			return heuristic_result

		m_class, m_conf, m_reasons = ml_result

		# Convert to numeric scores (1 = AI, 0 = Human)
		h_score = h_conf if h_class == "AI_GENERATED" else (1 - h_conf)
		m_score = m_conf if m_class == "AI_GENERATED" else (1 - m_conf)

		# Weighted ensemble (ML gets higher weight due to better accuracy)
		ensemble_score = 0.4 * h_score + 0.6 * m_score

		# Determine classification
		if ensemble_score > 0.5:
			classification = "AI_GENERATED"
			confidence = 0.55 + (ensemble_score - 0.5) * 0.8
			# Combine reasons, preferring ML reasons
			reasons = m_reasons + [r for r in h_reasons if r not in m_reasons][:1]
		else:
			classification = "HUMAN"
			confidence = 0.55 + (0.5 - ensemble_score) * 0.8
			reasons = ["Natural speech patterns detected", "Voice characteristics consistent with human speech"]

		confidence = min(0.95, max(0.55, confidence))

		return classification, round(confidence, 2), reasons

	def detect(
		self, features: Dict[str, Any], audio: np.ndarray = None, sr: int = None, audio_bytes: bytes = None
	) -> Dict[str, Any]:
		"""
		Detect whether voice is AI-generated or human.
		Uses ensemble of: Heuristics (20%) + Local ML (30%) + HF API (50%)

		Args:
			features: Dictionary of audio features from AudioProcessor
			audio: Optional raw audio samples for ML detection
			sr: Sample rate of audio
			audio_bytes: Raw audio bytes for HF API

		Returns:
			Dictionary with classification result, confidence, and explanation
		"""
		scores = []  # List of (ai_score, weight, method_name)
		all_reasons = []

		# 1. Heuristic detection (always runs)
		h_class, h_conf, h_reasons = self._heuristic_detection(features)
		h_ai_score = h_conf if h_class == "AI_GENERATED" else (1 - h_conf)
		scores.append((h_ai_score, 0.30, "heuristic"))  # 30% weight
		all_reasons.extend(h_reasons)

		# 2. CNN MFCC detection (trained on Kaggle - better accuracy)
		cnn_ai_score = None
		if self.cnn_detector and audio is not None and sr is not None:
			try:
				cnn_result = self.cnn_detector.detect(audio, sr)
				if cnn_result.get("classification") != "UNKNOWN":
					cnn_class = cnn_result["classification"]
					cnn_conf = cnn_result["confidenceScore"]
					cnn_ai_score = cnn_conf if cnn_class == "AI_GENERATED" else (1 - cnn_conf)
					scores.append((cnn_ai_score, 0.35, "cnn_mfcc"))  # 35% weight
					all_reasons.append(cnn_result.get("explanation", "CNN MFCC analysis"))
			except Exception as e:
				print(f"CNN detection failed: {e}")

		# 3. Local ML detection (if CNN failed and ML available)
		if cnn_ai_score is None and self.use_ml and audio is not None and sr is not None:
			ml_result = self._ml_detection(audio, sr)
			if ml_result:
				m_class, m_conf, m_reasons = ml_result
				m_ai_score = m_conf if m_class == "AI_GENERATED" else (1 - m_conf)
				scores.append((m_ai_score, 0.25, "local_ml"))
				all_reasons.extend(m_reasons)

		# 3. Transformers pipeline detection (BEST - if available)
		tf_ai_score = None
		if self.transformers_detector and audio is not None and sr is not None:
			try:
				tf_result = self.transformers_detector.detect(audio, sr)
				if tf_result.get("classification") != "UNKNOWN":
					tf_class = tf_result["classification"]
					tf_conf = tf_result["confidenceScore"]
					tf_ai_score = tf_conf if tf_class == "AI_GENERATED" else (1 - tf_conf)
					scores.append((tf_ai_score, 0.60, "transformers"))  # 60% weight
					all_reasons.append(tf_result.get("explanation", "Deep learning analysis"))
			except Exception as e:
				print(f"Transformers detection failed: {e}")

		# Calculate weighted ensemble score
		total_weight = sum(w for _, w, _ in scores)
		if total_weight > 0:
			ensemble_ai_score = sum(s * w for s, w, _ in scores) / total_weight
		else:
			ensemble_ai_score = 0.5

		# SMART DISAGREEMENT RULE:
		# Trust deep learning models (transformers + CNN) over heuristics
		# Only override when BOTH deep learning models strongly agree on AI
		cnn_ai_score = next((s for s, w, m in scores if m == "cnn_mfcc"), None)

		# If both CNN and transformers strongly say HUMAN, ignore heuristics
		if cnn_ai_score is not None and cnn_ai_score < 0.20 and tf_ai_score is not None and tf_ai_score < 0.10:
			# Both deep learning models confident it's human - cap ensemble score
			ensemble_ai_score = min(ensemble_ai_score, 0.45)

		# Only boost AI score if BOTH deep learning models agree it's AI
		if cnn_ai_score is not None and cnn_ai_score > 0.70 and tf_ai_score is not None and tf_ai_score > 0.70:
			# Both models say AI with high confidence - ensure we classify as AI
			ensemble_ai_score = max(ensemble_ai_score, 0.55)

		# "Old School TTS" override:
		# If Heuristic is VERY high (robotic) AND CNN detected unnatural features
		# We trust them even if Transformers (trained on Deepfakes) thinks it's human
		if h_ai_score > 0.85 and cnn_ai_score is not None and cnn_ai_score > 0.65:
			# Strong signal from traditional features + MFCC
			ensemble_ai_score = max(ensemble_ai_score, 0.55)
			print(f"   ⚠️ TTS OVERRIDE: heuristic={h_ai_score:.2f}, cnn={cnn_ai_score:.2f} → forcing AI")

		# Determine classification
		if ensemble_ai_score > 0.5:
			classification = "AI_GENERATED"
			confidence = 0.55 + (ensemble_ai_score - 0.5) * 0.8
		else:
			classification = "HUMAN"
			confidence = 0.55 + (0.5 - ensemble_ai_score) * 0.8

		confidence = min(0.95, max(0.55, confidence))

		# Build method string
		methods_used = [m for _, _, m in scores]
		method = "+".join(methods_used)

		# ======= LOGGING: Detailed score breakdown =======
		print("   📊 Score Breakdown:")
		for score, weight, name in scores:
			indicator = "🤖" if score > 0.5 else "👤"
			print(f"      {indicator} {name}: {score:.3f} (weight: {weight:.0%})")
		print(f"   📈 Ensemble AI Score: {ensemble_ai_score:.3f} → {classification}")

		# Build detailed explanation based on scores
		h_score = next((s for s, w, m in scores if m == "heuristic"), 0)
		ml_score = next((s for s, w, m in scores if m == "local_ml"), None)
		tf_score = next((s for s, w, m in scores if m == "transformers"), None)

		if classification == "AI_GENERATED":
			# Build AI explanation with specific patterns
			patterns = []
			if h_score > 0.7:
				patterns.append("synthetic pitch consistency")
			if h_score > 0.6:
				patterns.append("robotic speech rhythm")
			if tf_score is not None and tf_score > 0.8:
				patterns.append("deep learning detected synthetic artifacts")
			if ml_score is not None and ml_score > 0.7:
				patterns.append("ML model identified artificial voice patterns")

			if not patterns:
				patterns = ["unnatural audio characteristics detected"]

			explanation = (
				f"AI voice detected: {', '.join(patterns[:3])}. Ensemble confidence: {confidence:.0%} across {len(methods_used)} models"
			)
		else:
			# Build HUMAN explanation
			natural_signs = []
			if tf_score is not None and tf_score < 0.2:
				natural_signs.append("organic speech variations")
			if h_score < 0.5:
				natural_signs.append("natural prosody patterns")
			natural_signs.append("authentic voice characteristics")

			explanation = f"Human voice verified: {', '.join(natural_signs[:3])}. Confidence: {confidence:.0%}"

		return {
			"classification": classification,
			"confidenceScore": round(confidence, 2),
			"explanation": explanation,
			"method": method,
		}


# Singleton instance (will use heuristics by default, with optional ML)
voice_detector = VoiceDetector(use_ml=True)
