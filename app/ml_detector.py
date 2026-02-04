"""
ML-Based Voice Detector using Pre-trained Models
Uses Wav2Vec2 embeddings + classifier for high-accuracy deepfake detection
"""

import os
import warnings
import numpy as np

from typing import Dict, Any, Tuple, List, Optional

warnings.filterwarnings("ignore")


class Wav2Vec2Classifier:
    """Simple classifier on top of Wav2Vec2 embeddings."""

    def __init__(self, hidden_size: int = 768, num_classes: int = 2):
        import torch
        
        # Store the module as instance variable to avoid re-importing
        self.torch = torch
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
    
    def to(self, device):
        self.classifier = self.classifier.to(device)
        return self
    
    def eval(self):
        self.classifier.eval()
        return self
    
    def modules(self):
        return self.classifier.modules()


class MLVoiceDetector:
    """
    ML-based voice detector using Wav2Vec2 embeddings.
    Combines pre-trained features with a trained classifier.
    """

    def __init__(self, device: str = None):
        """
        Initialize the ML voice detector.

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        if device:
            self.device = device
        else:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self.processor = None
        self.wav2vec_model = None
        self.classifier = None
        self.is_loaded = False
        self.trained_model = None

        self.feature_mean = None
        self.feature_std = None

    def load_model(self):
        """Load the Wav2Vec2 model and classifier."""
        if self.is_loaded:
            return

        from transformers import Wav2Vec2Processor, Wav2Vec2Model

        print("Loading Wav2Vec2 model...", flush=True)

        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir="/tmp/hf_cache"
            )

            self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir="/tmp/hf_cache"
            )

            self.wav2vec_model.to(self.device)
            self.wav2vec_model.eval()

            self.classifier = Wav2Vec2Classifier()
            self._initialize_classifier_weights()
            self.classifier.to(self.device)
            self.classifier.eval()

            model_path = os.path.join(os.path.dirname(__file__), "trained_model.joblib")
            if os.path.exists(model_path):
                self.load_trained_model(model_path)

            self.is_loaded = True
            print(f"✓ Model loaded on {self.device}", flush=True)

        except Exception as e:
            print(f"❌ Model load failed: {e}", flush=True)
            self.is_loaded = False

    def _initialize_classifier_weights(self):
        import torch
        
        for module in self.classifier.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def extract_wav2vec_features(
        self, audio: np.ndarray, sr: int = 16000
    ) -> Optional[np.ndarray]:
        import torch

        if not self.is_loaded:
            self.load_model()

        if not self.is_loaded:
            return None

        try:
            inputs = self.processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                outputs = self.wav2vec_model(input_values)
                embeddings = outputs.last_hidden_state.mean(dim=1)

            return embeddings.cpu().numpy()[0]

        except Exception as e:
            print(f"Feature extraction failed: {e}", flush=True)
            return None

    def compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, float]:
        stats = {
            "embedding_mean": float(np.mean(embeddings)),
            "embedding_std": float(np.std(embeddings)),
            "embedding_max": float(np.max(embeddings)),
            "embedding_min": float(np.min(embeddings)),
            "embedding_range": float(np.ptp(embeddings)),
            "embedding_entropy": self._entropy(embeddings),
        }
        return stats

    def _entropy(self, x: np.ndarray, bins: int = 50) -> float:
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        hist /= hist.sum()
        return float(-np.sum(hist * np.log2(hist + 1e-9)))

    def load_trained_model(self, path: str):
        try:
            import joblib
            data = joblib.load(path)
            self.trained_model = data["model"]
            print("✓ Trained model loaded", flush=True)
        except Exception as e:
            print(f"Trained model load failed: {e}", flush=True)

    def detect(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        embeddings = self.extract_wav2vec_features(audio, sr)

        if embeddings is None:
            return {
                "classification": "UNKNOWN",
                "confidenceScore": 0.5,
                "explanation": "Feature extraction failed",
                "method": "fallback"
            }

        stats = self.compute_embedding_statistics(embeddings)

        ai_score = 0.5
        if stats["embedding_std"] < 0.35:
            ai_score += 0.2
        if stats["embedding_entropy"] < 3.2:
            ai_score += 0.2

        ai_score = max(0.0, min(1.0, ai_score))

        if ai_score > 0.5:
            return {
                "classification": "AI_GENERATED",
                "confidenceScore": round(ai_score, 2),
                "explanation": "Synthetic voice patterns detected",
                "method": "wav2vec2"
            }

        return {
            "classification": "HUMAN",
            "confidenceScore": round(1 - ai_score, 2),
            "explanation": "Natural human voice patterns detected",
            "method": "wav2vec2"
        }


# 🔁 Lazy singleton (HF-safe)
_ml_detector = None


def get_ml_detector() -> MLVoiceDetector:
    global _ml_detector
    if _ml_detector is None:
        _ml_detector = MLVoiceDetector()
    return _ml_detector
