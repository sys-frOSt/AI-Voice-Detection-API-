"""
Transformers Pipeline Detector
Uses local transformers pipeline for deepfake detection.
Model: mo-thecreator/Deepfake-audio-detection (99%+ accuracy on gTTS)
"""
import os
import numpy as np
from typing import Dict, Any, Optional

class TransformersDetector:
    """
    Detects AI voices using local transformers pipeline.
    Works on CPU (slow) or GPU (fast).
    """
    
    MODEL_ID = "mo-thecreator/Deepfake-audio-detection"
    MAX_DURATION_SECONDS = 30  # Limit audio to 30 seconds to avoid memory issues
    
    def __init__(self):
        self.pipe = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the model (lazy loading to avoid startup delay)."""
        if self.is_loaded:
            return
            
        try:
            import os
            # Force CPU to avoid MPS memory issues on Mac
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            
            from transformers import pipeline
            import torch
            
            # Determine device - force CPU for stability
            device = "cpu"  # Force CPU to avoid MPS memory issues
            
            print(f"Loading deepfake detection model: {self.MODEL_ID} (device: {device})...")
            self.pipe = pipeline(
                'audio-classification', 
                model=self.MODEL_ID, 
                trust_remote_code=True,
                device=device
            )
            self.is_loaded = True
            print("✓ Deepfake detection model loaded")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_loaded = False
        
    def detect(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Detect if audio is AI-generated.
        
        Args:
            audio: Audio samples (numpy array, should be 16kHz)
            sr: Sample rate
            
        Returns:
            Detection result dictionary
        """
        if not self.is_loaded:
            self.load_model()
            
        if not self.is_loaded or self.pipe is None:
            return {
                'classification': 'UNKNOWN',
                'confidenceScore': 0.0,
                'explanation': 'Model failed to load',
                'method': 'transformers_failed'
            }
            
        try:
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Truncate to max duration to avoid memory issues
            max_samples = self.MAX_DURATION_SECONDS * sr
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Run inference
            result = self.pipe(audio)
            
            # Parse result: [{'score': 0.99, 'label': 'fake'}, {'score': 0.01, 'label': 'real'}]
            if not result:
                return {
                    'classification': 'UNKNOWN',
                    'confidenceScore': 0.0,
                    'explanation': 'No result from model',
                    'method': 'transformers_failed'
                }
            
            # Find fake/spoof score
            fake_score = 0.0
            real_score = 0.0
            
            for item in result:
                label = item['label'].lower()
                score = item['score']
                if label in ['fake', 'spoof', 'deepfake']:
                    fake_score = score
                elif label in ['real', 'bonafide', 'genuine']:
                    real_score = score
            
            # Determine classification
            if fake_score > real_score:
                classification = "AI_GENERATED"
                confidence = fake_score
                explanation = "Deep learning model detected synthetic speech patterns"
            else:
                classification = "HUMAN"
                confidence = real_score
                explanation = "Deep learning model confirmed natural human voice"
            
            return {
                'classification': classification,
                'confidenceScore': round(float(confidence), 4),
                'explanation': explanation,
                'method': 'transformers_pipeline'
            }
                
        except Exception as e:
            print(f"Transformers detection error: {e}")
            return {
                'classification': 'UNKNOWN',
                'confidenceScore': 0.0,
                'explanation': f'Detection error: {str(e)[:50]}',
                'method': 'transformers_error'
            }

# Singleton instance (lazy loaded)
transformers_detector = TransformersDetector()