"""
Audio Processing Module (CNN Version)
- Base64 decoding
- Audio conversion (mp3 / wav)
- MFCC extraction for CNN
"""

import base64
import io
import tempfile
import os
from typing import Tuple

import numpy as np
import librosa
from pydub import AudioSegment


class AudioProcessor:
    # cnn parameters
    SAMPLE_RATE = 16000
    N_MFCC = 40
    N_FFT = 1024
    HOP_LENGTH = 160        # 10 ms hop
    MAX_LEN = 300           # time frames (~3 sec)
    MIN_AUDIO_SEC = 0.5

    #  Base64 
    def decode_base64_audio(self, audio_base64: str) -> bytes:
        try:
            if "," in audio_base64:
                audio_base64 = audio_base64.split(",")[1]
            return base64.b64decode(audio_base64)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")

    #  Audio Conversion 
    def convert_audio_to_samples(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        temp_path = None
        try:
            is_wav = audio_bytes[:4] in [b"RIFF", b"riff"]
            suffix = ".wav" if is_wav else ".mp3"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(audio_bytes)
                temp_path = f.name

            # Try librosa
            try:
                audio, sr = librosa.load(
                    temp_path, sr=self.SAMPLE_RATE, mono=True
                )
                return audio.astype(np.float32), sr
            except Exception:
                pass

            # Try soundfile
            try:
                import soundfile as sf
                audio, sr = sf.read(temp_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != self.SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
                return audio.astype(np.float32), self.SAMPLE_RATE
            except Exception:
                pass

            # Fallback: pydub (needs ffmpeg)
            audio = AudioSegment.from_file(temp_path)
            buf = io.BytesIO()
            audio.export(buf, format="wav")
            buf.seek(0)
            audio, sr = librosa.load(buf, sr=self.SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), sr

        except Exception as e:
            raise ValueError(f"Audio conversion failed: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # CNN MFCC
    def extract_mfcc_cnn(self, audio: np.ndarray, sr: int) -> np.ndarray:
        min_len = int(self.MIN_AUDIO_SEC * sr)
        if len(audio) < min_len:
            raise ValueError("Audio too short for detection")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.N_MFCC,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH
        )

        # NORMALIZATION (MANDATORY)
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        # Pad / trim time axis
        if mfcc.shape[1] < self.MAX_LEN:
            pad = self.MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
        else:
            mfcc = mfcc[:, :self.MAX_LEN]

        # Shape → (1, 40, T)
        return mfcc[np.newaxis, :, :].astype(np.float32)

    def process_audio_for_cnn(self, audio_base64: str) -> np.ndarray:
        """Process base64 audio and return MFCC features for CNN."""
        audio_bytes = self.decode_base64_audio(audio_base64)
        audio, sr = self.convert_audio_to_samples(audio_bytes)
        mfcc = self.extract_mfcc_cnn(audio, sr)
        return mfcc
    
    def process_audio_with_samples(self, audio_base64: str) -> Tuple[dict, np.ndarray, int]:
        """Process audio and return features dict, raw samples, and sample rate."""
        audio_bytes = self.decode_base64_audio(audio_base64)
        audio, sr = self.convert_audio_to_samples(audio_bytes)
        
        # Extract basic features for heuristic detection
        features = {
            'pitch_std': 50.0,  # Placeholder
            'pitch_range': 200.0,
            'spectral_centroid_std': 500.0,
            'rms_std': 0.05,
            'zcr_std': 0.05,
            'voiced_ratio': 0.5,
            'mfcc_0_std': 100.0,
            'delta_mfcc_0_std': 1.0,
            'hf_energy_ratio': 0.1,
        }
        
        return features, audio, sr


# Singleton (use everywhere)
audio_processor = AudioProcessor()
