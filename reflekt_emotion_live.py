"""
Reflekt Emotion Detection Engine - Production Ready (v1.2)
==========================================================
- DeepFace webcam emotion analysis (face)
- Age + arousal -> vibrancy
- Confidence gating, smoothing, frame skipping
- Voice fusion hook (call update_voice(...) from a mic thread later)
- Optional REST API (/emotion) for front-end canvases
- JSON / CSV / NDJSON session exports

Requires:
    pip install deepface opencv-python numpy
Optional (for REST API):
    pip install flask
"""

from __future__ import annotations

import cv2
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import threading

# ---- Try to import DeepFace early and be friendly if missing
try:
    from deepface import DeepFace
except Exception as exc:
    print("DeepFace not found or failed to import.")
    print("    Install with:  pip install deepface")
    raise

# ============================================================================ #
# NUMPY TYPE CONVERSION HELPER
# ============================================================================ #

def convert_to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj

# ============================================================================ #
# 1) DATA STRUCTURES
# ============================================================================ #

@dataclass
class EmotionFrame:
    """Standardized emotion packet (one analyzed moment)."""
    timestamp: float
    frame_number: int
    dominant: str
    confidence: float
    valence: float
    arousal: float
    all_emotions: Dict[str, float]
    quality: str  # 'high' | 'medium' | 'low' | 'uncertain' | 'no_face'
    processing_time_ms: float
    age: Optional[int] = None
    vibrancy: Optional[float] = None

    def to_json(self) -> str:
        data = asdict(self)
        # Convert all numpy types to native Python types
        data = convert_to_native(data)
        return json.dumps(data, ensure_ascii=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Convert all numpy types to native Python types
        return convert_to_native(data)


@dataclass
class SessionMetadata:
    """Session-level metadata (nice for archive / chain)."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_frames: int
    frames_analyzed: int
    average_fps: float
    config: dict
    emotion_summary: Dict[str, int]


# ============================================================================ #
# 2) CORE ENGINE
# ============================================================================ #

class ReflektEmotionEngine:
    """
    Emotion engine using DeepFace.
    - Call analyze_frame(frame) to get EmotionFrame
    - The engine stores history for smoothing and session export
    """

    # Mappings from categorical emotions -> valence/arousal contributions
    EMOTION_MAPS = {
        "valence": {
            "angry": -0.7, "disgust": -0.6, "fear": -0.8,
            "happy": 0.9, "sad": -0.8, "surprise": 0.4, "neutral": 0.0
        },
        "arousal": {
            "angry": 0.8, "disgust": 0.3, "fear": 0.9,
            "happy": 0.7, "sad": -0.5, "surprise": 0.9, "neutral": 0.0
        }
    }

    def __init__(self, config: dict | None = None):
        # Config defaults
        self.config = {
            "frame_skip": 5,               # process every Nth frame
            "smoothing_window": 3,         # moving avg window
            "min_confidence": 0.30,        # 0.0..1.0 (DeepFace gives %; we compare *100)
            "backend": "opencv",           # 'opencv' fastest, 'retinaface' higher quality
            "enforce_detection": False,    # don't crash if no face
            "use_smoothed_output": True,   # smooth valence/arousal out
            "serve_api": False,            # spin up REST API
            "api_port": 5055,              # REST API port
            "fusion_weights": {"face": 0.7, "voice": 0.3}  # face/voice blending
        }
        if config:
            self.config.update(config)

        # State
        self.frame_count: int = 0
        self.analyzed_count: int = 0
        self.session_log: List[EmotionFrame] = []
        self.emotion_history: deque[EmotionFrame] = deque(
            maxlen=self.config["smoothing_window"]
        )
        self.last_valid_emotion: Optional[EmotionFrame] = None
        self.latest_frame: Optional[EmotionFrame] = None  # for API

        # Voice state (set via update_voice)
        self.voice_last: Optional[Dict[str, float]] = None

        # Performance tracking
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "frames_processed": 0,
            "fps_history": deque(maxlen=30),
        }

        # Session
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: float = time.time()

        # Pre-warm DeepFace (optional: avoid first-frame latency spikes)
        try:
            _ = DeepFace.analyze(
                np.zeros((64, 64, 3), dtype=np.uint8),
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self.config["backend"],
                silent=True,
            )
        except Exception:
            pass

    # --------------- Voice fusion hook ---------------- #

    def update_voice(self, valence: float, arousal: float, dominant: str | None = None):
        """
        Call this periodically from your voice thread with computed voice V/A.
        Values should be in valence [-1,1], arousal [0,1].
        """
        # clamp
        v = max(-1.0, min(1.0, float(valence)))
        a = max(0.0, min(1.0, float(arousal)))
        self.voice_last = {"valence": v, "arousal": a, "dominant": dominant}

    # --------------- Core compute helpers ---------------- #

    def compute_valence_arousal(self, emotions: Dict[str, float]) -> Tuple[float, float]:
        """
        Convert DeepFace emotion probabilities to valence/arousal.
        DeepFace returns a % dict like {'happy': 92.1, ...}
        """
        # Weighted sums of normalized (0..1) probabilities
        v = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["valence"][e]
            for e in self.EMOTION_MAPS["valence"]
        )
        a = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["arousal"][e]
            for e in self.EMOTION_MAPS["arousal"]
        )
        # Clamp for sanity
        v = max(-1.0, min(1.0, v))
        a = max(0.0, min(1.0, a))
        return round(v, 3), round(a, 3)

    def _assess_quality(self, confidence_percent: float) -> str:
        if confidence_percent > 70:
            return "high"
        elif confidence_percent > 40:
            return "medium"
        elif confidence_percent > 20:
            return "low"
        else:
            return "uncertain"

    def _maybe_smooth(self, frame: EmotionFrame) -> EmotionFrame:
        """Replace frame's valence/arousal with moving average if enabled."""
        if not self.config["use_smoothed_output"] or len(self.emotion_history) < 2:
            return frame
        avg_v = float(np.mean([e.valence for e in self.emotion_history]))
        avg_a = float(np.mean([e.arousal for e in self.emotion_history]))
        frame.valence = round(avg_v, 3)
        frame.arousal = round(avg_a, 3)
        # recompute vibrancy
        if frame.age is not None:
            frame.vibrancy = round((frame.arousal ** 0.8) * (1 - (min(frame.age, 70) / 100)), 3)
        else:
            frame.vibrancy = round(frame.arousal ** 0.8, 3)
        return frame

    def _fuse_modalities(self, frame: EmotionFrame) -> EmotionFrame:
        """Blend face val/arousal with last voice val/arousal if available."""
        if not self.voice_last:
            return frame
        wf = float(self.config["fusion_weights"]["face"])
        wv = float(self.config["fusion_weights"]["voice"])
        fused_v = round(wf * frame.valence + wv * self.voice_last["valence"], 3)
        fused_a = round(wf * frame.arousal + wv * self.voice_last["arousal"], 3)
        frame.valence, frame.arousal = fused_v, fused_a
        # update vibrancy after fusion
        if frame.age is not None:
            frame.vibrancy = round((fused_a ** 0.8) * (1 - (min(frame.age, 70) / 100)), 3)
        else:
            frame.vibrancy = round(fused_a ** 0.8, 3)
        return frame

    def _handle_detection_failure(self) -> Optional[EmotionFrame]:
        """Return last valid reading with 'no_face' quality."""
        if self.last_valid_emotion:
            prev = self.last_valid_emotion
            return EmotionFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                dominant=prev.dominant,
                confidence=0.0,
                valence=prev.valence,
                arousal=prev.arousal,
                all_emotions=prev.all_emotions,
                quality="no_face",
                processing_time_ms=0.0,
                age=prev.age,
                vibrancy=prev.vibrancy,
            )
        return None

    # --------------- Main analysis ---------------- #

    def analyze_frame(self, frame: np.ndarray) -> Optional[EmotionFrame]:
        """
        Analyze a BGR OpenCV frame -> EmotionFrame or None
        """
        t0 = time.time()
        try:
            raw = DeepFace.analyze(
                frame,
                actions=["emotion", "age"],
                enforce_detection=self.config["enforce_detection"],
                detector_backend=self.config["backend"],
                silent=True,
            )
            res = raw[0] if isinstance(raw, list) else raw
            emotions = res.get("emotion", {})
            if not emotions:
                return self._handle_detection_failure()

            dominant = res.get("dominant_emotion", "neutral")
            confidence = float(emotions.get(dominant, 0.0))
            age = res.get("age")
            age = int(age) if isinstance(age, (int, float, np.integer, np.floating)) else None

            valence, arousal = self.compute_valence_arousal(emotions)
            # vibrancy: arousal^0.8 damped by age
            if age is not None:
                vibrancy = round((arousal ** 0.8) * (1 - (min(age, 70) / 100)), 3)
            else:
                vibrancy = round(arousal ** 0.8, 3)

            quality = self._assess_quality(confidence)

            ef = EmotionFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                dominant=dominant,
                confidence=round(confidence, 2),
                valence=valence,
                arousal=arousal,
                all_emotions={k: round(float(v), 2) for k, v in emotions.items()},
                quality=quality,
                processing_time_ms=round((time.time() - t0) * 1000.0, 2),
                age=age,
                vibrancy=vibrancy,
            )

            # confidence gate: mark 'low' explicitly if below threshold
            if confidence < (self.config["min_confidence"] * 100.0) and ef.quality == "medium":
                ef.quality = "low"

            # history & smoothing
            self.analyzed_count += 1
            self.emotion_history.append(ef)
            ef = self._maybe_smooth(ef)

            # modality fusion
            ef = self._fuse_modalities(ef)

            # state
            self.last_valid_emotion = ef
            self.latest_frame = ef
            self.performance_metrics["total_processing_time"] += (time.time() - t0) * 1000.0
            self.performance_metrics["frames_processed"] += 1

            return ef

        except Exception:
            return self._handle_detection_failure()

    # --------------- Metrics / Export ---------------- #

    def get_smoothed_emotion(self) -> Optional[Dict]:
        """Return smoothed summary (dominant, mean val/aro)."""
        if len(self.emotion_history) < 2:
            return None
        avg_v = float(np.mean([e.valence for e in self.emotion_history]))
        avg_a = float(np.mean([e.arousal for e in self.emotion_history]))
        doms = [e.dominant for e in self.emotion_history]
        dominant = max(set(doms), key=doms.count)
        return {
            "dominant": dominant,
            "valence": round(avg_v, 3),
            "arousal": round(avg_a, 3),
            "window_size": len(self.emotion_history),
        }

    def get_performance_metrics(self) -> Dict:
        fp = self.performance_metrics["frames_processed"]
        if fp == 0:
            return {"status": "no_data"}
        avg_ms = self.performance_metrics["total_processing_time"] / fp
        est_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        return {
            "frames_analyzed": self.analyzed_count,
            "total_frames": self.frame_count,
            "skip_ratio": f"1:{self.config['frame_skip']}",
            "avg_processing_ms": round(avg_ms, 2),
            "estimated_fps": round(est_fps, 1),
        }

    def log_frame(self, ef: EmotionFrame):
        self.session_log.append(ef)

    def export_session(self, filepath: str | None = None, format: str = "json") -> str:
        """
        Export the full session:
            - json: metadata + frames + perf
            - csv:  compact table
            - ndjson: one line per frame (stream-friendly)
        """
        out_dir = Path("reflekt_sessions")
        out_dir.mkdir(exist_ok=True)
        if filepath is None:
            filepath = out_dir / f"session_{self.session_id}.{format}"

        # Summaries
        counts: Dict[str, int] = {}
        for f in self.session_log:
            counts[f.dominant] = counts.get(f.dominant, 0) + 1

        elapsed = max(1e-6, (time.time() - self.start_time))
        metadata = SessionMetadata(
            session_id=self.session_id,
            start_time=self.start_time,
            end_time=time.time(),
            total_frames=self.frame_count,
            frames_analyzed=self.analyzed_count,
            average_fps=self.analyzed_count / elapsed,
            config=self.config,
            emotion_summary=counts,
        )

        if format == "json":
            export_data = {
                "metadata": convert_to_native(asdict(metadata)),
                "frames": [f.to_dict() for f in self.session_log],
                "performance": convert_to_native(self.get_performance_metrics()),
            }
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(export_data, fh, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv
            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([
                    "timestamp", "frame_number", "dominant", "confidence",
                    "valence", "arousal", "vibrancy", "age",
                    "quality", "processing_time_ms"
                ])
                for f in self.session_log:
                    w.writerow([
                        f.timestamp, f.frame_number, f.dominant, f.confidence,
                        f.valence, f.arousal, f.vibrancy, f.age,
                        f.quality, f.processing_time_ms
                    ])

        elif format == "ndjson":
            with open(filepath, "w", encoding="utf-8") as fh:
                for f in self.session_log:
                    fh.write(f.to_json() + "\n")

        else:
            raise ValueError("format must be one of: json | csv | ndjson")

        return str(filepath)


# ============================================================================ #
# 3) LIVE CAPTURE (UI LOOP)
# ============================================================================ #

class ReflektLiveCapture:
    """Handles camera capture, overlays, hotkeys, and graceful shutdown."""

    def __init__(self, engine: ReflektEmotionEngine, camera_index: int = 0):
        self.engine = engine
        self.camera_index = camera_index
        self.cam: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False

    def start(self, headless: bool = False):
        self.cam = cv2.VideoCapture(self.camera_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        print(f"Reflekt Engine started (Session: {self.engine.session_id})")
        print(f"Config: frame_skip={self.engine.config['frame_skip']}, "
              f"smoothing={self.engine.config['smoothing_window']}, "
              f"backend={self.engine.config['backend']}")
        print("Controls: Q quit | W withdraw/reveal | S archive | M metrics\n")

        self.running = True
        fps_timer = time.time()
        fps_counter = 0

        try:
            while self.running:
                ret, frame = self.cam.read()
                if not ret:
                    print("Frame grab failed; retrying...")
                    time.sleep(0.05)
                    continue

                if self.paused:
                    self._draw_paused_overlay(frame)
                    if not headless:
                        cv2.imshow("Reflekt Live", frame)
                        key = cv2.waitKey(1) & 0xFF
                        self._handle_key(key)
                    time.sleep(0.05)
                    continue

                self.engine.frame_count += 1
                fps_counter += 1

                ef: Optional[EmotionFrame] = None
                if self.engine.frame_count % self.engine.config["frame_skip"] == 0:
                    ef = self.engine.analyze_frame(frame)
                    if ef:
                        self.engine.log_frame(ef)
                        print(ef.to_json())

                if not headless:
                    self._draw_overlay(frame, ef)
                    cv2.imshow("Reflekt Live", frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)

                # FPS accounting
                if time.time() - fps_timer >= 1.0:
                    self.engine.performance_metrics["fps_history"].append(fps_counter)
                    fps_counter = 0
                    fps_timer = time.time()

        except KeyboardInterrupt:
            print("Interrupted by user")

        finally:
            self.cleanup()

    def _handle_key(self, key: int):
        if key == ord("q"):
            print("Exiting confession booth...")
            self.running = False
        elif key == ord("w"):  # W = Withdraw (pause/resume)
            self.paused = not self.paused
            if self.paused:
                print("You have withdrawn from the mirror.")
            else:
                print("The mirror clears... reflection resumes.")
        elif key == ord("s"):
            path = self.engine.export_session()
            print(f"Confession archived -> {path}")
        elif key == ord("m"):
            metrics = self.engine.get_performance_metrics()
            print("Reflection Metrics:")
            for k, v in metrics.items():
                print(f"   {k}: {v}")
            print()

    def _draw_overlay(self, frame: np.ndarray, emotion: Optional[EmotionFrame]):
        """Draw emotion overlay on frame."""
        if emotion:
            colors = {
                "happy": (0, 255, 0),
                "sad": (255, 100, 0),
                "angry": (0, 0, 255),
                "surprise": (255, 0, 255),
                "fear": (128, 0, 128),
                "disgust": (0, 165, 255),
                "neutral": (128, 128, 128)
            }
            color = colors.get(emotion.dominant, (255, 255, 255))

            text = f"{emotion.dominant.upper()} ({emotion.confidence}%)"
            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            va_text = f"V:{emotion.valence:+.2f} A:{emotion.arousal:+.2f}"
            cv2.putText(frame, va_text, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vib_text = f"Vibrancy:{emotion.vibrancy:.2f}" if emotion.vibrancy is not None else ""
            if vib_text:
                cv2.putText(frame, vib_text, (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            if emotion.age is not None:
                age_text = f"Age:{emotion.age}"
                cv2.putText(frame, age_text, (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            quality_color = {
                "high": (0, 255, 0),
                "medium": (0, 255, 255),
                "low": (0, 165, 255),
                "uncertain": (80, 80, 80),
                "no_face": (0, 0, 255)
            }
            cv2.circle(frame, (frame.shape[1] - 30, 30), 14,
                       quality_color.get(emotion.quality, (128, 128, 128)), -1)

        if self.engine.performance_metrics["fps_history"]:
            avg_fps = float(np.mean(list(self.engine.performance_metrics["fps_history"])))
            cv2.putText(frame, f"FPS:{avg_fps:.1f}",
                        (frame.shape[1] - 140, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    def _draw_paused_overlay(self, frame: np.ndarray):
        """Draw 'Withdrawn' overlay to make pause feel symbolic."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        text = "WITHDRAWN"
        subtext = "Press W to return to reflection"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        (sw, sh), _ = cv2.getTextSize(subtext, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        x = (frame.shape[1] - tw) // 2
        y = (frame.shape[0] // 2) - 10
        sx = (frame.shape[1] - sw) // 2
        sy = y + 50

        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (240, 240, 240), 4)
        cv2.putText(frame, subtext, (sx, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)

    def cleanup(self):
        self.running = False
        if self.cam:
            self.cam.release()
        cv2.destroyAllWindows()
        path = self.engine.export_session()
        print(f"Session saved -> {path}")
        print(f"Total frames: {self.engine.frame_count}")
        print(f"Analyzed: {self.engine.analyzed_count}")
        print("Reflekt session complete")


# ============================================================================ #
# 4) OPTIONAL: EMBEDDED REST API (no extra file)
# ============================================================================ #

def start_api_server(engine: ReflektEmotionEngine, port: int = 5055):
    try:
        from flask import Flask, jsonify
    except Exception:
        print("Flask not installed; API disabled. Install with: pip install flask")
        return

    app = Flask(__name__)

    @app.get("/emotion")
    def emotion():
        f = engine.latest_frame
        return jsonify(f.to_dict() if f else {"status": "warming"})

    @app.get("/metrics")
    def metrics():
        return jsonify(convert_to_native(engine.get_performance_metrics()))

    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"REST API running at http://localhost:{port}/emotion")


# ============================================================================ #
# 5) MAIN
# ============================================================================ #

def main():
    # Choose behavior profile
    mode = "fast"  # change to "quality" for heavier detection
    presets = {
        "fast":    {"frame_skip": 6, "smoothing_window": 3, "backend": "opencv"},
        "quality": {"frame_skip": 2, "smoothing_window": 5, "backend": "retinaface"},
    }

    config = {
        "min_confidence": 0.30,
        "use_smoothed_output": True,
        "serve_api": True,       # turn on REST API for your web canvas
        "api_port": 5055,
        "fusion_weights": {"face": 0.7, "voice": 0.3},
    }
    config.update(presets[mode])

    engine = ReflektEmotionEngine(config=config)

    # OPTIONAL: if you already have a voice thread, it should periodically call:
    # engine.update_voice(valence=0.1, arousal=0.6, dominant="tense")

    # Start REST API (non-blocking)
    if config.get("serve_api", False):
        start_api_server(engine, port=config.get("api_port", 5055))

    # Start capture UI loop
    capture = ReflektLiveCapture(engine, camera_index=0)
    capture.start(headless=False)


if __name__ == "__main__":
    main()