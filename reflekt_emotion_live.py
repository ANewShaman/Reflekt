"""
Reflekt Emotion Detection Engine - FER Implementation (v2.1, fixed)
-------------------------------------------------------------------
- FIX: Convert OpenCV BGR -> RGB before FER detection (critical).
- FIX: Robust FER import (fallback to fer.fer.FER).
- FIX: MTCNN availability check + fallback to OpenCV mode.
- Better face selection (highest-confidence face).
- Reliable real-time overlay and session logging.
- JSON / CSV / NDJSON exports.

Requires:
    pip install fer opencv-python numpy
Optional (for REST API):
    pip install flask
"""

from __future__ import annotations

import cv2
import numpy as np
import time
import json
import queue
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import deque

# ---- Import FER with friendly error handling and fallbacks
FER_CLASS = None
try:
    from fer import FER as _FER
    FER_CLASS = _FER
except Exception:
    try:
        # Some versions expose the class under fer.fer
        from fer.fer import FER as _FER
        FER_CLASS = _FER
    except Exception:
        print("FER not found or failed to import.")
        print("    Install with:  pip install fer")
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
        data = convert_to_native(data)
        return json.dumps(data, ensure_ascii=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        return convert_to_native(data)


@dataclass
class SessionMetadata:
    """Session-level metadata."""
    session_id: str
    start_time: float
    end_time: Optional[float]
    total_frames: int
    frames_analyzed: int
    average_fps: float
    config: dict
    emotion_summary: Dict[str, int]


# ============================================================================ #
# 2) ASYNC EMOTION ENGINE (Non-blocking with FER)
# ============================================================================ #

class AsyncReflektEmotionEngine:
    """
    Async emotion engine using FER library.
    FER runs in background thread, main thread never blocks.
    UI stays smooth at 60 FPS.
    """

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
            "frame_skip": 8,               # Process every Nth frame
            "smoothing_window": 4,         # Moving avg window
            "min_confidence": 0.30,        # Threshold on dominant score (0..1)
            "use_smoothed_output": True,   # Smooth valence/arousal
            "serve_api": False,            # REST API toggle
            "api_port": 5055,              # REST API port
            "fusion_weights": {"face": 0.7, "voice": 0.3},
            "mtcnn": True,                 # Use MTCNN for face detection (more accurate, slower, optional dep)
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
        self.latest_frame: Optional[EmotionFrame] = None

        # Voice state
        self.voice_last: Optional[Dict[str, float]] = None

        # Performance tracking
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "frames_processed": 0,
            "fps_history": deque(maxlen=30),
            "queue_drops": 0,
        }

        # Session
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time: float = time.time()

        # Async processing queues
        self.frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self.result_queue: queue.Queue = queue.Queue(maxsize=5)
        
        # Initialize FER detector
        self.detector: Optional[FER_CLASS] = None
        self._init_detector()
        
        # Worker thread
        self._worker_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        # Pre-warm FER in worker thread
        self._warmup()

    def _init_detector(self):
        """Initialize FER detector with error handling and MTCNN availability check."""
        mtcnn_requested = bool(self.config.get("mtcnn", False))
        mtcnn_ok = False
        if mtcnn_requested:
            try:
                import mtcnn  # noqa: F401
                mtcnn_ok = True
            except Exception:
                print("! MTCNN requested but not installed. Falling back to OpenCV mode.")
                mtcnn_ok = False

        try:
            self.detector = FER_CLASS(mtcnn=mtcnn_requested and mtcnn_ok)
            self.config["mtcnn"] = (mtcnn_requested and mtcnn_ok)
            print(f"✓ FER detector initialized (MTCNN: {self.config['mtcnn']})")
        except Exception as e:
            print(f"✗ Failed to initialize FER: {e}")
            print("  Falling back to non-MTCNN (OpenCV) mode...")
            self.detector = FER_CLASS(mtcnn=False)
            self.config["mtcnn"] = False
            print("✓ FER detector initialized (OpenCV mode)")

    def _warmup(self):
        """Pre-warm FER to avoid first-frame latency."""
        def warmup_task():
            try:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                # Draw a simple face-like pattern to help detector
                cv2.circle(dummy, (320, 240), 80, (255, 255, 255), -1)
                if self.detector:
                    # IMPORTANT: convert to RGB for FER
                    rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
                    _ = self.detector.detect_emotions(rgb)
            except Exception:
                pass
        
        threading.Thread(target=warmup_task, daemon=True).start()

    # --------------- Background Worker Thread ---------------- #

    def _worker_loop(self):
        """Background thread that processes frames from queue."""
        while self._worker_running:
            try:
                frame_bgr = self.frame_queue.get(timeout=0.15)
                if frame_bgr is None:  # Shutdown signal
                    break
                
                result = self._analyze_frame_blocking(frame_bgr)
                if result:
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        # Keep newest result
                        try:
                            _ = self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.result_queue.put_nowait(result)
                        except queue.Full:
                            pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                continue

    def _pick_face(self, results: List[dict]) -> Optional[dict]:
        """Pick the best face result: highest dominant probability."""
        if not results:
            return None
        best = None
        best_score = -1.0
        for r in results:
            emos = r.get("emotions", {})
            if not emos:
                continue
            dom, score = max(emos.items(), key=lambda x: float(x[1]))
            if score > best_score:
                best = r
                best_score = float(score)
        return best

    def _analyze_frame_blocking(self, frame_bgr: np.ndarray) -> Optional[EmotionFrame]:
        """
        The actual FER analysis (blocking, runs in worker thread).
        CRITICAL: Convert BGR -> RGB before calling FER.
        """
        if not self.detector:
            return self._handle_detection_failure()
            
        t0 = time.time()
        try:
            # Convert BGR (OpenCV) -> RGB (FER expects RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # FER returns list of detected faces with emotions
            # Format: [{'box': (x,y,w,h), 'emotions': {'angry': 0.01, ...}}]
            results = self.detector.detect_emotions(frame_rgb)
            if not results:
                return self._handle_detection_failure()
            
            face_data = self._pick_face(results)
            if not face_data:
                return self._handle_detection_failure()

            emotions = face_data.get('emotions', {})
            if not emotions:
                return self._handle_detection_failure()

            # Dominant emotion and confidence (0..1 -> %)
            dominant, conf01 = max(emotions.items(), key=lambda x: float(x[1]))
            confidence = float(conf01) * 100.0  # percent

            # Normalize emotion values to 0-100 scale
            all_emotions = {k: round(float(v) * 100.0, 2) for k, v in emotions.items()}

            valence, arousal = self.compute_valence_arousal(all_emotions)
            vibrancy = round(arousal ** 0.8, 3)

            quality = self._assess_quality(confidence)

            ef = EmotionFrame(
                timestamp=time.time(),
                frame_number=self.frame_count,
                dominant=dominant,
                confidence=round(confidence, 2),
                valence=valence,
                arousal=arousal,
                all_emotions=all_emotions,
                quality=quality,
                processing_time_ms=round((time.time() - t0) * 1000.0, 2),
                age=None,
                vibrancy=vibrancy,
            )

            # Confidence gate (min_confidence is 0..1 in config)
            if confidence < (self.config["min_confidence"] * 100.0) and ef.quality == "medium":
                ef.quality = "low"

            # Apply smoothing and fusion
            self.analyzed_count += 1
            self.emotion_history.append(ef)
            ef = self._maybe_smooth(ef)
            ef = self._fuse_modalities(ef)

            # Update state
            self.last_valid_emotion = ef
            self.latest_frame = ef
            self.performance_metrics["total_processing_time"] += (time.time() - t0) * 1000.0
            self.performance_metrics["frames_processed"] += 1

            return ef

        except Exception:
            return self._handle_detection_failure()

    # --------------- Public API (Non-blocking) ---------------- #

    def submit_frame(self, frame_bgr: np.ndarray) -> bool:
        """
        Submit frame for analysis (non-blocking).
        Returns True if frame was queued, False if queue full (frame dropped).
        """
        try:
            self.frame_queue.put_nowait(frame_bgr.copy())
            return True
        except queue.Full:
            self.performance_metrics["queue_drops"] += 1
            return False

    def get_latest_result(self) -> Optional[EmotionFrame]:
        """
        Get most recent analysis result without blocking.
        Drains queue to always return the freshest result.
        """
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    # --------------- Voice Fusion ---------------- #

    def update_voice(self, valence: float, arousal: float, dominant: str | None = None):
        """Update voice emotion state (call from audio thread)."""
        v = max(-1.0, min(1.0, float(valence)))
        a = max(0.0, min(1.0, float(arousal)))
        self.voice_last = {"valence": v, "arousal": a, "dominant": dominant}

    # --------------- Helper Methods ---------------- #

    def compute_valence_arousal(self, emotions: Dict[str, float]) -> Tuple[float, float]:
        """Convert emotion probabilities (0..100) to valence/arousal in [-1..1] and [0..1]."""
        v = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["valence"][e]
            for e in self.EMOTION_MAPS["valence"]
        )
        a = sum(
            (float(emotions.get(e, 0.0)) / 100.0) * self.EMOTION_MAPS["arousal"][e]
            for e in self.EMOTION_MAPS["arousal"]
        )
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
        """Apply moving average smoothing if enabled."""
        if not self.config["use_smoothed_output"] or len(self.emotion_history) < 2:
            return frame
        
        avg_v = float(np.mean([e.valence for e in self.emotion_history]))
        avg_a = float(np.mean([e.arousal for e in self.emotion_history]))
        frame.valence = round(avg_v, 3)
        frame.arousal = round(avg_a, 3)
        frame.vibrancy = round(frame.arousal ** 0.8, 3)
        return frame

    def _fuse_modalities(self, frame: EmotionFrame) -> EmotionFrame:
        """Blend face with voice emotion if available."""
        if not self.voice_last:
            return frame
        
        wf = float(self.config["fusion_weights"]["face"])
        wv = float(self.config["fusion_weights"]["voice"])
        fused_v = round(wf * frame.valence + wv * self.voice_last["valence"], 3)
        fused_a = round(wf * frame.arousal + wv * self.voice_last["arousal"], 3)
        frame.valence, frame.arousal = fused_v, fused_a
        frame.vibrancy = round(fused_a ** 0.8, 3)
        return frame

    def _handle_detection_failure(self) -> Optional[EmotionFrame]:
        """Return last valid reading with 'no_face' quality, else None."""
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
                age=None,
                vibrancy=prev.vibrancy,
            )
        return None

    # --------------- Metrics / Export ---------------- #

    def log_frame(self, ef: EmotionFrame):
        """Log frame to session history."""
        self.session_log.append(ef)

    def get_smoothed_emotion(self) -> Optional[Dict]:
        """Return smoothed summary."""
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
        """Get performance statistics."""
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
            "estimated_analysis_fps": round(est_fps, 1),
            "queue_drops": self.performance_metrics["queue_drops"],
        }

    def export_session(self, filepath: str | None = None, format: str = "json") -> str:
        """Export session data (json/csv/ndjson)."""
        out_dir = Path("reflekt_sessions")
        out_dir.mkdir(exist_ok=True)
        if filepath is None:
            filepath = out_dir / f"session_{self.session_id}.{format}"
        else:
            filepath = Path(filepath)

        # Summary
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
                    "valence", "arousal", "vibrancy",
                    "quality", "processing_time_ms"
                ])
                for f in self.session_log:
                    w.writerow([
                        f.timestamp, f.frame_number, f.dominant, f.confidence,
                        f.valence, f.arousal, f.vibrancy,
                        f.quality, f.processing_time_ms
                    ])

        elif format == "ndjson":
            with open(filepath, "w", encoding="utf-8") as fh:
                for f in self.session_log:
                    fh.write(f.to_json() + "\n")

        else:
            raise ValueError("format must be one of: json | csv | ndjson")

        return str(filepath)

    def shutdown(self):
        """Gracefully shutdown worker thread."""
        self._worker_running = False
        try:
            self.frame_queue.put(None, timeout=0.5)
        except Exception:
            pass
        try:
            self.worker_thread.join(timeout=2.0)
        except Exception:
            pass


# ============================================================================ #
# 3) LIVE CAPTURE (UI Loop)
# ============================================================================ #

class ReflektLiveCapture:
    """Handles camera capture, overlays, hotkeys, and graceful shutdown."""

    def __init__(self, engine: AsyncReflektEmotionEngine, camera_index: int = 0):
        self.engine = engine
        self.camera_index = camera_index
        self.cam: Optional[cv2.VideoCapture] = None
        self.running = False
        self.paused = False
        
        # Latest emotion for display (may lag behind processing)
        self.display_emotion: Optional[EmotionFrame] = None

    def start(self, headless: bool = False):
        """Start the live capture loop."""
        self.cam = cv2.VideoCapture(self.camera_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        # Set camera properties for better performance
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cam.set(cv2.CAP_PROP_FPS, 30)

        print(f"Reflekt Engine started (Session: {self.engine.session_id})")
        print(f"Config: frame_skip={self.engine.config['frame_skip']}, "
              f"smoothing={self.engine.config['smoothing_window']}, "
              f"mtcnn={self.engine.config['mtcnn']}")
        print("Async mode: UI thread never blocks on analysis")
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

                # Submit frame for async analysis (non-blocking)
                if self.engine.frame_count % self.engine.config["frame_skip"] == 0:
                    _ = self.engine.submit_frame(frame)

                # Get latest result (non-blocking)
                latest_result = self.engine.get_latest_result()
                if latest_result:
                    # Accept even 'no_face' (so timeline is visible), but overlay shows last valid
                    if latest_result.quality != "no_face":
                        self.display_emotion = latest_result
                    self.engine.log_frame(latest_result)
                    print(latest_result.to_json())

                # Draw UI (always smooth, never blocks)
                if not headless:
                    self._draw_overlay(frame, self.display_emotion)
                    cv2.imshow("Reflekt Live", frame)
                    key = cv2.waitKey(1) & 0xFF
                    self._handle_key(key)

                # FPS accounting
                if time.time() - fps_timer >= 1.0:
                    self.engine.performance_metrics["fps_history"].append(fps_counter)
                    fps_counter = 0
                    fps_timer = time.time()

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self.cleanup()

    def _handle_key(self, key: int):
        """Handle keyboard input."""
        if key == ord("q"):
            print("Exiting confession booth...")
            self.running = False
        elif key == ord("w"):  # Withdraw (pause/resume)
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
            print("\nReflection Metrics:")
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

            text = f"{emotion.dominant.upper()} ({emotion.confidence:.1f}%)"
            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            va_text = f"V:{emotion.valence:+.2f} A:{emotion.arousal:+.2f}"
            cv2.putText(frame, va_text, (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            vib_text = f"Vibrancy:{emotion.vibrancy:.2f}" if emotion.vibrancy is not None else ""
            if vib_text:
                cv2.putText(frame, vib_text, (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

            quality_color = {
                "high": (0, 255, 0),
                "medium": (0, 255, 255),
                "low": (0, 165, 255),
                "uncertain": (80, 80, 80),
                "no_face": (0, 0, 255)
            }
            cv2.circle(frame, (frame.shape[1] - 30, 30), 14,
                       quality_color.get(emotion.quality, (128, 128, 128)), -1)

        # UI FPS (not analysis FPS)
        if self.engine.performance_metrics["fps_history"]:
            avg_fps = float(np.mean(list(self.engine.performance_metrics["fps_history"])))
            cv2.putText(frame, f"UI FPS:{avg_fps:.1f}",
                        (frame.shape[1] - 160, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    def _draw_paused_overlay(self, frame: np.ndarray):
        """Draw 'Withdrawn' overlay."""
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
        """Cleanup resources and save session."""
        self.running = False
        if self.cam and self.cam.isOpened():
            self.cam.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        # Shutdown async engine
        self.engine.shutdown()
        
        path = self.engine.export_session()
        print(f"\nSession saved -> {path}")
        print(f"Total frames: {self.engine.frame_count}")
        print(f"Analyzed: {self.engine.analyzed_count}")
        print(f"Queue drops: {self.engine.performance_metrics['queue_drops']}")
        print("Reflekt session complete")


# ============================================================================ #
# 4) OPTIONAL: EMBEDDED REST API
# ============================================================================ #

def start_api_server(engine: AsyncReflektEmotionEngine, port: int = 5055):
    """Start REST API server in background thread."""
    try:
        from flask import Flask, jsonify
    except Exception:
        print("Flask not installed; API disabled. Install with: pip install flask")
        return

    app = Flask(__name__)

    @app.get("/emotion")
    def emotion():
        f = engine.latest_frame
        return jsonify(f.to_dict() if f else {"status": "warming_up"})

    @app.get("/metrics")
    def metrics():
        return jsonify(convert_to_native(engine.get_performance_metrics()))

    @app.get("/health")
    def health():
        return jsonify({
            "status": "running",
            "session_id": engine.session_id,
            "worker_alive": engine.worker_thread.is_alive(),
        })

    def _run():
        """Flask app runner (suppresses logs for cleaner output)."""
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print(f"✓ REST API running at http://localhost:{port}/emotion")
    print(f"  Endpoints: /emotion | /metrics | /health\n")


# ============================================================================ #
# 5) MAIN
# ============================================================================ #

def main():
    """
    Main entry point with optimized defaults for FER.
    """
    # Choose behavior profile
    mode = "fast"  # Options: "fast" | "balanced" | "quality"
    
    presets = {
        "fast": {
            "frame_skip": 10,
            "smoothing_window": 4,
            "mtcnn": False,  # OpenCV is faster
            "min_confidence": 0.25,
        },
        "balanced": {
            "frame_skip": 8,
            "smoothing_window": 5,
            "mtcnn": False,
            "min_confidence": 0.30,
        },
        "quality": {
            "frame_skip": 6,
            "smoothing_window": 6,
            "mtcnn": True,  # MTCNN is more accurate (if installed)
            "min_confidence": 0.35,
        },
    }

    config = {
        "use_smoothed_output": True,
        "serve_api": True,           # Enable REST API
        "api_port": 5055,
        "fusion_weights": {"face": 0.7, "voice": 0.3},
    }
    config.update(presets.get(mode, presets["fast"]))

    print("=" * 70)
    print("  Reflekt Emotion Engine v2.1 - FER Implementation (fixed)")
    print("=" * 70)
    print(f"  Mode:              {mode.upper()}")
    print(f"  Architecture:      Non-blocking async (worker thread)")
    print(f"  Backend:           FER (Facial Expression Recognition)")
    print(f"  Face Detection:    {'MTCNN' if config['mtcnn'] else 'OpenCV Haar Cascade'}")
    print(f"  Frame Skip:        1:{config['frame_skip']}")
    print(f"  Smoothing Window:  {config['smoothing_window']} frames")
    print(f"  Min Confidence:    {config['min_confidence']:.0%}")
    print("=" * 70)
    print()

    # Create async engine
    try:
        engine = AsyncReflektEmotionEngine(config=config)
    except Exception as e:
        print(f"✗ Failed to initialize engine: {e}")
        return

    # OPTIONAL: Voice integration example
    # def voice_worker():
    #     while True:
    #         valence, arousal = analyze_voice_from_mic()
    #         engine.update_voice(valence=valence, arousal=arousal, dominant="tense")
    #         time.sleep(0.1)
    # threading.Thread(target=voice_worker, daemon=True).start()

    # Start REST API (non-blocking)
    if config.get("serve_api", False):
        start_api_server(engine, port=config.get("api_port", 5055))

    # Start capture UI loop
    capture = ReflektLiveCapture(engine, camera_index=0)
    
    try:
        capture.start(headless=False)
    except KeyboardInterrupt:
        print("\n✓ Graceful shutdown initiated...")
    except Exception as e:
        print(f"\n✗ Error during capture: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        print("✓ Cleaning up resources...")
        if hasattr(capture, 'running') and capture.running:
            capture.cleanup()
        else:
            engine.shutdown()
            if capture.cam and capture.cam.isOpened():
                capture.cam.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("✓ Shutdown complete")


if __name__ == "__main__":
    main()
