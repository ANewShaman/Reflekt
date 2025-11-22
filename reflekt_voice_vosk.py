"""
Reflekt Voice Emotion Module - VOSK Edition (v1.2 - ENHANCED DEBUGGING)
------------------------------------------------------------------------
Fixed microphone capture with comprehensive diagnostics.

Requirements:
    pip install vosk sounddevice
    Download a VOSK model: https://alphacephei.com/vosk/models
"""

import json
import queue
import threading
import time
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer


class ReflektVoiceVOSK:
    """
    Voice emotion module with enhanced debugging and diagnostics.
    """

    def __init__(self, engine=None, model_path=None, sample_rate=16000):
        # Auto-detect model path
        if model_path is None:
            model_path = self._find_vosk_model()
            if model_path is None:
                print("No VOSK model found in current directory")
                model_path = "vosk-model-small-en-us-0.15"  # fallback
        self.engine = engine
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.running = False
        
        # Initialize voice text attribute on engine
        if self.engine:
            self.engine.voice_last_text = ""

        # Queue for raw microphone audio
        self.audio_q = queue.Queue()

        # Debug counters
        self.audio_chunks_received = 0
        self.words_recognized = 0
        self.last_audio_level = 0.0

        # Emotional keyword → VA mapping
        self.keyword_map = {
            "tired":      (-0.3, -0.2),
            "sad":        (-0.6, -0.3),
            "cry":        (-0.7, +0.2),
            "crying":     (-0.7, +0.2),
            "angry":      (-0.7, +0.6),
            "mad":        (-0.7, +0.6),
            "lost":       (-0.4, +0.1),
            "empty":      (-0.6, -0.1),
            "stress":     (-0.3, +0.5),
            "stressed":   (-0.3, +0.5),
            "overwhelmed":(-0.5, +0.6),
            "okay":       (+0.1, 0.0),
            "fine":       (0.0, 0.0),
            "good":       (+0.3, +0.1),
            "great":      (+0.5, +0.2),
            "hope":       (+0.4, +0.1),
            "hopeful":    (+0.4, +0.1),
            "love":       (+0.8, +0.4),
            "happy":      (+0.7, +0.3),
            "calm":       (+0.3, -0.2),
            "relaxed":    (+0.4, -0.3),
            "excited":    (+0.6, +0.7),
            "worried":    (-0.4, +0.4),
            "anxious":    (-0.5, +0.5),
            "afraid":     (-0.6, +0.6),
            "scared":     (-0.6, +0.6),
            "depressed":  (-0.8, -0.4),
            "frustrated": (-0.5, +0.4),
        }

        # Load VOSK model safely
        self._load_model()
        
        # Check audio devices
        self._check_audio_devices()

    def _find_vosk_model(self):
        """Auto-detect VOSK model folder."""
        from pathlib import Path
        current_dir = Path.cwd()
        
        # Look for folders starting with "vosk-model" or in "models" subfolder
        potential_paths = [
            # Direct in current directory
            *list(current_dir.glob("vosk-model*")),
            # In models subdirectory
            *list(current_dir.glob("models/vosk-model*")),
            # Common alternative names
            current_dir / "vosk-model-small-en-us-0.15",
            current_dir / "models" / "vosk-model-small-en-us-0.15",
        ]
        
        for path in potential_paths:
            if path.exists() and path.is_dir():
                # Verify it has required structure
                if (path / "am" / "final.mdl").exists():
                    print(f"✓ Auto-detected model: {path}")
                    return str(path)
        
        return None

    def _load_model(self):
        """Load VOSK model with error handling."""
        try:
            print(f"Loading VOSK model from: {self.model_path}")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            
            # Enable partial results for more responsive feedback
            self.recognizer.SetWords(True)
            
            print("✓ VOSK model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load VOSK model: {e}")
            print("  Make sure the model folder exists and contains:")
            print("    - am/final.mdl")
            print("    - graph/HCLG.fst")
            print("    - ivector/final.ie")
            self.model = None
            self.recognizer = None

    def _check_audio_devices(self):
        """List available audio input devices."""
        print("\n🎤 Available Audio Input Devices:")
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default = " [DEFAULT]" if idx == sd.default.device[0] else ""
                    print(f"  [{idx}] {device['name']}{default}")
                    print(f"      Channels: {device['max_input_channels']}, "
                          f"Sample Rate: {device['default_samplerate']} Hz")
        except Exception as e:
            print(f"  Error querying devices: {e}")
        print()

    # ---------------------------------------------------------------
    # Public Controls
    # ---------------------------------------------------------------
    def start(self):
        """Start the voice recognition system."""
        if not self.model or not self.recognizer:
            print("Voice module unavailable – model failed to load")
            return

        self.running = True
        print("✓ Starting VOSK speech listener...")
        print("  Speak clearly into your microphone...")
        print("  Debug: Audio levels will be shown below\n")

        audio_thread = threading.Thread(target=self._audio_capture_thread, daemon=True)
        process_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        debug_thread = threading.Thread(target=self._debug_monitor_thread, daemon=True)

        audio_thread.start()
        process_thread.start()
        debug_thread.start()

    def stop(self):
        """Stop the voice recognition system."""
        self.running = False
        print("✓ VOSK voice module stopped")
        print(f"  Stats: {self.audio_chunks_received} audio chunks, "
              f"{self.words_recognized} words recognized")

    # ---------------------------------------------------------------
    # Audio Capture Thread
    # ---------------------------------------------------------------
    def _audio_capture_thread(self):
        """Captures microphone audio and pushes it to queue."""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"⚠ Audio status: {status}")
            
            if self.running:
                # Calculate audio level for debugging
                audio_data = np.frombuffer(indata, dtype=np.int16)
                self.last_audio_level = float(np.abs(audio_data).mean())
                
                self.audio_chunks_received += 1
                self.audio_q.put(bytes(indata))

        try:
            # Smaller blocksize for more responsive capture
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=4000,  # Reduced from 8000
                dtype="int16",
                channels=1,
                callback=callback
            ):
                print("✓ Microphone stream opened")
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"✗ Microphone error: {e}")
            print("  Common fixes:")
            print("    - Check microphone permissions")
            print("    - Ensure microphone is not in use by another app")
            print("    - Try selecting a different audio device")

    # ---------------------------------------------------------------
    # Processing Thread
    # ---------------------------------------------------------------
    def _process_audio_thread(self):
        """Process audio stream → text → emotion."""
        last_keyword = None
        partial_timeout = time.time()

        while self.running:
            try:
                data = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            # Check for partial results (interim transcription)
            if self.recognizer.AcceptWaveform(data):
                # Final result
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    self.words_recognized += len(text.split())
                    print(f" [RECOGNIZED] {text}")
                    self._process_text(text, last_keyword)
                    partial_timeout = time.time()
            else:
                # Partial result (real-time feedback)
                if time.time() - partial_timeout > 2.0:  # Show partial every 2 sec
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text:
                        print(f" [Hearing...] {partial_text}")
                        partial_timeout = time.time()

    def _process_text(self, text, last_keyword):
        """Process recognized text for emotions."""
        # Update engine with latest text
        if self.engine:
            self.engine.voice_last_text = text

        # Detect emotional keywords
        detected = self._extract_keywords(text)

        if detected:
            keyword = detected[0]
            valence, arousal = self.keyword_map[keyword]

            # Avoid repeating same keyword
            if keyword != last_keyword:
                print(f" [EMOTION DETECTED] '{keyword}' → "
                      f"Valence={valence:+.2f}, Arousal={arousal:+.2f}")
                last_keyword = keyword

            # Send emotion to fusion engine
            if self.engine:
                self.engine.update_voice(valence, arousal, keyword)

    # ---------------------------------------------------------------
    # Debug Monitor Thread
    # ---------------------------------------------------------------
    def _debug_monitor_thread(self):
        """Monitor and display audio capture stats."""
        while self.running:
            time.sleep(5.0)  # Update every 5 seconds
            
            # Audio level indicator
            level_bars = int(self.last_audio_level / 100)  # Scale to ~50 bars max
            level_visual = "█" * min(level_bars, 50)
            
            print(f"Debug: Audio Level: {level_visual} ({self.last_audio_level:.0f})")
            print(f"   Chunks received: {self.audio_chunks_received}, "
                  f"Words: {self.words_recognized}, Queue: {self.audio_q.qsize()}")
            
            if self.last_audio_level < 50:
                print("   ⚠ Audio level very low - speak louder or check mic!")

    # ---------------------------------------------------------------
    # Keyword Extraction
    # ---------------------------------------------------------------
    def _extract_keywords(self, text: str):
        """Extract emotional words from recognized text."""
        words = text.lower().split()
        found = []
        for w in words:
            # Exact match
            if w in self.keyword_map:
                found.append(w)
            # Partial match (e.g., "feeling" in "I'm feeling sad")
            else:
                for keyword in self.keyword_map:
                    if keyword in w or w in keyword:
                        found.append(keyword)
                        break
        return found


# ============================================================================
# STANDALONE TEST MODE
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  VOSK Voice Module - Standalone Test Mode")
    print("="*70)
    print("\nThis will test the microphone and speech recognition.")
    print("Speak emotional words like: happy, sad, angry, calm, etc.\n")
    
    # Mock engine for testing
    class MockEngine:
        def __init__(self):
            self.voice_last_text = ""
        
        def update_voice(self, valence, arousal, dominant):
            print(f"Engine updated: V={valence:+.2f}, A={arousal:+.2f}, "
                  f"emotion='{dominant}'")
    
    mock_engine = MockEngine()
    voice = ReflektVoiceVOSK(engine=mock_engine)
    
    try:
        voice.start()
        print("\n Voice module running. Press Ctrl+C to stop.\n")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n Stopping...")
        voice.stop()
        print("✓ Test complete\n")