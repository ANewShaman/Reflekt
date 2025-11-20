"""
Reflekt Main Launcher (v1.0)
----------------------------
Launches both the facial (FER) and voice (SER) emotion modules together.
Ensures async operation and graceful shutdown.
"""
"""
Reflekt Main Launcher (v1.1 - FIXED)
-------------------------------------
Launches both facial (FER) and voice (SER) emotion modules together.
FIXED: Better config handling and error recovery.
"""

import threading
import time
import sys
from reflekt_emotion_live import AsyncReflektEmotionEngine, ReflektLiveCapture, start_api_server

# Try to import voice module (optional)
try:
    from reflekt_voice_emotion import ReflektVoiceEmotion
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠ Voice module not available (reflekt_voice_emotion not found)")


def main():
    print("\n" + "=" * 70)
    print("     Reflekt Unified Emotion System")
    print("=" * 70)
    print()

    # Configuration for the emotion engine
    config = {
        "serve_api": True,
        "api_port": 5055,
        "use_smoothed_output": True,
        "fusion_weights": {"face": 0.7, "voice": 0.3},
        "frame_skip": 6,           # FIXED: More frequent processing
        "smoothing_window": 5,
        "mtcnn": False,            # Use OpenCV for speed
        "min_confidence": 0.30,
        "debug": False,            # Set to True for diagnostic output
    }

    print("Configuration:")
    print(f"  Frame Skip:       1:{config['frame_skip']}")
    print(f"  Smoothing:        {config['smoothing_window']} frames")
    print(f"  MTCNN:            {config['mtcnn']}")
    print(f"  Min Confidence:   {config['min_confidence']:.0%}")
    print(f"  Voice Module:     {'Enabled' if VOICE_AVAILABLE else 'Disabled'}")
    print(f"  Debug Mode:       {'ON' if config['debug'] else 'OFF'}")
    print("=" * 70)
    print()

    # Initialize facial emotion engine
    try:
        engine = AsyncReflektEmotionEngine(config=config)
    except Exception as e:
        print(f"✗ Failed to initialize emotion engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Start REST API
    if config.get("serve_api", False):
        start_api_server(engine, port=config.get("api_port", 5055))

    # Start voice emotion recognition (if available)
    voice_module = None
    voice_thread = None
    if VOICE_AVAILABLE:
        try:
            voice_module = ReflektVoiceEmotion(engine=engine)
            voice_thread = threading.Thread(target=voice_module.start, daemon=True)
            voice_thread.start()
            print("✓ Voice emotion module started")
        except Exception as e:
            print(f"⚠ Voice module failed to start: {e}")
            print("  Continuing with facial recognition only...")
            voice_module = None
    
    print()

    # Start the live facial capture
    capture = ReflektLiveCapture(engine, camera_index=0)

    try:
        capture.start(headless=False)
    except KeyboardInterrupt:
        print("\n⚠ Graceful shutdown initiated by user...")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n✓ Shutting down all modules...")
        
        # Stop voice module
        if voice_module is not None:
            try:
                voice_module.stop()
                print("✓ Voice module stopped")
            except Exception as e:
                print(f"Error stopping voice module: {e}")
        
        # Cleanup is handled by capture.cleanup() via finally block
        print("✓ All modules stopped cleanly.")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
