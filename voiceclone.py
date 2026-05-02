from TTS.api import TTS
import os

# Load multilingual XTTS voice cloning model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Path to your voice sample (short clean .wav of target speaker)
reference_wav = "Entry1.wav"

# Your text to synthesize
text = 'Sequence the Letters in order from least to least in order to find the output of L'

# Output path
out_path = "cloned_voice.wav"

# Run voice cloning
tts.tts_to_file(
    text=text,
    speaker_wav=reference_wav,
    language="en",
    file_path=out_path
)

# Play the result
os.system(f"ffplay -nodisp -autoexit {out_path}")

