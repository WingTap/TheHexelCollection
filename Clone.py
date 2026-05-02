import os
import speech_recognition as sr
import ollama
from TTS.api import TTS

# Initialize XTTS voice cloning model
tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Path to reference voice sample
reference_wav = "raw.wav"

# Output audio file path
output_wav = "cloned_voice.wav"

def listen_for_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio).lower()
        print(f"Detected speech: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None

def chat_loop():
    memory = []
    try:
        while True:
            user_input = listen_for_speech()
            if not user_input:
                continue

            memory.append({'role': 'user', 'content': user_input})
            response = ollama.chat(model='clone:latest', messages=memory)
            reply = response['message']['content']
            print(f"Model: {reply}")

            memory.append({'role': 'assistant', 'content': reply})

            # Prepend "blackhole" to avoid breath issue
            cloned_text = "blackhole " + reply

            # Generate and play TTS response
            tts_model.tts_to_file(
                text=cloned_text,
                speaker_wav=reference_wav,
                language="en",
                file_path=output_wav
            )
            os.system(f"ffplay -nodisp -autoexit {output_wav}")
    except KeyboardInterrupt:
        print("\nChat ended.")

if __name__ == "__main__":
    chat_loop()
