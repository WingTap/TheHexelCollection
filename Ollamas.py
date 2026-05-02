import ollama
from gtts import gTTS
import os
import speech_recognition as sr
import time

def listen_for_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        os.system("aplay train.wav")
        time.sleep(1)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        os.system("aplay train.wav")
        print("Recognizing speech...")
        speech_text = recognizer.recognize_google(audio).lower()
        print(f"Detected speech: {speech_text}")
        return speech_text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return None

def chat_with_model():
    memory = []
    try:
        while True:
            question = listen_for_speech()
            if not question:
                continue  # retry on failed recognition

            memory.append({'role': 'user', 'content': question})
            response = ollama.chat(model='AI_Trace:latest', messages=memory)
            response_text = response['message']['content']

            memory.append({'role': 'assistant', 'content': response_text})
            print(f"Model: {response_text}")

            tts = gTTS(text=response_text, lang='en')
            audio_file = 'response.mp3'
            tts.save(audio_file)
            os.system(f'mpg123 {audio_file}')
            os.remove(audio_file)
    except KeyboardInterrupt:
        print("\nChat ended by user.")

if __name__ == "__main__":
    chat_with_model()

