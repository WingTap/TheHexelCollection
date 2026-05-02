import os
import ollama
import speech_recognition as sr
from gtts import gTTS

def speak(text):
    gTTS(text).save("output.mp3")
    os.system("mpg123 output.mp3")

def listen():
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 25
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Clearing Ambience...")
        print("Scanning...")
        audio = recognizer.listen(source)
    return recognizer.recognize_google(audio)

def chat_with_model():
    memory = []
    while True:
        try:
            question = listen()
            if not question:
                continue
            print(f"User: {question}")
            if question.lower() == "exit":
                speak("User Ghost Has Exited The System")
                break
            memory.append({'role': 'user', 'content': question})
            response = ollama.chat(model='AI_Trace:latest', messages=memory)
            response_text = response['message']['content']
            memory.append({'role': 'assistant', 'content': response_text})
            print(f"Model: {response_text}")
            speak(response_text)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass
        except KeyboardInterrupt:
            print("\nChat ended by user.")
            break

chat_with_model()
