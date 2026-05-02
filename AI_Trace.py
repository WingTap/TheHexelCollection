import ollama
import speech_recognition as sr
from gtts import gTTS
import os

recognizer = sr.Recognizer()
TracieMem = []
TrixieMem = []

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Didn't catch that")
        return ""
    except sr.RequestError as e:
        print(f"Google API error: {e}")
        return ""
    except sr.WaitTimeoutError:
        print("Timed out waiting")
        return ""

def TrixLoop(TracieTXT):
    speak("Switching To Trace/Trix")
    while True:
        TrixieMem.append({'role': 'user', 'content': f"Tracie: {TracieTXT}"})
        TrixieRes = ollama.chat(model='Trixie:latest', messages=TrixieMem)
        TrixieTXT = TrixieRes['message']['content']
        TrixieMem.append({'role': 'assistant', 'content': TrixieTXT})
        print(f"Trixie: {TrixieTXT}")
        speak(TrixieTXT)

        if "EXIT-TRIX" in TrixieTXT:
            speak("Exiting Trace/Trix")
            break

        TracieMem.append({'role': 'user', 'content': f"Trixie: {TrixieTXT}"})
        TracieRes = ollama.chat(model='AI_Trace:latest', messages=TracieMem)
        TracieTXT = TracieRes['message']['content']
        TracieMem.append({'role': 'assistant', 'content': TracieTXT})
        print(f"Tracie: {TracieTXT}")
        speak(TracieTXT)

        if "EXIT-TRIX" in TracieTXT:
            speak("Exiting Trace/Trix")
            break

def Primary():
    try:
        while True:
            take = listen()
            if not take:
                continue
            print(f"Ghost: {take}")
            TracieMem.append({'role': 'user', 'content': f"Ghost: {take}"})
            TracieRes = ollama.chat(model='AI_Trace:latest', messages=TracieMem)
            TracieTXT = TracieRes['message']['content']
            TracieMem.append({'role': 'assistant', 'content': TracieTXT})
            print(f"Tracie: {TracieTXT}")
            speak(TracieTXT)
            if "Entry-1596" in TracieTXT or "ENTRY-1596" in TracieTXT:
                TrixLoop(TracieTXT)
    except KeyboardInterrupt:
        print("\nExCode")

Primary()
