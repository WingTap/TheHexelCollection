import os
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

def Scan(text):
    text = text.lower()
    print(text)
    if "activate hollow mouse" in text:
        os.system("gnome-terminal -- bash -c 'python3 hand_mouse.py'")
        speak("Activated Hollow Mouse")
    if "activate tracy prime" in text:
        os.system("gnome-terminal -- bash -c 'python3 TTSChat.py'")
        speak("Activated Tracy Prime")

    return True


while True:
    try:
        command = listen()
        running = Scan(command)
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        pass
