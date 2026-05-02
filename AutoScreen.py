import ollama
import pyautogui
pyautogui.FAILSAFE = False
import re

def Primary():
    PantherVisionMR = []
    PantherLLavaMR = []
    try:
        while True:
            screenshot = pyautogui.screenshot()
            screenshot.save("WhatHappin.png")
            PantherLLavaRP = ollama.chat(
            model="granite3.2-vision",
            messages=[
                     {"role": "user", "content": "Make a detailed Description", "images": ["./WhatHappin.png"]}
                ],
            )
            PantherLLavaTXT = PantherLLavaRP['message']['content']
            print(PantherLLavaTXT)
            PantherLLavaMR.append({'role': 'assistant', 'content': PantherLLavaTXT})
            PantherVisionRP = ollama.chat(
            model="PantherVision",
            messages=[
                     {"role": "user", "content": PantherLLavaTXT,}
                ],
            )
            PantherVisionTXT = PantherVisionRP['message']['content']
            print(PantherVisionTXT)
            match = re.search(r'Coordinates:\s*(\d+)\s*,\s*(\d+)', PantherVisionTXT)
            if match:
                x, y = map(int, match.groups())
                pyautogui.moveTo(x, y)
                pyautogui.click()  
    except KeyboardInterrupt:
        print("\nReLock Chat DeActivated")

Primary()
