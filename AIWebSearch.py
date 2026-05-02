import ollama
import requests
import re
from bs4 import BeautifulSoup

Mem = []

take = input("Enter Prompt: ")

MODEL = "Stacy"
UA = {"User-Agent": "Mozilla/5.0 (Linux x86_64)"}
MAX_CHARS = 12000


def chat(msg):
    Mem.append({"role": "user", "content": msg})
    res = ollama.chat(model=MODEL, messages=Mem)
    out = res["message"]["content"].strip()
    Mem.append({"role": "assistant", "content": out})
    return out


def fetch_clean_html(url):
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        t.decompose()
    return soup.get_text(" ", strip=True)


while True:
    print("\n\n")

    # 1. Give prompt
    chat(f"User prompt: {take}")

    # 2. Ask model for a single best site
    url = chat(
        "Return the single best authoritative website for this topic.\n"
        "Only return a URL. No explanation."
    ).split()[0]

    print("[+] URL:", url)

    # 3. Fetch + clean site text
    text = fetch_clean_html(url)
    text = re.sub(r"\s+", " ", text)[:MAX_CHARS]

    # 4. Give site content
    chat("Website content:\n" + text)

    # 5. Summarize
    summary = chat(
        "Summarize concisely. Preserve key facts. No filler."
    )

    print("\n" + summary + "\n")

    # 6. Save
    with open("summary.txt", "a", encoding="utf-8") as f:
        f.write(summary + "\n\n")

