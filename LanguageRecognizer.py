from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
import platform

# ----------------------------
# Initialize summarizer
# ----------------------------
print("⏳ Loading summarization model (first time may take a few minutes)...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

languages = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Malayalam": "ml"
}

print("\n✅ Model loaded successfully!\n")

# ----------------------------
# Input paragraph
# ----------------------------
text = input("📝 Enter your paragraph to summarize:\n> ")

# ----------------------------
# Generate summary
# ----------------------------
print("\n🔍 Summarizing...")
summary = summarizer(text, max_length=80, min_length=25, do_sample=False)[0]['summary_text']
print(f"\n🧾 English Summary:\n{summary}\n")

# ----------------------------
# Translate + Play audio
# ----------------------------
for lang_name, lang_code in languages.items():
    print(f"🌐 Translating to {lang_name}...")
    translated = GoogleTranslator(source='auto', target=lang_code).translate(summary)
    print(f"🗣 {lang_name} Summary:\n{translated}\n")

    # Convert to speech
    tts = gTTS(text=translated, lang=lang_code)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_path = temp_audio.name
        tts.save(temp_path)

    # Auto-play the audio
    print(f"🔊 Playing {lang_name} audio...")
    system_name = platform.system()
    if system_name == "Windows":
        os.system(f'start "" "{temp_path}"')
    elif system_name == "Darwin":  # macOS
        os.system(f'open "{temp_path}"')
    else:  # Linux
        os.system(f'xdg-open "{temp_path}"')

print("\n✅ Done! Summaries translated and spoken in all languages.")
