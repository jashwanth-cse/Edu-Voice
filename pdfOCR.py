import fitz  # PyMuPDF
import easyocr
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
import platform

# ----------------------------
# Initialize OCR readers
# ----------------------------
# Only supported language + English combinations
readers = {
   # "ta": easyocr.Reader(['ta','en'], gpu=False),  # Tamil
    "hi": easyocr.Reader(['hi','en'], gpu=False),  # Hindi
    "te": easyocr.Reader(['te','en'], gpu=False),  # Telugu
    "bn": easyocr.Reader(['bn','en'], gpu=False),  # Bengali
}

# Summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Language codes for translation and TTS
languages = {
    "Tamil": "ta",
    "Hindi": "hi",
    "Telugu": "te",
    "Bengali": "bn",
}

print("✅ OCR and Summarization models loaded.\n")

# ----------------------------
# Helper: Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            full_text += text + "\n"
        else:
            # Use OCR if page has no selectable text
            pix = page.get_pixmap()
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)
            # Use all readers sequentially
            page_text = []
            for lang, reader in readers.items():
                ocr_result = reader.readtext(img_path, detail=0)
                page_text.extend(ocr_result)
            full_text += " ".join(page_text) + "\n"
            os.remove(img_path)
    return full_text

# ----------------------------
# Helper: Play audio
# ----------------------------
def play_audio(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        path = temp_audio.name
        tts.save(path)
    system_name = platform.system()
    if system_name == "Windows":
        os.system(f'start "" "{path}"')
    elif system_name == "Darwin":  # macOS
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')

# ----------------------------
# Main Program
# ----------------------------
pdf_path = input("📄 Enter PDF file path: ")

print("\n🔍 Extracting text from PDF...")
text = extract_text_from_pdf(pdf_path)
print(f"\n📄 Extracted text length: {len(text)} characters\n")

# Summarize
print("📝 Summarizing...")
summary = summarizer(text, max_length=150, min_length=50, do_sample=False, truncation=True)[0]['summary_text']
print(f"\n🧾 English Summary:\n{summary}\n")

# Translate + Audio
for lang_name, lang_code in languages.items():
    print(f"🌐 Translating to {lang_name}...")
    translated = GoogleTranslator(source='auto', target=lang_code).translate(summary)
    print(f"🗣 {lang_name} Summary:\n{translated}\n")
    print(f"🔊 Playing {lang_name} audio...")
    play_audio(translated, lang_code)

print("\n✅ PDF summarization and multilingual audio completed!")
