import os
import tempfile
import platform
import fitz  # PyMuPDF
import easyocr
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS

# ----------------------------
# Configuration
# ----------------------------
lang_pairs = [
    ['ta', 'en'],  # Tamil
    ['te', 'en'],  # Telugu
    ['bn', 'en'],  # Bengali
    ['hi', 'en'],  # Hindi
]

languages = {
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
}

# ----------------------------
# Helper: Initialize EasyOCR readers
# ----------------------------
def init_readers(lang_pairs):
    readers = {}
    for pair in lang_pairs:
        lang = pair[0]
        # Check if model exists locally to avoid re-download
        model_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR", lang)
        if os.path.exists(model_dir):
            print(f"✅ Using cached EasyOCR model for {lang}")
            readers[lang] = easyocr.Reader(pair, gpu=False, model_storage_directory=os.path.join(os.path.expanduser("~"), ".EasyOCR"))
        else:
            print(f"⚡ Downloading EasyOCR model for {lang} (first time, may take a few minutes)")
            readers[lang] = easyocr.Reader(pair, gpu=False)
    return readers

readers = init_readers(lang_pairs)

# ----------------------------
# Initialize summarizer
# ----------------------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("✅ Summarizer model loaded.\n")

# ----------------------------
# Helper: Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            full_text += text + "\n"
        else:
            # Use OCR if page has no text
            pix = page.get_pixmap()
            img_path = f"temp_page_{page_number}.png"
            pix.save(img_path)

            # Try OCR for each language
            ocr_page_text = []
            for lang, reader in readers.items():
                ocr_result = reader.readtext(img_path, detail=0)
                if ocr_result:
                    ocr_page_text.append(" ".join(ocr_result))
            full_text += " ".join(ocr_page_text) + "\n"
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
    elif system_name == "Darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')

# ----------------------------
# Main Program
# ----------------------------
pdf_path = input("English_book.pdf")

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
