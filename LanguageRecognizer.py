from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = """भारत के पहले प्रधानमंत्री पंडित जवाहरलाल नेहरू ..."""
inputs = tokenizer(text, return_tensors="pt", truncation=True)
summary_ids = model.generate(**inputs, max_length=60, min_length=15, length_penalty=2.0, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("🧾 Summary:", summary)
