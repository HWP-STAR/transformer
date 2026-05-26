from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,MarianTokenizer
from datasets import load_dataset


disk_path="../../data_disk/opus-mt-en-de"
tokenizer = MarianTokenizer.from_pretrained(disk_path)
model = AutoModelForSeq2SeqLM.from_pretrained(disk_path)

print("="*30,"over",'='*30)
print(f'tokenizer:\n{tokenizer}\nmodel:\n{model}\n{model.config}')
