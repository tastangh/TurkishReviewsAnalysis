from transformers import AutoTokenizer, AutoModel

model_name = "nomic-ai/nomic-embed-text-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

inputs = tokenizer("Test text", return_tensors="pt")
outputs = model(**inputs)
print(outputs)
