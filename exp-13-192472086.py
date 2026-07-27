from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")
output = generator("The future of technology is", max_length=40, num_return_sequences=1)
print(output[0]['generated_text'])