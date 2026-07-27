from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentence = "Machine learning models are powerful tools."
tokens = tokenizer.tokenize(sentence)
print(tokens)