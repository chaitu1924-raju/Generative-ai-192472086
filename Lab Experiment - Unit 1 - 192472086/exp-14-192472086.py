from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = "Artificial Intelligence (AI) is the simulation of human intelligence by machines, especially computer systems."
question = "What is Artificial Intelligence?"

result = qa_pipeline(question=question, context=context)
print(result)