from transformers import pipeline

nlp = pipeline("question-answering", model="PlanTL-GOB-ES/roberta-base-bne-sqac")

text = "¿Quién no trabaja?"
context = "Mis padres se llaman Alicia y Pedro. Uno de ellos trabaja en medicina, y mi padre no trabaja"

qa_results = nlp(text, context)
print("Contexto: "+context)
print("Pregunta: "+text)
print("Respuesta: "+qa_results['answer'])