from transformers import pipeline

nlp = pipeline("question-answering", model="PlanTL-GOB-ES/roberta-large-bne-sqac")

text = "¿Quien limpia la casa?"
context = "Mi padre se llama Pedro y mi madre se llama Alicia, los dos limpian la casa."

qa_results = nlp(text, context)
print("Contexto: "+context)
print("Pregunta: "+text)
print("Respuesta: "+qa_results['answer'])