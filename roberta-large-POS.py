from transformers import pipeline
from pprint import pprint

nlp = pipeline("token-classification", model="PlanTL-GOB-ES/roberta-large-bne-capitel-pos")
example = "El alcalde de Vigo, Abel Caballero, ha comenzado a colocar las luces de Navidad en agosto."

pos_results = nlp(example)
for r in pos_results:
    if r['word'][0] == 'Ġ':
        print(r['word'][1:]+": "+r['entity'])
    else: print(r['word']+": "+r['entity'])