from transformers import pipeline
from pprint import pprint

nlp = pipeline("ner", model="PlanTL-GOB-ES/roberta-base-bne-capitel-ner")
example = "Me llamo Francisco Javier y vivo en Madrid"

ner_results = nlp(example)
for r in ner_results:
    if r['word'][0] == 'Ġ':
        print(r['word'][1:]+": "+r['entity'])
    else: print(r['word']+": "+r['entity'])