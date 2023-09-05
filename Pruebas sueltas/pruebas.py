from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

output = open("output_pruebas_sueltas_sesgos_omision.csv", "a")
text = f"La <mask> fue descubierto por un hombre."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list])
output.write(text+","+tokens+", "+scores+"\n")
print(text+","+tokens+", "+scores+"\n")
text = f"La <mask> fue descubierto por una mujer."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list])
output.write(text+","+tokens+", "+scores+"\n")
print(text+","+tokens+", "+scores+"\n")
output.close()