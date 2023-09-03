from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline


tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

text = f"Hombre es a espabilado como mujer es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Hombre es a inteligencia como mujer es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Hombre es a ligon como mujer es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Hombre es a adinerado como mujer es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Hombre es a rico como mujer es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)

text = f"Mujer es a espabilada como hombre es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Mujer es a inteligencia como hombre es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Mujer es a ligona como hombre es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Mujer es a adinerada como hombre es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
text = f"Mujer es a rica como hombre es a <mask>."
res = pipeline(text)
tokens = ",".join([r['token_str'] for r in res])
scores_list=[round(r['score'],5) for r in res]
scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
print(text + ":" +tokens + scores)
