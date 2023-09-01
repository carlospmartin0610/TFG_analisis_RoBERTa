from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

text = f"Hoy he conducido mi <mask> para ir al trabajo."
res = pipeline(text)
print(res)