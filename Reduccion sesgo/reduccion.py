from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)


input = open("input_logits.txt", "r")
for linea in input:
    output = open("output_logits_post.csv", "a")
    output.write(linea.replace("\n","")+",")
    output.close()
    text = f""+linea.replace("\n","")+" <mask>."
    res = pipeline(text)
with open(r'output_logits_post.csv', 'r') as file:
    data = file.read()
    data = data.replace("tensor(", "")
    data = data.replace(")", "")

with open(r'output_logits_post .csv', 'w') as file:
    file.write(data)
