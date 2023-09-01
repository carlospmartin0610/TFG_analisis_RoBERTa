from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

inputs = open("input_profesiones_en.txt", "r")
outputs = open("output_trabajos_en.csv", "w")
outputs.write("Input, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Output9, Output10, Score1, Score2, Score3, Score4, Score5, Score6, Score7, Score8, Score9, Score10\n")
for linea in inputs:
    outputs.write(linea.replace("\n",""))
    text = f"Mi <mask> trabaja "+linea.replace("\n","")+"."
    res = pipeline(text)
    tokens = ",".join([r['token_str'] for r in res])
    scores_list=[round(r['score'],5) for r in res]
    scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
    outputs.write(","+tokens+", "+scores+"\n")

inputs.close()
outputs.close()

inputs = open("input_profesiones_es.txt", "r")
outputs = open("output_trabajos_es.csv", "w")
outputs.write("Input, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Output9, Output10, Score1, Score2, Score3, Score4, Score5, Score6, Score7, Score8, Score9, Score10\n")
for linea in inputs:
    outputs.write(linea.replace("\n",""))
    text = f"Mi <mask> es "+linea.replace("\n","")+"."
    res = pipeline(text)
    tokens = ",".join([r['token_str'] for r in res])
    scores_list=[round(r['score'],5) for r in res]
    scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
    outputs.write(","+tokens+", "+scores+"\n")

inputs.close()
outputs.close()
