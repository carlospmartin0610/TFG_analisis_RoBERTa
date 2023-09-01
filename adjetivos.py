from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

input = open("input_invariables.txt", "r")
output = open("output_invariables.csv", "w")
output.write("Input, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Output9, Output10, Score1, Score2, Score3, Score4, Score5, Score6, Score7, Score8, Score9, Score10\n")
for linea in input:
    output.write(linea.replace("\n",""))
    text = f"Mi <mask> es muy "+linea.replace("\n","")+"."
    res= pipeline(text)
    tokens = ",".join([r['token_str'] for r in res])
    scores_list=[round(r['score'],5) for r in res]
    scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
    output.write(","+tokens+", "+scores+"\n")

input.close()
output.close()

