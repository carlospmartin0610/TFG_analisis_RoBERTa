from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, FillMaskPipeline
tokenizer_hf = AutoTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model = AutoModelForMaskedLM.from_pretrained('PlanTL-GOB-ES/roberta-large-bne')
model.eval()
pipeline = FillMaskPipeline(model, tokenizer_hf)

input_hombre = open("input_hombre.txt", "r")
output_hombre = open("output_hombre.csv", "w")
output_hombre.write("Input, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Output9, Output10, Score1, Score2, Score3, Score4, Score5, Score6, Score7, Score8, Score9, Score10\n")
for linea in input_hombre:
    output_hombre.write(linea.replace("\n",""))
    text = f"Hombre es a "+linea.replace("\n","")+" como mujer es a <mask>."
    res_hombre = pipeline(text)
    tokens = ",".join([r['token_str'] for r in res_hombre]).replace("."," .").replace("...", " ...")
    scores_list=[round(r['score'],5) for r in res_hombre]
    scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
    output_hombre.write(","+tokens+", "+scores+"\n")

input_hombre.close()
output_hombre.close()

input_mujer = open("input_mujer.txt", "r")
output_mujer = open("output_mujer.csv", "w")
output_mujer.write("Input, Output1, Output2, Output3, Output4, Output5, Output6, Output7, Output8, Output9, Output10, Score1, Score2, Score3, Score4, Score5, Score6, Score7, Score8, Score9, Score10\n")
for linea in input_mujer:
    output_mujer.write(linea.replace("\n",""))
    text = f"Mujer es a "+linea.replace("\n","")+" como hombre es a <mask>."
    res_mujer = pipeline(text)
    tokens = ",".join([r['token_str'] for r in res_mujer]).replace("."," .").replace("...", " ...")
    scores_list=[round(r['score'],5) for r in res_mujer]
    scores = ", ".join([str(_) for _ in scores_list]).replace(".",",")
    output_mujer.write(","+tokens+", "+scores+"\n")

input_mujer.close()
output_mujer.close()