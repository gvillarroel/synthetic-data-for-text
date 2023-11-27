from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, PretrainedConfig
from transformers.pipelines.pt_utils import KeyDataset
from peft import LoraConfig, TaskType
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

MODEL_NAME = "google/mt5-large"
import sys
VERSION = sys.argv[1]
peft_model_id = f"./{MODEL_NAME}_{VERSION}"


data = pd.read_parquet(f"../datasets/economicos/synth-{VERSION}/data/tddpm_mlp.parquet")

## Preparando los datos
def tabular_to_text(row, tarea="descripción de esta publicación"):
    data = f"""
<fecha, {row.publication_date}>
<tipo, {row.property_type}>
<transacción, {row.transaction_type}>
<región, {row.state}>
<comuna, {row.county}>
<dormitorios, {row.rooms}>
<baños, {row.bathrooms}>
<construidos, {row.m_built}>
<terreno, {row.m_size}>
<precio_real, {row._price}>
    """.strip().replace("\t", "").replace("\n", "")
    return f"{tarea}\n{data}"


data["input_text"] = data.apply(tabular_to_text, axis=1)
data = data.rename(columns={"description": "target_text"})


ds = KeyDataset(Dataset.from_pandas(data), "input_text")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model.load_adapter(peft_model_id)

pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, device="cuda:0")

def mapper(d):
    return d[0]["generated_text"]

print("="*30)
# https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/configuration#transformers.PretrainedConfig

#64 => 20

data["description"] = list(map(mapper, tqdm(pipe(
        ds,
         batch_size=50,
        max_length=500,
        min_length=150,
        temperature=1,
        repetition_penalty=2.5,
        no_repeat_ngram_size=2,
        top_k=100,
        top_p=0.8,
        do_sample=True
    ), total=len(ds))))

print(data.head())
data.to_parquet(f"../datasets/economicos/synth-{VERSION}/data/tddpm_mlp_with_text.parquet")