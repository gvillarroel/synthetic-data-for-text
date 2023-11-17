import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer,  Seq2SeqTrainingArguments
from datasets import Dataset
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
import sys
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model


MODEL_NAME = "google/mt5-large"
VERSION = sys.argv[1]
MAX_SRC_LEN = 150
MAX_TGT_LEN = 720

df_train = pd.read_parquet(f'../datasets/economicos/synth-{VERSION}/split/train.parquet').dropna(subset=["description"])

## Preparando los datos
def tabular_to_text(row, tarea="descripción de esta publicación"):
    data = f"""
<fecha, {row.publication_date}>
<precio, {row.price}>
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


df_train["input_text"] = df_train.apply(tabular_to_text, axis=1)
df_train = df_train.rename(columns={"description": "target_text"})


ds = Dataset.from_pandas(df_train)

train_test_split = ds.train_test_split(test_size=0.1)  # Ajusta el test_size según tus 

print(train_test_split)

# Tokenización de los datos
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)


def map_function(examples):
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=MAX_TGT_LEN, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "text": examples["input_text"],
        "text_label": examples["target_text"]
    }

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["input_text"], padding="max_length", truncation=True, max_length=MAX_SRC_LEN, return_tensors="pt")
    labels = tokenizer(examples["target_text"], padding="max_length", truncation=True, max_length=MAX_TGT_LEN, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

processed_datasets = train_test_split.map(
    tokenize_function, 
    batched=True,
    remove_columns=train_test_split["train"].column_names,
    cache_file_names={k:f"./cache_{VERSION}_{k}" for k in ["train", "test"]})

tokenized_train_dataset = processed_datasets["train"]
tokenized_eval_dataset = processed_datasets["test"]

# Carga del modelo + peft


#model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, peft_config)
#model.load_adapter("./results/checkpoint-245642")
model.print_trainable_parameters()

#from trl import SFTTrainer
#from trl.trainer import ConstantLengthDataset
#
#trainer = SFTTrainer(
#    model=model,
#    tokenizer=tokenizer,
#    train_dataset=tokenized_train_dataset,
#    eval_dataset=tokenized_eval_dataset,
#    peft_config=peft_config,
#    dataset_batch_size=4,
#    dataset_text_field="text"
#)


# Configuración de los argumentos de entrenamiento
training_args =  Seq2SeqTrainingArguments(
    output_dir=f"./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    resume_from_checkpoint="./results/checkpoint-245642"
)

# Crear el objeto Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

# Entrenamiento3
trainer.train(resume_from_checkpoint="./results/checkpoint-245642")

#trainer.evaluate()

# Guardar el modelo ajustado
model.save_pretrained(f"./{MODEL_NAME}_{VERSION}")


