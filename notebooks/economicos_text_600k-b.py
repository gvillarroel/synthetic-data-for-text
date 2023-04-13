import argparse
from argparse import ArgumentParser
from os.path import join, isfile
from os import listdir
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from rouge_score import rouge_scorer
import shutil
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler #Dataset,
from transformers import get_linear_schedule_with_warmup, AdamW
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd
import numpy as np
#.replace(to_replace="-1", value=np.nan)
df = pd.read_parquet('../datasets/economicos/synth-b/split/train.parquet')
#.replace(to_replace="None", value=np.nan).replace(to_replace=-1, value=np.nan)
print(df.shape)
df.sample(3)
CHAR_SEP = "<SEP>"


def convert(row):
    return {
        "text": [
            f"""<fecha, {(pd.Timestamp('2017-12-01') +  pd.DateOffset(int(row.publication_date or 0))).strftime('%Y-%m-%d')}>
<precio, {row.price}>
<tipo, {row.property_type}>
<transacción, {row.transaction_type}>
<región, {row.state}>
<comuna, {row.county}>
<dormitorios, {row.rooms or -1}>
<baños, {row.rooms or -1}>
<construidos, {row.m_built or -1}>
<terreno, {row.m_size or -1}>
<precio_real, {row._price}>
<titulo, {row.title}>
<dirección, {row.address}>""".replace("\n", " "),
"descripción de esta publicación"],

"target": row.description
        }

print(
    df.sample(1).apply(convert, axis=1).iloc[-1]
)
TMP_TEXTPATH = "./df_text-b.parquet"
if not os.path.exists(TMP_TEXTPATH):
    df_text = pd.DataFrame(df.apply(convert, axis=1).to_list())
    print(df_text.sample(3))
    df_text.to_parquet(TMP_TEXTPATH)
else:
    df_text = pd.read_parquet(TMP_TEXTPATH)

import random
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)
pl.seed_everything(42)

class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


CHAR_SEP = " "
MAX_SRC_LEN = 200
MAX_TGT_LEN = 720
class T5Finetuner(pl.LightningModule):

    def __init__(self, args, df, batch_size=1):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model)
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model)
        self.data = df
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.batch_size = batch_size

    def encode_text(self, text, target):
        source = self.tokenizer.batch_encode_plus(["<SEP>".join(text)], 
                                                max_length= MAX_SRC_LEN, 
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([target], 
                                                max_length=MAX_TGT_LEN,
                                                truncation=True,
                                                padding='max_length',
                                                return_tensors='pt')
        y = target['input_ids']
        target_id = y[:, :-1].contiguous()
        target_label = y[:, 1:].clone().detach()
        target_label[y[:, 1:] == self.tokenizer.pad_token_id] = -100 #in case the labels are not provided, empty string
        return source['input_ids'], source['attention_mask'], target_id, target_label

        
    
    def prepare_data(self):
        source_ids, source_masks, target_ids, target_labels = [], [], [], [] 
        for _, row in self.data.iterrows():
            source_id, source_mask, target_id, target_label = self.encode_text(row.text, row.target)
            source_ids.append(source_id)
            source_masks.append(source_mask)
            target_ids.append(target_id)
            target_labels.append(target_label)

        # Convert the lists into tensors
        source_ids = torch.cat(source_ids, dim=0)
        source_masks = torch.cat(source_masks, dim=0)
        target_ids = torch.cat(target_ids, dim=0)
        target_labels = torch.cat(target_labels, dim=0)
        # splitting the data to train, validation, and test
        data = TensorDataset(source_ids, source_masks, target_ids, target_labels)
        train_size, val_size = int(0.8 * len(data)), int(0.1 * len(data))
        test_size = len(data) - (train_size + val_size)
        self.train_dat, self.val_dat, self.test_dat = \
            random_split(data, [train_size, val_size, test_size])
    
    def forward(self, batch, batch_idx):
        source_ids, source_mask, target_ids, target_labels = batch[:4]
        return self.model(input_ids = source_ids, attention_mask = source_mask, 
                          decoder_input_ids=target_ids, labels=target_labels)
        
    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        return {'loss': loss, 'val_loss': loss}

    #def on_validation_epoch_end(self, outputs):
    #    loss = sum([o['loss'] for o in outputs]) / len(outputs)
    #    out = {'val_loss': loss}
    #    return {**out, 'log': out}

    def test_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        return {'loss': loss}

    #def on_test_epoch_end(self, outputs):
    #    loss = sum([o['loss'] for o in outputs]) / len(outputs)
    #    out = {'test_loss': loss}
    #    return {**out, 'log': out}
    
    def train_dataloader(self):
        return DataLoader(self.train_dat, batch_size=self.args.bs,
                          num_workers=4, sampler=RandomSampler(self.train_dat))

    def val_dataloader(self):
        return DataLoader(self.val_dat, batch_size=self.args.bs, num_workers=4,
                          sampler=SequentialSampler(self.val_dat))

    def test_dataloader(self):
        return DataLoader(self.test_dat, batch_size=self.args.bs, num_workers=4,
                          sampler=SequentialSampler(self.test_dat))    

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=1e-4)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=self.args.max_epochs * len(self.train_dat))
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def generate_summary(self, ctext, summ_len=150, text='', beam_search=2, repetition_penalty=2.5):
        source_id, source_mask, target_id, target_label = self.encode_text(ctext, text)
        self.model.eval()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids = source_id,
                attention_mask = source_mask, 
                max_length=summ_len, 
                truncation=True,
                num_beams=beam_search,
                repetition_penalty=repetition_penalty, 
                length_penalty=1.0, 
                early_stopping=True
                )
            prediction = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        if len(text) > 0:
            target = [self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in target_id]
            scores = self.scorer.score(target[0], prediction[0])
            return prediction, scores
        else:
            return prediction
        

    def save_core_model(self):
        store_path = join(self.args.output, self.args.name, 'core')
        self.model.save_pretrained(store_path)
        self.tokenizer.save_pretrained(store_path)

def default_args():
    p = ArgumentParser()
    args,_ = p.parse_known_args()
    args.max_epochs = 5
    args.model = "google/mt5-base"
    args.output = f"./B1-{args.model.replace('/','_')}"
    args.name = "DESCRIPCION_PROPIEDADES"
    args.bs = 1 # batch size
    return args


args = default_args()
# sampling the hyperparameters
args.lr = 2e-5
# setting up the right callbacks
cp_callback = pl.callbacks.ModelCheckpoint(
    join(args.output, args.name, f"trial_0", "{epoch}"),
    monitor="loss", mode="min")
metrics_callback = MetricsCallback()
summarizer = T5Finetuner(args, df_text)         # loading the model
#summarizer = T5Finetuner.load_from_checkpoint("./B-google_mt5-base/logs/DESCRIPCION_PROPIEDADES/trial_0/checkpoints/epoch=3-step=1746784.ckpt")
trainer = pl.Trainer(      # loading the trainer
    accelerator="gpu",
    devices=1,
    max_epochs=5,
    default_root_dir=args.output, 
    gradient_clip_val=1.0,
    callbacks=[metrics_callback],
    num_sanity_val_steps=0,
    logger=TensorBoardLogger(join(args.output, 'logs'), name=args.name, version=f'trial_0')
    )

trainer.fit(summarizer)                       # fitting the model
trainer.test(summarizer)                      # testing the model