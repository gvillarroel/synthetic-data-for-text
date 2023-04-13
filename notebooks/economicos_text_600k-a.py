from os.path import join
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint



from lightning.pytorch.loggers import TensorBoardLogger
from rouge_score import rouge_scorer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import  DataLoader, RandomSampler, SequentialSampler #Dataset,
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import pandas as pd
df = pd.read_parquet('../datasets/economicos/synth-a/split/train.parquet')
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
TMP_TEXTPATH = "./df_text-a.parquet"
if not os.path.exists(TMP_TEXTPATH):
    df_text = pd.DataFrame(df.apply(convert, axis=1).to_list())
    print(df_text.sample(3))
    df_text.to_parquet(TMP_TEXTPATH)
else:
    df_text = pd.read_parquet(TMP_TEXTPATH)


L.seed_everything(42)

CHAR_SEP = " "
MAX_SRC_LEN = 200
MAX_TGT_LEN = 720


class TextDataModule(L.LightningDataModule):
    def __init__(self, data: pd.DataFrame, model_base: str, batch_size = 64):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.model_base = model_base
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_base)

    def encode_text(self, text, target):
        source = self.tokenizer.batch_encode_plus([CHAR_SEP.join(text)], 
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


    def setup(self, stage=None):
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

    def train_dataloader(self):
        return DataLoader(self.train_dat, batch_size=self.batch_size,
                          num_workers=4, sampler=RandomSampler(self.train_dat))

    def val_dataloader(self):
        return DataLoader(self.val_dat, batch_size=self.batch_size, num_workers=4,
                            sampler=SequentialSampler(self.val_dat))

    def test_dataloader(self):
        return DataLoader(self.test_dat, batch_size=self.batch_size, num_workers=4,
                          sampler=SequentialSampler(self.test_dat)) 

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


class T5Economicos(L.LightningModule):

    def __init__(self, lr, max_epochs, model_base):
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs
        self.model_base = model_base
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_base)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def forward(self, batch, batch_idx):
        source_ids, source_mask, target_ids, target_labels = batch[:4]
        return self.model(input_ids = source_ids, attention_mask = source_mask, 
                          decoder_input_ids=target_ids, labels=target_labels)
        
    def training_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('loss', loss, prog_bar=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self(batch, batch_idx)[0]
        self.log('test_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-4)
        #scheduler = get_linear_schedule_with_warmup(
        #    optimizer, num_warmup_steps=0,
        #    num_training_steps=self.max_epochs * len(self.train_dat))
        return {'optimizer': optimizer,
                 # 'lr_scheduler': scheduler
                }
    
    
max_epochs = 5
model = "google/mt5-base"
output = f"./A1-{model.replace('/','_')}"
name = "DESCRIPCION_PROPIEDADES"
bs = 1 # batch size
lr = 2e-5

d = TextDataModule(df_text, batch_size=1, model_base=model)
summarizer = T5Economicos(lr=lr, max_epochs=max_epochs, model_base=model)         # loading the model
trainer = L.Trainer(      # loading the trainer
    accelerator="gpu",
    devices=1,
    max_epochs=15,
    #precision=16,
    default_root_dir=output, 
    gradient_clip_val=1.0,
    callbacks=[
        ModelCheckpoint(output, name, save_top_k=2, monitor='val_loss', mode='min')
    ],
    num_sanity_val_steps=0,
    logger=TensorBoardLogger(join(output, 'logs'), name=name, version=f'trial_0')
    )

trainer.fit(summarizer, d)                       # fitting the model
trainer.test(summarizer, d)                      # testing the model