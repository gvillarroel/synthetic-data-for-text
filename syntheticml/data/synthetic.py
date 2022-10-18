
from operator import imod
import pandas as pd
from sdv.tabular import GaussianCopula, TVAE, CTGAN, CopulaGAN
from sdv.metadata import table
import os
import json
import operator as op
import plotly.graph_objects as go

from .metrics import Metrics
from .charts import Charts


MODELS = {
    "copulagan" : CopulaGAN,
    "tvae" : TVAE,
    "gaussiancopula": GaussianCopula,
    "ctgan": CTGAN
}


class Synthetic:
    def __init__(self, df : pd.DataFrame, category_columns : tuple[str], id : str, synthetic_folder: str, text_columns : tuple[str] = (), models : tuple[str] = [], n_sample: int = 0) -> None:
        self.df = df
        self.category_columns = category_columns
        self.id = id
        self.n_sample = n_sample
        self.text_columns = text_columns
        self.synthetic_folder = synthetic_folder
        self.metadata_path = f"{self.synthetic_folder}/metadata.json"
        self.make_folders()
        self.metadata = self._gen_metadata()
        self.models = self._gen_models(models)
        self.metric = Metrics(self.df, self.metadata)
        self.synths = self._gen_synthetic()
        self.charts = Charts(self.metadata)
        self.cuda = False
        

    def make_folders(self)  -> dict:
        self.syntheticdata_folder = f"{self.synthetic_folder}/data"
        self.checkpoint_folder = f"{self.synthetic_folder}/checkpoint"
        self.report_folder = f"{self.synthetic_folder}/report"
        
        folders = [self.syntheticdata_folder, self.checkpoint_folder, self.report_folder]
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f, exist_ok=True)

    def _gen_metadata(self) -> dict:
        if not os.path.exists(self.metadata_path):
            meta = table.Table(primary_key=self.id, field_names=set(self.df.columns.to_list()) - set(self.text_columns), field_transformers={k:'LabelEncoder' for k in self.category_columns})
            meta.fit(self.df.astype({ k:"category" for k in self.category_columns }))
            meta.to_json(self.metadata_path)
        return json.load(open(self.metadata_path, "r"))
    
    def _gen_models(self, models: list[str]) -> dict:
        getter = op.itemgetter(*models)
        return { k: model(table_metadata=self.metadata) for k, model in zip(models,getter(MODELS))}

    def _gen_synthetic(self) -> dict:
        data_gen = {}
        for k, model in self.models.items():
            checkpoint = f"{self.checkpoint_folder}/{k}.ckp"
            if os.path.exists(checkpoint):
                model = model.load(checkpoint)
            else:
                model.fit(self.df)
                model.save(checkpoint)
            syndata_path = f"{self.syntheticdata_folder}/{k}_{self.n_sample}.parquet"
            if not os.path.exists(syndata_path):
                model.sample(self.n_sample).to_parquet(syndata_path)
            data_gen[k] = pd.read_parquet(syndata_path)
        return data_gen
        
    def process(self, cuda: bool = False) -> None:
        self.cuda = cuda
        self.fake_data = self._gen_synthetic()
        self.scores, self.reports = self.metric.get_scores(self.fake_data, self.report_folder)

    def current_metrics(self) -> pd.DataFrame:
        return self.metric.get_metrics(self.df)
    
    def get_metric(self, method:  str) -> pd.DataFrame:
        return self.metric.get_metrics(self.fake_data[method])

    def get_metrics_fake(self) -> dict:
        return {
            k: self.metric.get_metrics(fake_data)
            for k, fake_data in self.fake_data.items()
        }
    
    def get_charts(self, model_key : str, exclude_columns: set[str] = {}) -> list[go.Figure]:
        return self.charts.charts(self.df, self.synths[model_key], exclude_columns)