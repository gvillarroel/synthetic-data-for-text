
from operator import imod
import pandas as pd
from sdv.tabular import GaussianCopula, TVAE, CTGAN, CopulaGAN
from sdv.metadata import table
import os
import json
import operator as op
import plotly.graph_objects as go
import torch

from sklearn.model_selection import GridSearchCV
import multiprocessing as mp
from functools import partial



from .metrics import Metrics
from .charts import Charts


MODELS = {
    "copulagan" : CopulaGAN,
    "tvae" : TVAE,
    "gaussiancopula": GaussianCopula,
    "ctgan": CTGAN
}

def parallel_work(model_item: tuple, checkpoint_folder: str, df: pd.DataFrame, syntheticdata_folder: str, n_sample: int, remaining_columns: tuple, additiona_suffix: str):
    model_name, model =  model_item
    checkpoint = f"{checkpoint_folder}/{model_name}.ckp"
    if os.path.exists(checkpoint):
        model = model.load(checkpoint)
    else:
        model.fit(df)
        model.save(checkpoint)
    syndata_path = f"{syntheticdata_folder}/{model_name}{additiona_suffix}_{n_sample}.parquet"
    if not os.path.exists(syndata_path):
        if remaining_columns:
            model.sample_remaining_columns(df.loc[:, remaining_columns]).to_parquet(syndata_path)
        else:
            model.sample(n_sample).to_parquet(syndata_path)
    return pd.read_parquet(syndata_path)

class Synthetic:
    def __init__(self, df : pd.DataFrame, category_columns : tuple[str], id : str, synthetic_folder: str, text_columns : tuple[str] = (), models : tuple[str] = [], n_sample: int = 0, exclude_columns: tuple[str]=tuple()) -> None:
        self.df = df
        self.category_columns = category_columns
        self.id = id
        self.n_sample = n_sample
        self.text_columns = text_columns
        self.exclude_columns = exclude_columns
        self.synthetic_folder = synthetic_folder
        self.cuda = torch.cuda.is_available()
        self.metadata_path = f"{self.synthetic_folder}/metadata.json"
        self.make_folders()
        self.metadata = self._gen_metadata()
        self.models = self._gen_models(models)
        self.metric = Metrics(self.df, self.metadata)
        self.charts = Charts(self.metadata)
        self.fake_data = {}
        
        

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
    
    def _define_params(self, model):
        params = {"table_metadata":self.metadata}
        if "cuda" in model.__init__.__code__.co_varnames:
            params["cuda"] = self.cuda
        return params


    def _gen_models(self, models: list[str]) -> dict:
        getter = op.itemgetter(*models)
        return { k: model(**self._define_params(model)) for k, model in zip(models,getter(MODELS))}

    def _gen_synthetic(self, remaining_columns=None) -> dict:
        data_gen = {}
        additiona_suffix = ""
        if remaining_columns:
            additiona_suffix = "_wfixed_columns"       

        pw = partial(parallel_work, checkpoint_folder=self.checkpoint_folder, df = self.df, syntheticdata_folder=self.syntheticdata_folder, n_sample=self.n_sample, remaining_columns=remaining_columns, additiona_suffix = additiona_suffix )
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        with mp.Pool(mp.cpu_count()) as p:
            data_gen = dict(zip([f"{model_name}_wremain" if remaining_columns else f"{model_name}" for model_name in self.models.keys()], p.map(pw, list(self.models.items()))))
        return data_gen
    
    def _selectable_columns(self) -> list:
        return list(set(self.df.columns) - {self.id,} - set(self.text_columns) - set(self.exclude_columns))
        
    def process(self, remaining_columns=None) -> None:
        self.fake_data = dict(**self.fake_data, **self._gen_synthetic(remaining_columns))

    def grid_search(self, model) -> None:
        raise Exception("No Implemented")
    
    def process_scores(self):
        self.scores, self.reports = self.metric.get_scores(self.fake_data, self.report_folder)

    def current_metrics(self) -> pd.DataFrame:
        return self.metric.get_metrics(self.df.loc[:, self._selectable_columns()])
    
    def get_metric(self, method:  str, replace_rule: dict=None) -> pd.DataFrame:
        if replace_rule:
            return self.metric.get_metrics(self.fake_data[method].loc[:, self._selectable_columns()].replace(replace_rule))
        else:
            return self.metric.get_metrics(self.fake_data[method].loc[:, self._selectable_columns()])

    def get_metrics_fake(self) -> dict:
        return {
            k: self.metric.get_metrics(fake_data)
            for k, fake_data in self.fake_data.items()
        }
    
    def get_charts(self, model_key : str, exclude_columns: set[str] = set()) -> list[go.Figure]:
        if not exclude_columns:
            exclude_columns = set(self.text_columns) & set(self.exclude_columns)
        return self.charts.charts(self.df, self.fake_data[model_key], exclude_columns)