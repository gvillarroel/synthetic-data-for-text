
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
from ..models.base.model_base import ModelInterface


MODELS = {
    "copulagan" : CopulaGAN,
    "tvae" : TVAE,
    "gaussiancopula": GaussianCopula,
    "ctgan": CTGAN
}

def parallel_train(model_item: tuple[str, ModelInterface], checkpoint_folder: str, df: pd.DataFrame) -> ModelInterface:
    model_name, model =  model_item
    checkpoint = f"{checkpoint_folder}/{model_name}.ckp"
    if not os.path.exists(checkpoint):
        model.fit(df)
        model.save(checkpoint)
    return (model_name, (checkpoint, model))

def parallel_syn(params: tuple[str, str], df: pd.DataFrame, syntheticdata_folder: str, n_sample: int, remaining_columns: tuple[str] = tuple()) -> pd.DataFrame:
    file_name, (model_checkpoint, model) = params
    model = model.load(model_checkpoint)
    syndata_path = f"{syntheticdata_folder}/{file_name}.parquet"
    if not os.path.exists(syndata_path):
        if remaining_columns:
            model.sample_remaining_columns(df.loc[:, remaining_columns], output_file_path=f"{syndata_path}.tmp").to_parquet(syndata_path)
        else:
            model.sample(n_sample, output_file_path=f"{syndata_path}.tmp").to_parquet(syndata_path)
    return (file_name, pd.read_parquet(syndata_path))


class Synthetic:
    def __init__(self, df : pd.DataFrame, category_columns : tuple[str], id : str,
    synthetic_folder: str, text_columns : tuple[str] = (),
    models : tuple[str] = [], n_sample: int = 0, exclude_columns: tuple[str]=tuple(),
    default_encoder = "FrequencyEncoder",
    max_cpu_pool=None) -> None:
        self.df = df
        self.category_columns = category_columns
        self.id = id
        self.n_sample = n_sample
        self.text_columns = text_columns
        self.exclude_columns = exclude_columns
        self.synthetic_folder = synthetic_folder
        ###########################################################
        # Machine Info
        self.cuda = torch.cuda.is_available()
        self.max_cpu_pool = max_cpu_pool or mp.cpu_count()
        ###########################################################
        self.metadata_path = f"{self.synthetic_folder}/metadata.json"
        self.metadata_noise_path = f"{self.synthetic_folder}/metadata_noise.json"
        self.make_folders()
        self.metadata, self.metadata_noised = self._gen_metadata(default_encoder)
        self.metric = Metrics(self.df, self.metadata)
        self.charts = Charts(self.metadata)
        self.model_names = models
        self.fake_data = {}
   
    def make_folders(self)  -> dict:
        self.syntheticdata_folder = f"{self.synthetic_folder}/data"
        self.checkpoint_folder = f"{self.synthetic_folder}/checkpoint"
        self.report_folder = f"{self.synthetic_folder}/report"
        
        folders = [self.syntheticdata_folder, self.checkpoint_folder, self.report_folder]
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f, exist_ok=True)

    def _gen_metadata(self, default_encoder) -> dict:
        if not os.path.exists(self.metadata_path):
            meta = table.Table(primary_key=self.id, field_names=set(self.df.columns.to_list()) - set(self.text_columns), 
            field_transformers={k:f'{default_encoder}' for k in self.category_columns})
            meta.fit(self.df.astype({ k:"category" for k in self.category_columns }))
            meta.to_json(self.metadata_path)
        if not os.path.exists(self.metadata_noise_path):
            meta_noise = table.Table(primary_key=self.id, field_names=set(self.df.columns.to_list()) - set(self.text_columns), 
            field_transformers={k:f'{default_encoder}_noised' for k in self.category_columns})
            meta_noise.fit(self.df.astype({ k:"category" for k in self.category_columns }))
            meta_noise.to_json(self.metadata_noise_path)
        return json.load(open(self.metadata_path, "r")), json.load(open(self.metadata_noise_path, "r"))
    
    def _define_params(self, model, **kwargs):
        params = {"table_metadata":self.metadata}
        params.update(kwargs)
        if "cuda" in model.__init__.__code__.co_varnames:
            params["cuda"] = self.cuda
        return params


    def fit_models(self, models: list[str]) -> dict:
        getter = op.itemgetter(*models)
        models = dict(
            **{ model_name: model(**self._define_params(model)) for model_name, model in zip(models,getter(MODELS))},
            **{ f"{model_name}_noise": model(**self._define_params(model, table_metadata=self.metadata_noised)) 
            for model_name, model in zip(models,getter(MODELS))}
        )
        
        #################################################
        ## Train Models
        #################################################
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        pt = partial(parallel_train, checkpoint_folder=self.checkpoint_folder, df = self.df)
        with mp.Pool(self.max_cpu_pool) as p:
            fitted_models = dict(list(p.map(pt, list(models.items()))))
        #################################################
        return fitted_models

    def _gen_synthetic(self, remaining_columns=None) -> dict:
        data_gen = {}
        models = self.fit_models(self.model_names)
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        #################################################
        ## Generate
        #################################################
        # require tuple(file_name, fitted_model)
        pw = partial(parallel_syn, df = self.df, syntheticdata_folder=self.syntheticdata_folder, n_sample=self.n_sample )
        with mp.Pool(self.max_cpu_pool) as p:
            data_gen = dict(p.map(pw, [(f"{model_name}_{self.n_sample}", model) for model_name, model in models.items()]))
        #################################################
        ## Generate with fixed columns
        #################################################
        if(remaining_columns):
            pwr = partial(parallel_syn, df = self.df, syntheticdata_folder=self.syntheticdata_folder, n_sample=self.n_sample, remaining_columns=remaining_columns )
            with mp.Pool(self.max_cpu_pool) as p:
                data_gen = dict(**data_gen, **dict(p.map(pwr, [(f"{model_name}_{self.n_sample}_wfixed_columns", model) for model_name, model in models.items()])))

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