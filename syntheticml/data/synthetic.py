
from operator import imod
import pandas as pd
from sdv.tabular import GaussianCopula, TVAE, CTGAN, CopulaGAN
from sdv.metadata import table
import os
import json
import operator as op
import plotly.graph_objects as go
import torch

from sklearn.model_selection import GridSearchCV, train_test_split
import multiprocessing as mp
from functools import partial

import numpy as np
from copy import copy


from .metrics import Metrics
from .charts import Charts
from ..models.base.model_base import ModelInterface

from ..models.smote.sdv import SDVSMOTE

from ..models.tab_ddpm.sdv import SDV_MLP

MODELS = {
    "copulagan": CopulaGAN,
    "tvae": TVAE,
    "gaussiancopula": GaussianCopula,
    "ctgan": CTGAN,
    "smote-enc": SDVSMOTE,
    "tddpm_mlp": SDV_MLP
}


def parallel_train(model_item: tuple[str, ModelInterface], checkpoint_folder: str, df: pd.DataFrame) -> ModelInterface:
    model_name, model = model_item
    checkpoint = f"{checkpoint_folder}/{model_name}.ckp"
    if not os.path.exists(checkpoint):
        print("="*30)
        print("Fitting")
        print(checkpoint)
        print("="*30)
        model.fit(df)
        model.save(checkpoint)
        print("="*30)
        print("saved fit")
        print("="*30)
    return (model_name, (checkpoint, model,))


def parallel_syn(params: tuple[str, str, tuple[str, str]], df: pd.DataFrame, syntheticdata_folder: str, n_sample: int, remaining_columns: tuple[str] = tuple(), additional_parameters: dict = {}) -> pd.DataFrame:
    file_name, model_name, (model_checkpoint, model) = params
    model = model.load(model_checkpoint)
    syndata_path = f"{syntheticdata_folder}/{file_name}.parquet"
    p = {}
    if not os.path.exists(syndata_path):
        f = model.sample_remaining_columns if remaining_columns else model.sample
        p = {}
        if "output_file_path" in f.__code__.co_varnames:
            p["output_file_path"] = f"{syndata_path}.tmp"
        if "df" in f.__code__.co_varnames:
            p["df"] = df.loc[:, remaining_columns] if remaining_columns else df
        if "num_rows" in f.__code__.co_varnames:
            p["num_rows"] = n_sample
        if "n_sample" in f.__code__.co_varnames:
            p["n_sample"] = n_sample
        if model_name in additional_parameters:
            for pp in additional_parameters[model_name].keys():
                if pp in f.__code__.co_varnames:
                    p[pp] = additional_parameters[model_name][pp]
        # return (model_name, p)
        new_data = f(**p)
        print(new_data.head(1))
        if len(new_data.columns) > 2:
            new_data.to_parquet(syndata_path, compression='snappy', engine='pyarrow', version='2.6')
    return (file_name, pd.read_parquet(syndata_path))


class Synthetic:
    def __init__(self,
                 df: pd.DataFrame,
                 target_column: str,
                 category_columns: tuple[str],
                 id: str,
                 synthetic_folder: str,
                 text_columns: tuple[str] = (),
                 models: tuple[str] = [],
                 n_sample: int = 0, exclude_columns: tuple[str] = tuple(),
                 default_encoder="FrequencyEncoder",
                 random_state=42,
                 max_cpu_pool=None) -> None:
        self.df = df
        self.random_state = random_state
        
        self.category_columns = category_columns
        self.primary_key = id
        self.target_column = target_column
        self.n_sample = n_sample or df.shape[0]
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
        if not os.path.exists(f"{self.split}/train.parquet"):
            self.train, self.hold = train_test_split(
                df, test_size=0.2, random_state=random_state)
        else:
            self.train = pd.read_parquet(f"{self.split}/train.parquet")
            self.hold = pd.read_parquet(f"{self.split}/hold.parquet")
        self.metadata, self.metadata_noised = self._gen_metadata(
            default_encoder)
        self.metric = Metrics(self.df, self.train, self.hold, self.metadata, includes=list(set(self.df.columns) - set(self.exclude_columns) - set(self.df.select_dtypes(include=np.datetime64).columns)))
        self.charts = Charts(self.metadata)
        self.model_names = models
        self.fake_data = {}
        self.save_hold()

    def save_hold(self) -> None:
        train_path = f"{self.split}/train.parquet"
        hold_path = f"{self.split}/hold.parquet"
        if not os.path.exists(train_path):
            print(self.train.dtypes)
            self.train.to_parquet(train_path)
        if not os.path.exists(hold_path):
            self.hold.to_parquet(hold_path)

    def make_folders(self) -> None:
        self.syntheticdata_folder = f"{self.synthetic_folder}/data"
        self.checkpoint_folder = f"{self.synthetic_folder}/checkpoint"
        self.report_folder = f"{self.synthetic_folder}/report"
        self.split = f"{self.synthetic_folder}/split"

        folders = [self.syntheticdata_folder,
                   self.checkpoint_folder, self.report_folder, self.split]
        for f in folders:
            if not os.path.exists(f):
                os.makedirs(f, exist_ok=True)

    def _gen_metadata(self, default_encoder) -> dict:
        if not os.path.exists(self.metadata_path):
            meta = table.Table(primary_key=self.primary_key, field_names=set(self.train.columns.to_list()) - set(self.text_columns),
                               field_transformers={k: f'{default_encoder}' for k in self.category_columns})
            meta.fit(self.train.astype(
                {k: "category" for k in self.category_columns}))
            meta.to_json(self.metadata_path)
        if not os.path.exists(self.metadata_noise_path):
            meta_noise = table.Table(primary_key=self.primary_key, field_names=set(self.train.columns.to_list()) - set(self.text_columns),
                                     field_transformers={k: f'{default_encoder}_noised' for k in self.category_columns})
            meta_noise.fit(self.train.astype(
                {k: "category" for k in self.category_columns}))
            meta_noise.to_json(self.metadata_noise_path)
        return json.load(open(self.metadata_path, "r")), json.load(open(self.metadata_noise_path, "r"))

    def _define_params(self, model, **kwargs):
        params = {"table_metadata": self.metadata}
        for key, value in kwargs.items():
            if key in model.__init__.__code__.co_varnames:
                params[key] = value
        if "cuda" in model.__init__.__code__.co_varnames:
            params["cuda"] = self.cuda
        if "df" in model.__init__.__code__.co_varnames:
            params["df"] = self.df
        return params

    def fit_models(self, models: list[str]) -> dict:
        getter = op.itemgetter(*models)
        base_params = dict(
            target_column=self.target_column,
            exclude_columns=self.exclude_columns
        )
        models = dict(
            **{model_name: model(**self._define_params(model, **{**{"checkpoint": os.path.join(self.checkpoint_folder, model_name)}, **base_params})) for model_name, model in zip(models, getter(MODELS))},
            **{f"{model_name}_noise": model(**self._define_params(model, **{**{"checkpoint": os.path.join(self.checkpoint_folder, f"{model_name}_noise"), "table_metadata": self.metadata_noised}, **base_params}))
                for model_name, model in zip(models, getter(MODELS))}
        )

        #################################################
        # Train Models
        #################################################
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        pt = partial(parallel_train,
                     checkpoint_folder=self.checkpoint_folder, df=self.train)
        if self.max_cpu_pool > 1:
            with mp.Pool(self.max_cpu_pool) as p:
                fitted_models = dict(list(p.map(pt, list(models.items()))))
        else:
            fitted_models = dict(list(map(pt, list(models.items()))))
        #################################################
        return fitted_models

    def _gen_synthetic(self, remaining_columns=None, additional_parameters={}) -> dict:
        data_gen = {}
        models = self.fit_models(self.model_names)
        try:
            torch.multiprocessing.set_start_method('spawn')
        except:
            pass
        #################################################
        # Generate
        #################################################
        # require tuple(file_name, fitted_model)
        # Here df is used instead of train since df is the whole set of elements

        def fn_model_name(model_name):
            final_name = f"{model_name}_{self.n_sample}"
            if remaining_columns:
                final_name += "_wfixed_columns"
            if additional_parameters and model_name in additional_parameters:
                final_name += "_" + \
                    "_".join(
                        [f"{k}={v}" for k, v in additional_parameters[model_name].items()])
            return final_name

        if additional_parameters:
            models = dict(
                filter(lambda t: t[0] in additional_parameters.keys(), models.items()))

        pw = partial(parallel_syn, df=self.df, syntheticdata_folder=self.syntheticdata_folder,
                     n_sample=self.n_sample, remaining_columns=remaining_columns, additional_parameters=additional_parameters)

        if self.max_cpu_pool > 1:
            with mp.Pool(self.max_cpu_pool) as p:
                data_gen = dict(p.map(pw, [(fn_model_name(
                    model_name), model_name, model,) for model_name, model in models.items()]))
        else:
            data_gen = dict(list(map(pw, [(fn_model_name(
                    model_name), model_name, model,) for model_name, model in models.items()])))

        return data_gen

    def _selectable_columns(self) -> list:
        return list(set(self.train.columns) - {self.primary_key, } - set(self.text_columns) - set(self.exclude_columns))

    def process(self, remaining_columns=None, additional_parameters={}) -> None:
        self.fake_data = dict(
            **self.fake_data, **self._gen_synthetic(remaining_columns, additional_parameters))
        # Quito elementos que hayan generado 0 data
        iters = list(self.fake_data.items())
        for model_name, pd_data in iters:
            if pd_data.shape[0] == 0:
                del self.fake_data[model_name]
    
    def get_details(self):
        return self.metric.get_details(list(self.fake_data.keys()), self.report_folder)

    def grid_search(self, model) -> None:
        raise Exception("No Implemented")

    def process_scores(self):
        self.scores, self.reports = self.metric.get_scores(
            self.fake_data, self.report_folder)
        _, self.privacy_metrics = self.metric.calculate_privacy(
            self.fake_data, self.report_folder)

    def current_metrics(self) -> pd.DataFrame:
        return self.metric.get_metrics(self.train.loc[:, self._selectable_columns()])

    def get_metric(self, method:  str, replace_rule: dict = None) -> pd.DataFrame:
        if replace_rule:
            return self.metric.get_metrics(self.fake_data[method].loc[:, self._selectable_columns()].replace(replace_rule))
        else:
            return self.metric.get_metrics(self.fake_data[method].loc[:, self._selectable_columns()])

    def get_metrics_fake(self) -> dict:
        return {
            k: self.metric.get_metrics(fake_data)
            for k, fake_data in self.fake_data.items()
        }

    def get_charts(self, model_key: str, exclude_columns: set[str] = set(), privacy_cut=0.05) -> list[go.Figure]:

        serie_real = self.privacy_metrics.loc[model_key, :]["DCR TH"]
        serie_fake = self.privacy_metrics.loc[model_key, :]["DCR SH"]

        serie_real = serie_real[serie_real <=
                                np.quantile(serie_real, privacy_cut)]
        serie_fake = serie_fake[serie_fake <=
                                np.quantile(serie_fake, privacy_cut)]

        privacy_chart = self.charts.privacy(serie_real, serie_fake, model_key)

        if not exclude_columns:
            exclude_columns = set(self.text_columns) & set(
                self.exclude_columns)
        return [privacy_chart] + self.charts.charts(self.train, { model_key: self.fake_data[model_key] }, exclude_columns)

    def get_multiple_charts(self, models: list[str], exclude_columns: set[str] = set(), privacy_cut=0.05) -> list[go.Figure]:

        serie_real = self.privacy_metrics.loc[models[0], :]["DCR TH"]
        serie_real = serie_real[serie_real <=
                                np.quantile(serie_real, privacy_cut)]
        
        serie_fakes = []
        for model_key in models:
            serie_fake = self.privacy_metrics.loc[model_key, :]["DCR SH"]
            serie_fake = serie_fake[serie_fake <=
                                np.quantile(serie_fake, privacy_cut)]
            serie_fakes.append((model_key, serie_fake,))

        privacy_chart = self.charts.privacys(serie_real, dict(serie_fakes))

        return [privacy_chart] + self.charts.charts(self.train, {model_key: self.fake_data[model_key] for model_key in models}, exclude_columns)