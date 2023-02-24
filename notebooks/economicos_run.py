import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp


if __name__ == '__main__':

    df = pd.read_parquet('../datasets/economicos/raw/full_dedup_economicos_step0.parquet')


    category_columns=("property_type", "transaction_type", "state", "county", "rooms", "bathrooms", "m_built", "m_size", "source", )
    # TODO: Estudiar implicancia de valores nulos en categorias y numeros
    df_converted = df.fillna(dict(
            property_type = "None",
            transaction_type = "None",
            state = "None",
            county = "None",
            rooms = -1,
            bathrooms = -1,
            m_built = -1,
            m_size = -1,
            source = "None"
    )).fillna(-1).astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
    df_converted = df.replace(to_replace="None", value=np.nan).replace(to_replace=-1, value=np.nan).dropna().astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
    basedate = pd.Timestamp('2017-12-01')
    dtime = df_converted.pop("publication_date")
    df_converted["publication_date"] = dtime.apply(lambda x: (x - basedate).days)
    syn = Synthetic(df_converted, 
            id="url", 
            category_columns=category_columns,
            text_columns=("description", "price", "title", "address", "owner", ),
            exclude_columns=tuple(),
            synthetic_folder = "../datasets/economicos/synth",
            models=['copulagan', 'tvae', 'gaussiancopula', 'ctgan', 'smote-enc', 'tddpm_mlp'],
            n_sample = df_converted.shape[0],
            target_column="_price",
            max_cpu_pool=1
    )

    syn.process()
    syn.process_scores()
    print(syn._selectable_columns())
    print(syn.train.loc[:, syn._selectable_columns()])
    
    print(syn.current_metrics())
    
    

    # remaining_columns=('view','condition','waterfront'),
    # syn.process()
    #syn.process(additional_parameters={"smote-enc": {"frac_lam_del": 0.3}})
    #syn.process(additional_parameters={"smote-enc": {"frac_lam_del": 0.5, "k_neighbours": 1}})
    #syn.process(additional_parameters={"smote-enc": {"frac_lam_del": 0.1, "k_neighbours": 1}})
    #syn.process(additional_parameters={"smote-enc": {"frac_lam_del": 1, "k_neighbours": 1}})
    #syn.process(additional_parameters={"smote-enc": {"k_neighbours": 20}})
    #syn.process(additional_parameters={"smote-enc": {"k_neighbours": 1}})
    #syn.process(additional_parameters={"smote-enc": {"k_neighbours": 10}})

    # syn.process_scores()
    # print(syn.scores)
    #print(syn.metric.get_scores(syn.fake_data, syn.report_folder))
