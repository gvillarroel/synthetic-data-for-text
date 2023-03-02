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
    print(df_converted.shape)
    basedate = pd.Timestamp('2017-12-01')
    dtime = df_converted.pop("publication_date")
    df_converted["publication_date"] = dtime.apply(lambda x: (x - basedate).days)
    syn = Synthetic(df_converted, 
            id="url", 
            category_columns=category_columns,
            text_columns=("description", "price", "title", "address", "owner", ),
            exclude_columns=tuple(),
            synthetic_folder = "../datasets/economicos/synthb",
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
