import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp
import os

def test_train(args):
    lrc, ntc, sts, btsc, rtdlc, syn, df = args
    #notebooks/economicos_good/2e-06_10_100000_5000_1024-512-256
    
    checkpoint = "con_fechas_600k"
    if os.path.exists(f"{checkpoint}/final_model.pt") or os.path.exists(f"{checkpoint}/exit"):
        return (checkpoint, 1)    
    model = SDV_MLP(syn.metadata, 
                    "_price", 
                    exclude_columns=syn.exclude_columns, 
                    df=df, 
                    batch_size=btsc, 
                    steps=sts, 
                    checkpoint=checkpoint,
                    num_timesteps=ntc,
                    weight_decay=0.0,
                    lr=lrc,
                    model_params=dict(rtdl_params=dict(
                        dropout=0.0,
                        d_layers=rtdlc
                    ))
                    )
    model.fit(syn.train.sample(frac=1))
    model.save(f"{checkpoint}/final_model.pt")
    return (checkpoint, 1)

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
    #lrs = np.linspace(2e-6, 2e-3, 10)
    lrs = [2e-6]#np.linspace(2e-6, 2e-3, 5)
    num_timesteps = [10] #np.linspace(10, 1000, 3, dtype=int)
    batch_size = [5000] #np.linspace(2500, 5000, 3, dtype=int)
    steps = [500]#[10000000]#np.linspace(150000, 500000, 5, dtype=int)
    rtdl_params = [
        [1024, 512, 256],
        #[512, 256],
        #[256, 128],
        #[256, 128, 128],
        #[256, 128, 128, 128]        
    ]
    try:
        torch.multiprocessing.set_start_method('spawn')
    except:
        pass
    with mp.Pool(1) as p:
        fitted_models = dict(list(p.map(test_train, itertools.product(lrs, num_timesteps, steps, batch_size, rtdl_params, [syn], [df_converted]))))

   
