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
    checkpoint = "economicos_good2/" +  "_".join(
            map(str, [lrc, ntc, sts, btsc, "-".join(map(str, rtdlc))]))
    checkpoint = "economicos_good2/con_fechas"
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
    model.fit(syn.train)
    model.save(f"{checkpoint}/final_model.pt")
    return (checkpoint, 1)


if __name__ == '__main__':

    df = pd.read_parquet('../datasets/economicos/synth/split/train.parquet')

    category_columns=("property_type", "transaction_type", "state", "county", "rooms", "bathrooms", "m_built", "m_size", "source", )
    # TODO: Estudiar implicancia de valores nulos en categorias y numeros
    df_converted = df.dropna().astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
    syn = Synthetic(df_converted, 
            id="url", 
            category_columns=category_columns,
            text_columns=("description", "price", "title", "address", "owner",),
            exclude_columns=tuple(),
            synthetic_folder = "../datasets/economicos/synth",
            models=['copulagan', 'tvae', 'gaussiancopula', 'ctgan', 'smote-enc'],
            n_sample = df.shape[0],
            target_column="_price"
    )
    
    #lrs = np.linspace(2e-6, 2e-3, 10)
    lrs = [2e-6]#np.linspace(2e-6, 2e-3, 5)
    num_timesteps = [10] #np.linspace(10, 1000, 3, dtype=int)
    batch_size = [5000] #np.linspace(2500, 5000, 3, dtype=int)
    steps = [1000]#[10000000]#np.linspace(150000, 500000, 5, dtype=int)
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

    # print(model.X_train[:1])
    # print(model.y_train[:1])
    #
    #X_sample, Y_sample = model.diffusion.sample(10, y_count.float())
#
#
    # print(X_sample[:1])
    # print(Y_sample['y'])

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
