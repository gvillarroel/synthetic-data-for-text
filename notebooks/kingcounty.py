import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp

def test_train(args):
    lrc, ntc, sts, btsc, rtdlc, syn, df = args
    checkpoint = "ordinal-3/" +  "_".join(
            map(str, [lrc, ntc, sts, btsc, "-".join(map(str, rtdlc))]))
    model = SDV_MLP(syn.metadata, 
                    "price", 
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

    df = pd.read_csv('../datasets/kingcounty/raw/kc_house_data.csv')

    syn = Synthetic(df,
                    id="id",
                    #category_columns=("condition", "floors", "grade", "view", "waterfront", "yr_built", "yr_renovated", "zipcode", "bathrooms", "bedrooms",),
                    category_columns=("condition", "floors", "grade", "view",
                                      "waterfront", "zipcode", "bathrooms", "bedrooms",),
                    synthetic_folder="../datasets/kingcounty/synth",
                    models=['tddpm_mlp'],
                    n_sample=21613,
                    max_cpu_pool=1,
                    target_column="price"
                    )

    #lrs = np.linspace(2e-6, 2e-3, 10)
    lrs = np.linspace(2e-6, 2e-3, 5)
    num_timesteps = [10, 100]#np.linspace(10, 1000, 3, dtype=int)
    batch_size = np.linspace(2500, 5000, 3, dtype=int)
    steps = np.linspace(50000, 100000, 3, dtype=int)
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
    with mp.Pool(8) as p:
        fitted_models = dict(list(p.map(test_train, itertools.product(lrs, num_timesteps, steps, batch_size, rtdl_params, [syn], [df]))))

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
