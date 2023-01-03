import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp


if __name__ == '__main__':

    df = pd.read_csv('../datasets/kingcounty/raw/kc_house_data.csv')

    syn = Synthetic(df,
                    id="id",
                    #category_columns=("condition", "floors", "grade", "view", "waterfront", "yr_built", "yr_renovated", "zipcode", "bathrooms", "bedrooms",),
                    category_columns=("condition", "floors", "grade", "view",
                                      "waterfront", "zipcode", "bathrooms", "bedrooms",),
                    synthetic_folder="../datasets/kingcounty/synth",
                    models=['tddpm_mlp', 'smote-enc'],
                    n_sample=21613,
                    max_cpu_pool=1,
                    target_column="price"
                    )

    syn.process()
    syn.process_scores()
    

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
