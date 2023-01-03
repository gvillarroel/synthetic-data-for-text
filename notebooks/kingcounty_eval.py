import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder




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

    model = SDV_MLP.load("ordinal-3/0.0005015000000000001_10_50000_2500_1024-512-256/final_model.pt")
   
    print(syn.train[["id"] + model.sorted_columns].head(3).T)
    
    dfy = model.sample(3).astype(syn.train[["id"] + model.sorted_columns].dtypes)
    print(dfy.T)

    
