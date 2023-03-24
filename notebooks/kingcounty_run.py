import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp
import os


if __name__ == '__main__':

    df = pd.read_csv('../datasets/kingcounty/raw/kc_house_data.csv')

    syn = Synthetic(df,
                    id="id",
                    #category_columns=("condition", "floors", "grade", "view", "waterfront", "yr_built", "yr_renovated", "zipcode", "bathrooms", "bedrooms",),
                    category_columns=("condition", "floors", "grade", "view",
                                      "waterfront", "zipcode", "bathrooms", "bedrooms",),
                    synthetic_folder="../datasets/kingcounty/synth-d",
                    models=MODELS.keys(),
                    n_sample=21613,
                    max_cpu_pool=4,
                    target_column="price",
                    )

    syn.process()
    syn.process_scores()
    

    best_model = "tddpm_mlp_21613"
    second_best_model = "smote-enc_21613"

    from syntheticml.data.charts import Charts
    
    folder_path = f"../docs/tesis/imagenes/kingcounty/{best_model}"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    for fig in syn.get_charts(best_model, {'date', 'id', 'zipcode', 'lat', 'long', 'yr_renovated'}):
        if fig:
            file_name = f'{fig.layout.title.text.replace(":","").replace(" ","_").lower()}.svg'
            fig.write_image(f"{folder_path}/{file_name}")
            display(fig.show("png"))