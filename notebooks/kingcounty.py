import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
if __name__ == '__main__': 
    
    df = pd.read_csv('../datasets/kingcounty/raw/kc_house_data.csv');
    df.sample(3)
    
    syn = Synthetic(df, 
            id="id",
            #category_columns=("condition", "floors", "grade", "view", "waterfront", "yr_built", "yr_renovated", "zipcode", "bathrooms", "bedrooms",),
            category_columns=("condition", "floors", "grade", "view", "waterfront", "zipcode", "bathrooms", "bedrooms",),
            synthetic_folder = "../datasets/kingcounty/synth",
            models=MODELS.keys(),
            n_sample = 21613,
            max_cpu_pool=8
    )
    syn.process(remaining_columns=('view','condition','waterfront'))
    syn.process_scores()