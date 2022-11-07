from syntheticml.models.smote.smote import MySMOTENC, MySMOTE
import numpy as np
from ..base.model_base import ModelInterface
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import copy




class SDVSMOTE(ModelInterface):
    def __init__(self, table_metadata: dict, target_column: str, k_neighbours:int=5, exclude_columns:set[str]=set(), seed=42) -> None:
        super().__init__()
        self.metadata = table_metadata
        self.column_id = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "id" and col != target_column and col not in exclude_columns][0]
        self.cat_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "categorical" and col != target_column and col not in exclude_columns]
        self.num_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "numerical" and col != target_column and col not in exclude_columns]
        self.k_neighbours = k_neighbours
        self.target_column = target_column
        self.is_regression = table_metadata["fields"][target_column]["type"] != "categorical"
        self.seed = seed    

    def fit(self, train: pd.DataFrame):
        self.train = train

        self.n_num_features = len(self.num_columns)
        n_cat_features = len(self.cat_columns)

        if self.is_regression:
            X_train = train.loc[:, self.num_columns + [self.target_column]]
            self.y_train = np.where( train.loc[:, self.target_column] > np.median(train.loc[:, self.target_column]), 1, 0)
        else:
            X_train = train.drop(columns=[self.target_column])
            self.y_train = train.loc[:, self.target_column]
        
        self.cat_features = list(range(self.n_num_features, self.n_num_features+n_cat_features))
        self.scaler = MinMaxScaler().fit(X_train)
        X_train_scaled = self.scaler.transform(X_train).astype(object)
        self.X_train_scaled_cat = np.concatenate( [X_train_scaled, train.loc[:, self.cat_columns].to_numpy()] , axis=1, dtype=object)
        self.sorted_columns = self.num_columns + [self.target_column] + self.cat_columns

    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def sample_remaining_columns(self, df: pd.DataFrame, output_file_path: str):
        return pd.DataFrame([], columns=[self.column_id] + self.sorted_columns)

    def sample(self, n_sample: int, output_file_path: str):
        frac_samples = (self.train.shape[0] + n_sample) / self.train.shape[0]
        frac_lam_del = 0.0
        lam1 = 0.0 + frac_lam_del / 2
        lam2 = 1.0 - frac_lam_del / 2

        y_cats, y_counts = np.unique(self.y_train, return_counts=True)
        strat = dict(zip(y_cats, (y_counts*(frac_samples)).astype(int) ))

        if len(self.cat_columns) > 0:
            sm = MySMOTENC(
                lam1=lam1,
                lam2=lam2,
                random_state=self.seed,
                k_neighbors=self.k_neighbours,
                categorical_features=self.cat_features,
                sampling_strategy=strat
            )
        else:
            sm = MySMOTE(
                    lam1=lam1,
                    lam2=lam2,
                    random_state=self.seed,
                    k_neighbors=self.k_neighbours,
            )
        X_res, y_res = sm.fit_resample(self.X_train_scaled_cat, self.y_train)
        outsample = X_res[self.X_train_scaled_cat.shape[0]:]
        num_outsample = self.scaler.inverse_transform(outsample[:, :self.n_num_features+1])
        cat_outsample = outsample[:, self.n_num_features+1:]
        complete_outsample = np.concatenate([num_outsample, cat_outsample], axis=1)
        return pd.DataFrame(complete_outsample, columns=self.sorted_columns).reset_index().rename(columns={"index":self.column_id})
        
