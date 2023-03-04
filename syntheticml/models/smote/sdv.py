from syntheticml.models.smote.smote import MySMOTENC, MySMOTE
import numpy as np
from ..base.model_base import ModelInterface
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import copy
from functools import partial




class SDVSMOTE(ModelInterface):
    def __init__(self, table_metadata: dict, target_column: str, exclude_columns:set[str]=set(), seed=42, df:pd.DataFrame=None, max_data=50000, batch_size=5000) -> None:
        super().__init__()
        self.metadata = table_metadata
        self.column_id = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "id" and col != target_column and col not in exclude_columns][0]
        self.cat_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "categorical" and col != target_column and col not in exclude_columns]
        self.num_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] in ["numerical","datetime"] and col != target_column and col not in exclude_columns]
        self.date_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "datetime" and col != target_column and col not in exclude_columns]
        self.target_column = target_column
        self.is_regression = table_metadata["fields"][target_column]["type"] != "categorical"
        self.seed = seed
        self.max_data = max_data
        self.batch_size = batch_size

        self.to_datetime = partial(pd.to_datetime)

        self._make_ohe_scaler(df)

    def _make_ohe_scaler(self, df: pd.DataFrame):
        self.n_num_features = len(self.num_columns)
        n_cat_features = len(self.cat_columns)

        if self.is_regression:
            X_train = df.loc[:, self.num_columns + [self.target_column]]
            self.y_train = np.where( df.loc[:, self.target_column] > np.median(df.loc[:, self.target_column]), 1, 0)
        else:
            X_train = df.drop(columns=[self.target_column])
            self.y_train = df.loc[:, self.target_column]
        
        self.cat_features = list(range(self.n_num_features, self.n_num_features+n_cat_features))
        self.scaler = MinMaxScaler().fit(X_train.apply(pd.to_numeric))
        self.ohe = OrdinalEncoder(dtype=np.float16).fit(df.loc[:, self.cat_columns])

    def fit(self, train: pd.DataFrame):
        self.train = train
        print(self.num_columns + [self.target_column] + self.cat_columns)
        if self.is_regression:
            self.sorted_columns = self.num_columns + [self.target_column] + self.cat_columns
        else:
            self.sorted_columns = self.num_columns + self.cat_columns
    
    def get_scaled(self):
        train = self.train.sample(min(self.max_data, self.train.shape[0]))
        #train = self.train
        if self.is_regression:
            X_train = train.loc[:, self.num_columns + [self.target_column]]
            y_train = np.where( train.loc[:, self.target_column] > np.median(train.loc[:, self.target_column]), 1, 0)
            cats, counts = np.unique(y_train, return_counts=True)
            m_count = min(counts)
            indexes = np.concatenate(X_train.groupby(y_train, group_keys=False).apply(lambda x: x.sample(m_count).index).values)
            y_train = pd.DataFrame(y_train, index=X_train.index).loc[indexes,:].values
            X_train = X_train.loc[indexes,:]
            X_cat = self.ohe.transform(train.loc[indexes, self.cat_columns])
        else:
            X_train = train.drop(columns=[self.target_column])
            y_train = train.loc[:, self.target_column]
            X_cat = self.ohe.transform(train.loc[:, self.cat_columns])
        

        X_train_scaled = self.scaler.transform(X_train.apply(pd.to_numeric)).astype(object)
        X_train_scaled_cat = np.concatenate( [X_train_scaled, X_cat] , axis=1, dtype=object)

        return X_train_scaled_cat, y_train
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def sample_remaining_columns(self, df: pd.DataFrame, output_file_path: str):
        return self._sample(df.shape[0], output_file_path=output_file_path)

    def _sample(self, n_sample: int, output_file_path: str, frac_lam_del: float = 0.0, k_neighbours:int=5):
        scaled, y_true = self.get_scaled()
        frac_samples = (scaled.shape[0] + n_sample) / scaled.shape[0]
        lam1 = 0.0 + frac_lam_del / 2
        lam2 = 1.0 - frac_lam_del / 2

        y_cats, y_counts = np.unique(y_true, return_counts=True)
        strat = dict(zip(y_cats, (y_counts*(frac_samples)).astype(int) ))
        
        if len(self.cat_columns) > 0:
            sm = MySMOTENC(
                lam1=lam1,
                lam2=lam2,
                random_state=self.seed,
                k_neighbors=k_neighbours,
                categorical_features=self.cat_features,
                sampling_strategy=strat
            )
        else:
            sm = MySMOTE(
                    lam1=lam1,
                    lam2=lam2,
                    random_state=self.seed,
                    k_neighbors=k_neighbours,
            )
        X_res, y_res = sm.fit_resample(scaled, y_true)
        outsample = X_res[scaled.shape[0]:]
        num_outsample = self.scaler.inverse_transform(outsample[:, :self.scaler.n_features_in_])
        cat_outsample = self.ohe.inverse_transform(outsample[:, self.scaler.n_features_in_:])
        complete_outsample = np.concatenate([num_outsample, cat_outsample], axis=1)
        return pd.DataFrame(complete_outsample, columns=self.sorted_columns)
        
    def sample(self, n_sample: int, output_file_path: str, frac_lam_del: float = 0.0, k_neighbours:int=5):
        ns = self._sample(min(n_sample, self.batch_size), output_file_path, frac_lam_del, k_neighbours)        
        while ns.shape[0] < n_sample:
            print(f"sampling smote left:{n_sample-ns.shape[0]}")
            ns = pd.concat([ns, self._sample(min(n_sample+1-ns.shape[0], self.batch_size), output_file_path, frac_lam_del, k_neighbours)]) 
        
        ns = pd.concat([ ns.loc[:, self.date_columns].apply(self.to_datetime), ns.loc[:, list(set(ns.columns) - set(self.date_columns))]], axis=1)
        return ns.reset_index().rename(columns={"index":self.column_id})