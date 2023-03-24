import numpy as np
from ..base.model_base import ModelInterface
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer, OrdinalEncoder, StandardScaler

import copy
from . import MLPDiffusion, ResNetDiffusion, GaussianMultinomialDiffusion, Trainer, FastTensorDataLoader
import zero
import torch
import os
from copy import deepcopy



def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    print(indices)
    print(X)
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

class SDVTABDDPM(ModelInterface):
    MODEL = None
    def __init__(self, table_metadata: dict, 
    target_column: str, 
    exclude_columns:set[str]=set(), seed=42, 
    df:pd.DataFrame=None, 
    gaussian_loss_type="mse", 
    scheduler="cosine", 
    lr = 0.0005015,
    weight_decay = 0.0,
    steps = 75000,
    num_timesteps = 10,
    batch_size=3750,
    model_params:dict = dict(
        rtdl_params=dict(
            dropout=0.0,
            d_layers=[1024, 512, 256]
        )
    ),
    checkpoint = None,
    ohetype = "OrdinalEncoder"
    ) -> None:
        super().__init__()
        self.metadata = table_metadata
        self.column_id = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "id" and col != target_column and col not in exclude_columns][0]
        self.cat_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "categorical" and col != target_column and col not in exclude_columns]
        self.num_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "numerical" and col != target_column and col not in exclude_columns]
        self.date_columns = [col for col, col_type in table_metadata["fields"].items() if col_type["type"] == "datetime" and col != target_column and col not in exclude_columns]
        self.num_columns = self.num_columns + self.date_columns
        self.target_column = target_column
        self.is_regression = table_metadata["fields"][target_column]["type"] != "categorical"
        self.seed = seed
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps = steps
        self.num_features = self._make_ohe_scaler(df, ohetype)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.batch_size = batch_size
        # models
        # un arreglo de los tamaños de cada categorías (elementos únicos)
        if df.loc[:, self.cat_columns].nunique().to_numpy().sum() == 0 or isinstance(self.ohe, OneHotEncoder):
            # https://github.com/rotot0/tab-ddpm/blob/236a502c8bf398b12f179c1ef9059d30fcea23ad/scripts/train.py#L110
            #categorical_sizes = np.array([0])
            categorical_sizes = df.loc[:, self.cat_columns].nunique().to_numpy()
        else:
            categorical_sizes = df.loc[:, self.cat_columns].nunique().to_numpy()
        # las categorías + las variables numericas
        model_params["d_in"] = self.n_num_features + np.sum(categorical_sizes)
        #model_params["d_in"] = self.n_num_features + self.ohe.get_feature_names_out().shape[0]
        model_params["num_classes"] = self.model_num_classes
        model_params["is_y_cond"] = self.is_y_cond
        self.checkpoint = checkpoint
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=categorical_sizes,
            num_numerical_features=self.n_num_features,
            denoise_fn=self.MODEL(**model_params),
            gaussian_loss_type=gaussian_loss_type,
            num_timesteps=num_timesteps,
            scheduler=scheduler,
            device=self.device
        )
    
    def _make_ohe_scaler(self, df: pd.DataFrame, ohetype = "OrdinalEncoder"):
        self.n_num_features = len(self.num_columns) + self.is_regression
        self.n_cat_features = len(self.cat_columns) 
        self.y_scaler = None

        if self.is_regression:
            self.sorted_columns = self.num_columns + [self.target_column] + self.cat_columns
        else:
            self.sorted_columns = self.num_columns + self.cat_columns


        if self.is_regression:
            X_num = df.loc[:, self.num_columns + [self.target_column]]
            y_train = df.loc[:, self.target_column]
            self.model_num_classes = 0
            self.is_y_cond = False

            # https://github.com/rotot0/tab-ddpm/blob/5ac62c686ab177afcf7ae97492e15ac99984a14a/lib/data.py#L344
            # self.y_scaler = StandardScaler().fit(self.y_train.values.reshape(-1, 1))
            self.y_scaler = QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(y_train.shape[0] // 30, 1000), 10),
                random_state=self.seed,
            ).fit(y_train.values.reshape(-1, 1))
        else:
            X_num = df.loc[:, self.num_columns]
            y_train = df.loc[:, self.target_column]
            self.model_num_classes=len(df.loc[:, self.target_column].unique())
            self.is_y_cond = False
        
        
        #self.scaler = MinMaxScaler().fit(X_train)
        # This is from the paper and also config files say that  it uses quantile
        # https://github.com/rotot0/tab-ddpm/blob/5ac62c686ab177afcf7ae97492e15ac99984a14a/lib/data.py#L218
        self.scaler = QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_num.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=self.seed,
        ).fit(X_num)

        self.cat_features = list(range(self.n_num_features, self.n_num_features+self.n_cat_features))
        X_cat = df.loc[:, self.cat_columns]
        if ohetype == "OrdinalEncoder":
            # https://github.com/rotot0/tab-ddpm/blob/5ac62c686ab177afcf7ae97492e15ac99984a14a/lib/data.py#L289
            unknown_value = np.iinfo('int64').max - 3
            self.ohe = OrdinalEncoder(
                handle_unknown='use_encoded_value',  # type: ignore[code]
                unknown_value=unknown_value,  # type: ignore[code]
                dtype='int64',  # type: ignore[code]
            ).fit(X_cat)
        else:
            self.ohe = OneHotEncoder(
                handle_unknown='ignore',
                sparse=False,
                dtype=np.float32
            ).fit(X_cat)

        
        return self.n_num_features + self.n_cat_features


    def fit(self, train: pd.DataFrame):
        self.diffusion.to(self.device)
        
        # Select just columns than are required 
        X_train = train.loc[:, self.sorted_columns]

        if self.is_regression:
            X_num = X_train.loc[:, self.num_columns + [self.target_column]]
        else:
            X_num = X_train.loc[:, self.num_columns]
        X_cat = X_train.loc[:, self.cat_columns]
        
        # Scale the y if is required
        self.y_train = train.loc[:, self.target_column].to_numpy()
        if self.y_scaler:
            y_train = self.y_scaler.transform(train.loc[:, self.target_column].to_numpy().reshape(-1, 1))
        else:
            y_train = train.loc[:, self.target_column].to_numpy()
        y_train = torch.from_numpy(y_train)


        # scale num values
        X_train_scaled = self.scaler.transform(X_num).astype(np.float32)
        # scale cats
        X_cat_ohe = self.ohe.transform(X_cat)
        # join num + cats
        X_train = torch.from_numpy(np.concatenate( [X_train_scaled, X_cat_ohe] , axis=1).astype(np.float32))
        
        # move model to device
        self.diffusion.to(self.device)
        
        
        def loader():
            while True:
                yield from FastTensorDataLoader(X_train, y_train, batch_size=self.batch_size, shuffle=True)
        train_loader = loader()
        self.diffusion.train()
        trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=self.lr,
            weight_decay=self.weight_decay,
            steps=self.steps,
            device=self.device,
            checkpoint=self.checkpoint
        )
        trainer.run_loop()
        self.diffusion.cpu()
        if self.checkpoint:
            self.diffusion.load_state_dict(torch.load(os.path.join(self.checkpoint, "best_model.pt")))

        
        
    def save(self, filepath: str):
        print(f"SAVING to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print("saved")

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def sample(self, n_sample: int):
        self.diffusion.to(self.device)
        self.diffusion.eval()
        _, empirical_class_dist = torch.unique(torch.from_numpy(self.y_train), return_counts=True)
        x_gen, y_gen = self.diffusion.sample_all(n_sample, self.batch_size, empirical_class_dist.float(), ddim=False)
        X_num = self.scaler.inverse_transform(x_gen[:, :self.n_num_features].numpy())
        ## TODO: Pending treatment for round values (discrete)

        if self.cat_columns:
            x_cat_tmp = x_gen[:, self.n_num_features:]
            if isinstance(self.ohe, OneHotEncoder):
                x_cat_tmp = to_good_ohe(self.ohe, x_cat_tmp)
            X_cat = self.ohe.inverse_transform(x_cat_tmp)



        X_gen = np.concatenate([X_num, X_cat], axis=1)
        return pd.DataFrame(X_gen, columns=self.sorted_columns[:X_gen.shape[1]]).reset_index().rename(columns={"index":self.column_id})


class SDV_MLP(SDVTABDDPM):
    MODEL = MLPDiffusion


class SDV_REST(SDVTABDDPM):
    MODEL = ResNetDiffusion
