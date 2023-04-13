import pandas as pd
from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport
import json
from statsmodels.stats.descriptivestats import describe
from scipy import stats
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors



class Metrics:
    def __init__(self, 
    real_data : pd.DataFrame, train_data: pd.DataFrame, 
    hold_data: pd.DataFrame, 
    metadata: dict,
    includes: list=[]
    ) -> None:
        self.real_data = real_data
        self.train_data = train_data
        self.hold_data = hold_data
        self.metadata = metadata
        if not includes:
            self.includes = list(real_data.columns)
        else:
            self.includes = includes
        self._make_ohe(real_data)

    def _make_ohe(self, real_data):
        self.columns_number = tuple(set(real_data.select_dtypes(include=['number']).columns) & set(self.includes))
        self.columns_category = tuple(set(real_data.select_dtypes(include=['category']).columns) & set(self.includes))
        self.ohe = OneHotEncoder().fit(real_data.loc[:, self.columns_category])
        self.mms = MinMaxScaler().fit(real_data.loc[:, self.columns_number])

    #https://github.com/rotot0/tab-ddpm/blob/236a502c8bf398b12f179c1ef9059d30fcea23ad/scripts/resample_privacy.py
    # distance to closest record (DCR)
    # nearest neighbour distance ratio (NNDR)
    def privacy(self, df_base: pd.DataFrame, df_target: pd.DataFrame):
        
        X_base = self.mms.transform(df_base.loc[:, self.columns_number])
        X_target = self.mms.transform(df_target.loc[:, self.columns_number])
        if self.columns_category:
            
            X_cat_real = self.ohe.transform(df_base.loc[:, self.columns_category]) / np.sqrt(2)
            X_cat_fake = self.ohe.transform(df_target.loc[:, self.columns_category]) / np.sqrt(2)
            X_base = np.concatenate([X_base, X_cat_real.todense()], axis=1)
            X_target = np.concatenate([X_target, X_cat_fake.todense()], axis=1)

        #dist_rf = pairwise_distances(X_fake, Y=X_real, metric='l2', n_jobs=-1)
        dist_rf = pairwise_distances(X_base[:min(50000, X_base.shape[0])], Y=X_target[:min(50000, X_base.shape[0])], metric='minkowski', n_jobs=-1)
        
        smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
        smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]   
        min_dist_rf = np.array([i[0] for i in smallest_two_rf])
        #fifth_perc_rf = np.percentile(min_dist_rf,5)

        return min_dist_rf
    
    def calculate_privacy(self, fake_data: dict, report_folder):
        privacy_path = f"{report_folder}/privacy.npy"
        frame = []
        dists = []
        columns_to_compare = list(set(self.train_data.columns) & set(self.includes) & set(self.hold_data.columns))
        if os.path.exists(privacy_path):
            dist_TH = np.load(privacy_path)
        else:
            dist_TH = self.privacy(self.train_data.loc[:, columns_to_compare], df_target=self.hold_data.loc[:, columns_to_compare])
            np.save(privacy_path, dist_TH)
        #dist_HT = self.privacy(self.hold_data, self.train_data)
        for model_name, df_fake in fake_data.items():
            columns_to_compare = list(set(self.train_data.columns) & set(self.includes) & set(df_fake.columns))
            privacy_path_model_ST = f"{report_folder}/privacy_{model_name}_ST.npy"
            privacy_path_model_SH = f"{report_folder}/privacy_{model_name}_SH.npy"
            if os.path.exists(privacy_path_model_ST):
                dist_ST = np.load(privacy_path_model_ST)
            else:
                dist_ST = self.privacy(df_fake, df_target=self.train_data)
                np.save(privacy_path_model_ST, dist_ST)
            if os.path.exists(privacy_path_model_SH):
                dist_SH = np.load(privacy_path_model_SH)
            else:
                dist_SH = self.privacy(df_fake, df_target=self.hold_data)
                np.save(privacy_path_model_SH, dist_SH)
            
            #dist_HS = self.privacy(self.hold_data, df_fake)
            
            frame.append({
                "name": model_name, 
                "DCR ST min": np.min(dist_ST),
                "DCR ST median": np.median(dist_ST),
                "DCR ST 5th": np.quantile(dist_ST, 0.05), 
                "DCR SH min": np.min(dist_SH),
                "DCR SH median": np.median(dist_SH),
                "DCR SH 5th": np.quantile(dist_SH, 0.05), 
                "DCR TH min": np.min(dist_TH),
                "DCR TH median": np.median(dist_TH),
                "DCR TH 5th": np.quantile(dist_TH, 0.05),
            })
            dists.append({
                "name": model_name, 
                "DCR ST": dist_ST,
                "DCR SH": dist_SH,
                "DCR TH": dist_TH,
                #"DCR HS": dist_HS,
                #"DCR HT": dist_HT
            })
        return pd.DataFrame(frame).set_index("name"), pd.DataFrame(dists).set_index("name") 

    def get_details(self, models: list[str], report_folder):
        result = {}
        for model_name in models:
            
            report_path = f"{report_folder}/{model_name}.rpt"
            diagnostic_path = f"{report_folder}/{model_name}_diagnostic.rpt"
            rpt = QualityReport().load(report_path)
            dia = DiagnosticReport().load(diagnostic_path)
            result[model_name] = dict(
                report = dict(
                    column_shape = rpt.get_details('Column Shapes'),
                    column_pair_trends = rpt.get_details('Column Pair Trends'),
                ),
                diagnostic = dict(
                    synthesis = dia.get_details('Synthesis'),
                    coverage = dia.get_details('Coverage'),
                    boundaries = dia.get_details('Boundaries'),
                )
            )
        return result


    def get_scores(self, fake_data: dict, report_folder) -> tuple[pd.DataFrame, dict]:
        scores = []
        reports = {}
        for model_name, df_fake in fake_data.items():
            report_path = f"{report_folder}/{model_name}.rpt"
            diagnostic_path = f"{report_folder}/{model_name}_diagnostic.rpt"
            report = QualityReport()
            diagnostic = DiagnosticReport()
            if os.path.exists(report_path):
                report = report.load(report_path)
            else:
                columns_to_compare = list(set(self.train_data.columns) & set(self.includes) & set(df_fake.columns))
                metadata = self.metadata.to_dict()
                metadata["columns"] = {k:v for k,v in metadata["columns"].items() if k in columns_to_compare}
                report.generate(self.real_data.loc[:,columns_to_compare], df_fake.loc[:,columns_to_compare], metadata)
                report.save(report_path)
            
            if os.path.exists(diagnostic_path):
                diagnostic = diagnostic.load(diagnostic_path)
            else:
                columns_to_compare = list(set(self.train_data.columns) & set(self.includes) & set(df_fake.columns))
                metadata = self.metadata.to_dict()
                metadata["columns"] = {k:v for k,v in metadata["columns"].items() if k in columns_to_compare}
                diagnostic.generate(self.real_data.loc[:,columns_to_compare], df_fake.loc[:,columns_to_compare], metadata)
                diagnostic.save(diagnostic_path)

            scores.append( dict(
                name = model_name,
                type = "avg",
                score = report.get_score()
            ))
            scores.append( dict(
                name = model_name,
                type = "Column Shapes",
                score = report._property_breakdown["Column Shapes"]
            ))
            scores.append( dict(
                name = model_name,
                type = "Column Pair Trends",
                score = report._property_breakdown["Column Pair Trends"]
            ))
            for k,v in diagnostic.get_properties().items():
                scores.append( dict(
                    name = model_name,
                    type = k,
                    score = v
                ))                
            reports[model_name] = report
        pd_scores = pd.DataFrame(scores).set_index("name")
        privacy_frame, _ = self.calculate_privacy(fake_data, report_folder)
        return pd_scores.join(privacy_frame), reports
    
    def is_categorical(self, serie: pd.Series) -> bool:
        return self.metadata.columns[serie.name]["sdtype"] == "categorical"
    def is_datetime(self, serie: pd.Series) -> bool:
        return self.metadata.columns[serie.name]["sdtype"] == "datetime"

    def get_probs_from_serie(self, serie: pd.Series) -> pd.DataFrame:
        values, counts = np.unique(serie, return_counts=True)
        total = np.sum(counts)
        df = pd.DataFrame(dict(label=values, count=counts)).set_index("label").sort_values("count", ascending=False)
        df.loc[:, "prob"] = df["count"] / total
        return df


    def get_metrics_from_serie(self, serie: pd.Series) -> dict:
        probs = self.get_probs_from_serie(serie)
        if self.is_categorical(serie) or self.is_datetime(serie):
            try:
                desc = describe(serie.astype('category'), stats=["nobs", "missing"])[serie.name] 
            except KeyError:
                desc = {
                    "nobs": serie.shape[0],
                    "missing": serie.isna().count()
                }
        else:
            print(serie.name)
            desc = describe(serie, 
            stats=["nobs", "missing", "mean", "std_err", "ci", "ci", "std", "iqr", "iqr_normal", "mad", "mad_normal", "coef_var", "range", "max", "min", "skew", "kurtosis", "jarque_bera", "mode", "freq", "median", "percentiles", "distinct"],
            percentiles=[0.1,1,5,25,75,95,99,99.9]
            )[serie.name] 
        

        metrics = dict(
            name = serie.name,
            top5 = probs.head(5).index.values,
            top5_freq = probs.head(5)["count"].values,
            top5_prob = probs.head(5).prob.values,
            is_categorical = self.is_categorical(serie)
        )
        return dict(**metrics, **desc)


    def get_metrics(self, df : pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([
            self.get_metrics_from_serie(df[column]) for column in tuple(set(self.includes) & set(df.columns))
        ])