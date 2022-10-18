import pandas as pd
from sdmetrics.reports.single_table import QualityReport

import json
from statsmodels.stats.descriptivestats import describe
from scipy import stats
import pandas as pd
import numpy as np
import os

class Metrics:
    def __init__(self, real_data : pd.DataFrame, metadata: dict) -> None:
        self.real_data = real_data
        self.metadata = metadata
    
    def get_scores(self, fake_data: dict, report_folder) -> tuple[pd.DataFrame, dict]:
        scores = []
        reports = {}
        for k, df_fake in fake_data.items():
            report_path = f"{report_folder}/{k}.rpt"
            report = QualityReport()
            if os.path.exists(report_path):
                report = report.load(report_path)
            else:
                report.generate(self.real_data.loc[:,df_fake.columns], df_fake, self.metadata)
                report.save(report_path)
            scores.append( dict(
                name = k,
                type = "avg",
                score = report.get_score()
            ))
            scores.append( dict(
                name = k,
                type = "Column Shapes",
                score = report._property_breakdown["Column Shapes"]
            ))
            scores.append( dict(
                name = k,
                type = "Column Pair Trends",
                score = report._property_breakdown["Column Pair Trends"]
            ))
            reports[k] = report
        return pd.DataFrame(scores), reports
    
    def is_categorical(self, serie: pd.Series) -> bool:
        return self.metadata["fields"][serie.name]["type"] == "categorical"

    def get_probs_from_serie(self, serie: pd.Series) -> pd.DataFrame:
        values, counts = np.unique(serie, return_counts=True)
        total = np.sum(counts)
        df = pd.DataFrame(dict(label=values, count=counts)).set_index("label").sort_values("count", ascending=False)
        df.loc[:, "prob"] = df["count"] / total
        return df


    def get_metrics_from_serie(self, serie: pd.Series) -> dict:
        probs = self.get_probs_from_serie(serie)
        if self.is_categorical(serie):
            desc = describe(serie.astype('category'), stats=["nobs", "missing"])[serie.name] 
        else:
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
            self.get_metrics_from_serie(df[column]) for column in df.columns
        ])