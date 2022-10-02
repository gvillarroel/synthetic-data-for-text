import json
from statsmodels.stats.descriptivestats import describe
from scipy import stats
import pandas as pd
import numpy as np


def topn(df, serie, n):
  v, c = np.unique(serie.astype(str), return_counts=True)
  t = np.sum(c)
  df = pd.DataFrame(dict(labels=v, count=c)).set_index("labels").sort_values("count", ascending=False).head(n)
  df.loc[:, "prob"] = df["count"] / t
  return df


def get_top_stats(df, columns):
  return pd.concat(
      [
          pd.DataFrame({col: [tuple(topn(df, df[col], 5).index.values)] for col in columns}).rename(index={0:"top5"}),
          pd.DataFrame({col: [tuple(topn(df, df[col], 5)["count"].values)] for col in columns}).rename(index={0:"top5_freq"}),
          pd.DataFrame({col: [tuple(topn(df, df[col], 5)["prob"].values)] for col in columns}).rename(index={0:"top5_prob"})
      ]
  )


def get_cat_stats(df, columns):
  descripcion = describe(
      df.loc[:,columns].astype("category"),
      stats=["nobs", "missing"]      
  )
  descripcion = pd.concat([descripcion, get_top_stats(df, columns)], axis=0)
  return descripcion

def get_all_stats(df, columns):
  descripcion = describe(
      df.loc[:,columns],
      ntop=3,
      stats=["nobs", "missing", "mean", "std_err", "ci", "ci", "std", "iqr", "iqr_normal", "mad", "mad_normal", "coef_var", "range", "max", "min", "skew", "kurtosis", "jarque_bera", "mode", "freq", "median", "percentiles", "distinct", "top"],
      percentiles=[0.1,1,5,25,75,95,99,99.9]
  )
  descripcion.loc["mean_5_95",:] = [stats.tmean(df[col], (descripcion.loc["5.0%", col], descripcion.loc["95.0%", col], )) for col in descripcion.columns] 
  descripcion.loc["tstd_5_95",:] = [stats.tstd(df[col], (descripcion.loc["5.0%", col], descripcion.loc["95.0%", col], )) for col in descripcion.columns]

  if "top5" in descripcion.index:
    descripcion.drop(index=["top5"],inplace=True)
  #descripcion = descripcion.append( pd.DataFrame({col: [tuple(topn(df[col], 5).index.values)] for col in descripcion.columns}).rename(index={0:"top5"}) )

  if "top5_freq" in descripcion.index:
    descripcion.drop(index=["top5_freq"],inplace=True)
  #descripcion = descripcion.append( pd.DataFrame({col: [tuple(topn(df[col], 5)["count"].values)] for col in descripcion.columns}).rename(index={0:"top5_freq"}) )

  if "top5_prob" in descripcion.index:
    descripcion.drop(index=["top5_prob"],inplace=True)
  #descripcion = descripcion.append( pd.DataFrame({col: [tuple(topn(df[col], 5)["prob"].values)] for col in descripcion.columns}).rename(index={0:"top5_prob"}) )
  descripcion = pd.concat([descripcion, get_top_stats(df, columns)], axis=0)


  return descripcion


def stats_csv_cat_con(datapath, metadatapath):
    with open(metadatapath) as fmetadata:
        df = pd.read_csv(datapath)
        metadata = json.load(fmetadata)
        COLUMNAS_CONTINUAS = tuple(metadata["COLUMNAS_CONTINUAS"])
        COLUMNAS_CATEGORICAS = tuple(metadata["COLUMNAS_CATEGORICAS"])
        return get_all_stats(df, COLUMNAS_CONTINUAS), get_cat_stats(df, COLUMNAS_CATEGORICAS)

def stats_csv(datapath, metadatapath):
    return pd.concat(list(stats_csv_cat_con(datapath, metadatapath)), axis=1)
