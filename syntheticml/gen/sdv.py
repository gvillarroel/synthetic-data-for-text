from sdv.tabular import GaussianCopula
from sdv.metadata import table
import pandas as pd
import os
import json


def gen_sdv_metadata_from_csv(datapath, column_id, columns_to_mimic, categorical_columns, svd_metadatapath):
    df = pd.read_csv(datapath)    
    COLUMNAS_ID             = column_id
    COLUMNAS_CATEGORICAS    = categorical_columns
    COLUMNAS_A_MIMIC        = columns_to_mimic
    meta = table.Table(primary_key=COLUMNAS_ID, field_names=COLUMNAS_A_MIMIC | {COLUMNAS_ID,}, field_transformers={k:'LabelEncoder' for k in COLUMNAS_CATEGORICAS})
    meta.fit(df.astype({ k:"category" for k in COLUMNAS_CATEGORICAS }))
    meta.to_json(svd_metadatapath)


def gen_sdv_from_csv(datapath, metadatapath, checkpoint):
    with open(metadatapath) as fmetadata:
        metadata = json.load(fmetadata)
        df = pd.read_csv(datapath)    
        model = GaussianCopula(table_metadata=metadata)
        if os.path.exists(checkpoint):
            model.load(checkpoint)
        else:
            model.fit(df)
            model.save(checkpoint)
        
        return model


def gen_sdv_to_parquet(model, n, outputpath):
    if os.path.exists(outputpath):
        pass
    else:
        model.sample(n).to_parquet(outputpath)

from sdmetrics import load_demo
from sdmetrics.reports.single_table import QualityReport

def compare_sdv(df_real, df_fake, metadatapath):
    with open(metadatapath) as fmetadata:
        metadata = json.load(fmetadata)
        report = QualityReport()
        report.generate(df_real.loc[:,df_fake.columns], df_fake, metadata)
        return report