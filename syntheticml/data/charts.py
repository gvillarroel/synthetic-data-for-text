import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px


class Charts:
    def __init__(self, metadata: dict, color_real :str='rgba(12,10,255,0.5)', color_synthetic: str='rgba(255,0,0,0.5)') -> None:
        self.metadata = metadata
        self.color_real = color_real
        self.color_synthetic = color_synthetic

    def is_categorical(self, serie: pd.Series) -> bool:
        return self.metadata["fields"][serie.name]["type"] == "categorical"

    def is_id(self, serie: pd.Series) -> bool:
        return self.metadata["fields"][serie.name]["type"] == "id"
    
    def get_serie_title(self, serie) -> str:
        return serie.name

    def chart_categorical(self, serie_real : pd.Series, serie_fake : pd.Series) -> go.Figure:
        x, y = np.unique( serie_real, return_counts=True)
        x2, y2 = np.unique( serie_fake, return_counts=True)
        fig = go.Figure(data=[
            go.Bar(name='Real', x=x, y=y, marker_color=self.color_real),
            go.Bar(name='Synthetic', x=x2, y=y2, marker_color=self.color_synthetic)
        ], layout=dict(title=self.get_serie_title(serie_real)))
        return fig

    def chart_continues(self, serie_real : pd.Series, serie_fake : pd.Series) -> go.Figure:
        tmin = serie_real.quantile(0.001)
        tmax = serie_real.quantile(0.95)
        fig = go.Figure(data=[
            go.Histogram(x=serie_real[((serie_real <= tmax) & (serie_real >= tmin) )], name="Real", marker_color=self.color_real),
            go.Histogram(x=serie_fake[((serie_fake <= tmax) & (serie_real >= tmin) )], name="Synthetic", marker_color=self.color_synthetic),
        ], layout=dict(title=self.get_serie_title(serie_real), barmode='overlay'))

        return fig
    

    def chart(self, serie_real: pd.Series, serie_fake: pd.Series) -> go.Figure:
        if self.is_categorical(serie_real):
            return self.chart_categorical(serie_real, serie_fake)
        elif self.is_id(serie_real):
            return None
        else:
            return self.chart_continues(serie_real, serie_fake)
    
    def charts(self, df_real : pd.DataFrame, df_fake: pd.DataFrame, exclude_columns: set[str]={}) -> list[go.Figure]:
        columns = tuple((set(df_real.columns) & set(df_fake.columns)) - exclude_columns)
        return [self.chart(df_real[column], df_fake[column]) for column in columns]
    
    def pair_corr(self, df_real : pd.DataFrame, df_fake: pd.DataFrame, exclude_columns: set[str]=set()) -> list[go.Figure]:
        columns = list((set(df_real.columns) & set(df_fake.columns)) - exclude_columns)
        fig = make_subplots(rows=1, cols=2, column_titles=['Real', 'Synthetic'], horizontal_spacing= 0.09, vertical_spacing=0)
        ix = df_real[columns].corr(numeric_only=True).abs().sort_values('price', ascending=True).index

        fig.add_trace(
            px.imshow(df_real.loc[:, ix].corr(numeric_only=True).abs()).data[0],
            row=1, col=1
        )
        fig.add_trace(
            px.imshow(df_fake.loc[:, ix].corr(numeric_only=True).abs()).data[0],
            row=1, col=2
        )
        return fig
