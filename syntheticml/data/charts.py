from turtle import width
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import math



class Charts:
    def __init__(self, metadata: dict, color_real :str='rgba(0,75,147,0.5)', color_synthetic: str='rgba(230,51,41, 0.5)') -> None:
        self.metadata = metadata
        self.color_real = color_real
        self.color_synthetic = color_synthetic
        self.colors = [color_synthetic, "rgba(254,207,68,0.5)", "rgba(0,126,72, 0.5)"]

    def is_categorical(self, serie: pd.Series) -> bool:
        return self.metadata.columns[serie.name]["sdtype"] == "categorical"

    def is_id(self, serie: pd.Series) -> bool:
        return self.metadata.primary_key == serie.name
    
    def get_serie_title(self, serie) -> str:
        return f"{serie.name}"

    def chart_categorical(self, serie_real : pd.Series, serie_fake : dict[str, pd.Series], max_categories=10) -> go.Figure:
        x, y = zip(*list(serie_real.astype(str).value_counts().to_frame().head(max_categories).to_dict()[serie_real.name].items()))
        y = np.array(list(y)) / sum(y)
        data=[
            go.Bar(name='Real', x=x, y=y, marker_color=self.color_real)
        ]

        next_colors = (
            self.colors[i%len(self.colors)]
            for i in range(20)
        )

        for key, df_fake in serie_fake.items():
            x2, y2 = np.unique( df_fake[df_fake.astype(str).isin(x)].astype(str), return_counts=True)
            y2 = np.array(list(y2)) / sum(y2)
            data.append(go.Bar(name=key, x=x2, y=y2, marker_color=next(next_colors)))
                                
        fig = go.Figure(data=data, layout=dict(title=self.get_serie_title(serie_real)))
        return fig

    def chart_continues(self, serie_real : pd.Series, serie_fake : dict[str, pd.Series]) -> go.Figure:
        tmin = serie_real.quantile(0.001)
        tmax = serie_real.quantile(0.95)
        
        data = [
            go.Histogram(x=serie_real[((serie_real <= tmax) & (serie_real >= tmin) )], name="Real", marker_color=self.color_real, histnorm='percent')
        ]
        
        next_colors = (
            self.colors[i%len(self.colors)]
            for i in range(20)
        )

        for key, df_fake in serie_fake.items():
            data.append(
                go.Histogram(x=df_fake[((df_fake <= tmax) & (df_fake >= tmin) )], name=key, marker_color=next(next_colors), histnorm='percent')
            )
                
        fig = go.Figure(data=data, layout=dict(title=self.get_serie_title(serie_real), barmode='overlay'))

        return fig
    

    def chart(self, serie_real: pd.Series, serie_fake: dict[str, pd.Series], categorical_min=0) -> go.Figure:
        if self.is_categorical(serie_real):
            return self.chart_categorical(serie_real, serie_fake)
        elif self.is_id(serie_real):
            return None
        else:
            return self.chart_continues(serie_real, serie_fake)
    
    def charts(self, df_real : pd.DataFrame, df_fake: dict[str, pd.DataFrame], exclude_columns: set[str]={}) -> list[go.Figure]:
        columns = tuple((set(df_real.columns) & set(df_fake[list(df_fake.keys())[0]].columns)) - exclude_columns)
        return [self.chart(df_real[column], { k: df_fake[k][column] for k in df_fake.keys()} ) for column in columns]
    
    def pair_corr(self, df_real : pd.DataFrame, df_fake: pd.DataFrame, exclude_columns: set[str]=set(), sort_column: str=None) -> list[go.Figure]:
        columns = list((set(df_real.columns) & set(df_fake.columns)) - exclude_columns)
        fig = make_subplots(rows=1, cols=2, column_titles=['Real', 'Synthetic'], horizontal_spacing= 0.16, vertical_spacing=0, column_widths=[1000, 1000])
        if sort_column:
            ix = df_real[columns].corr(numeric_only=True).abs().sort_values(sort_column, ascending=True).index
        else:
            ix = df_real[columns].corr(numeric_only=True).abs().index

        fig.add_trace(
            px.imshow(df_real.loc[:, ix].corr(numeric_only=True).abs()).data[0],
            row=1, col=1
        )
        fig.add_trace(
            px.imshow(df_fake.loc[:, ix].corr(numeric_only=True).abs()).data[0],
            row=1, col=2
        )
        fig.update_layout(dict(width=1000))
        return fig

    def all_pair_corr(self, df_real: pd.DataFrame, df_fakes: dict[str, pd.DataFrame], exclude_columns: set[str], sort_column: str=None, max_col: int=4) -> list[go.Figure]:
        columns = list((set(df_real.columns) & set(df_fake.columns)) - exclude_columns)
        n_c = len(df_fakes) + 1
        fig = make_subplots(rows=math.floor(n_c//max_col)+1, cols=min(n_c, max_col), column_titles=['Real'] + list(df_fakes.keys()), horizontal_spacing= 0.16, vertical_spacing=0, column_widths=[1000, 1000])
        if sort_column:
            ix = df_real[columns].corr(numeric_only=True).abs().sort_values(sort_column, ascending=True).index
        else:
            ix = df_real[columns].corr(numeric_only=True).abs().index

        fig.add_trace(
            px.imshow(df_real.loc[:, ix].corr(numeric_only=True).abs()).data[0],
            row=1, col=1
        )
        for i, k in enumerate(df_fakes.keys()):
            fig.add_trace(
                px.imshow(df_fakes[k].loc[:, ix].corr(numeric_only=True).abs()).data[0],
                row=i//max_col, col=i%max_col
            )
        fig.update_layout(dict(width=1000))
        return fig

    def privacy(self, dist_real: np.array, dist_syn: np.array, model_name: str= "Synthetic"):
        fig = go.Figure(data=[
            go.Histogram(x=dist_real, name="Real", marker_color=self.color_real),
            go.Histogram(x=dist_syn, name=model_name, marker_color=self.color_synthetic),
        ], layout=dict(title="Privacy", barmode='overlay'))
        return fig
    
    def privacies(self, dist_real: np.array, dist_syn: dict[str,np.array]):
        data = [go.Histogram(x=dist_real, name="Real", marker_color=self.color_real)]
        
        next_colors = (
            self.colors[i%len(self.colors)]
            for i in range(20)
        )

        for k, v in dist_syn.items():
            data.append(
                go.Histogram(x=v, name=k, marker_color=next(next_colors))
            )
        
        fig = go.Figure(data=data, layout=dict(title="Privacy", barmode='overlay'))
        return fig