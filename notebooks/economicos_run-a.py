import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
from syntheticml.models.tab_ddpm.sdv import SDV_MLP
import torch
import numpy as np
import itertools
import multiprocessing as mp
from syntheticml.data.synthetic import MODELS
from pylatexenc.latexencode import unicode_to_latex

import os
import sys
DATASET_VERSION=sys.argv[1]
DATASET_NAME = "Economicos"

if __name__ == '__main__':
    df = pd.read_parquet('../datasets/economicos/raw/full_dedup_economicos_step0.parquet')

    #category_columns=("property_type", "transaction_type", "state", "county", "rooms", "bathrooms", "m_built", "m_size", "source", )
    category_columns=("property_type", "transaction_type", "state", "county", "rooms", "bathrooms", "source", )
    df_converted = df.dropna().astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
    print(df_converted.shape)
    basedate = pd.Timestamp('2017-12-01')
    dtime = df_converted.pop("publication_date")
    df_converted["publication_date"] = dtime.apply(lambda x: (x - basedate).days)
    syn = Synthetic(df_converted, 
            id="url", 
            category_columns=category_columns,
            text_columns=("description", "price", "title", "address", "owner", "source", "url", ),
            exclude_columns=tuple(),
            synthetic_folder = f"../datasets/economicos/synth-{DATASET_VERSION}",
            models=["copulagan", "tvae", "gaussiancopula", "ctgan", "smote-enc", 'tddpm_mlp'],
            n_sample = df_converted.shape[0],
            target_column="_price",
            max_cpu_pool=1,
            model_parameters=dict(
                tddpm_mlp=dict(
                        batch_size=3750,
                        steps=3000000,
                        num_timesteps=100,
                        lr=5e-4,
                        model_params=dict(
                                rtdl_params=dict(
                                        dropout=0.0,
                                        d_layers=[1024, 512, 256]
                                )
                        )
                )
            )
    )

    syn.process()
    syn.process_scores()
    print(syn._selectable_columns())
    print(syn.train.loc[:, syn._selectable_columns()])
    
    print(syn.current_metrics())

import numpy as np
def is_same(x):
    """
    subset=coverage_score.columns[2:],
        props='bfseries:;',
        axis=1
    """
    values = [i if type(i) == str else '{:.2e}' if i < 1.0 else round(i,5) for i in x.values]
    return [ 'cellcolor:[rgb]{0.9, 0.54, 0.52};' if i == 0 or values[0] == v else '' for i,v in enumerate(values)]

def print_charts(folder_path, model_name, figs):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    model_tex = open(f"{folder_path}/../{model_name}.tex", "w")
    relative_path = folder_path.replace("../docs/tesis/", "")
    for fig in figs:
        if fig:
            file_name = f'{fig.layout.title.text.replace(":","").replace(" ","_").lower()}'
            field_name = ' '.join(map(str.capitalize, file_name.split('_')))
            fig.update_layout(
                title=f"Distribución Variable {field_name}",
                xaxis_title=f"Total <{field_name}>",
                yaxis_title="Frequency",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.write_image(f"{folder_path}/{file_name}.svg")
            field_name_ = field_name.replace("_", "\_")
            model_name_ = model_name.replace("_", "-").split("-")[0]
            with open(f"{folder_path}/{file_name}.tex", "w") as ltext:
                ltext.write(f"""\\begin{{figure}}[H]
    \\centering
    \\includesvg[scale=.7,inkscapelatex=false]{{{relative_path}/{file_name}.svg}}
    \\caption{{Frecuencia del campo {field_name_.capitalize()} en el modelo real y {model_name_.lower() if model_name_ != "top2" else "Top 2"}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})}}
    \\label{{frecuency-{field_name}-{model_name}}}
\\end{{figure}}""")
            print(f"{folder_path}/{file_name}.svg")
            model_tex.write(f'\input{{{relative_path}/{file_name}.tex}}\n')
    model_tex.close()

def max_min_first(x):
    i, *v = x.values
    if isinstance(i, str) or isinstance(v[0], str) \
          or isinstance(i, list) or isinstance(v[0], list) \
            or isinstance(i, np.ndarray) or isinstance(v[0], np.ndarray):
        return ['']*len(x)
    
    mm = np.abs(np.array(v)-i)
    vmin = np.min(mm)
    vmax = np.max(mm)
    if vmin == vmax:
        return ['']*len(x)
    else:
        return [''] + ['bfseries:;' if vv==vmin else 'cellcolor:[rgb]{0.9, 0.54, 0.52};' if vv==vmax else '' for vv in mm]


if __name__ == '__main__':
    best_model = "tddpm_mlp"
    second_best_model = "smote-enc"

    # Frecuency Maps
    from syntheticml.data.charts import Charts
    base_path = f"../docs/tesis/datasets/economicos-{DATASET_VERSION}"
    print_charts(
        f"{base_path}/{best_model}", 
        best_model,
        syn.get_charts(
            best_model, 
            set(syn.text_columns) | set(syn.exclude_columns)
            )
    )
    
    print_charts(
        f"{base_path}/{second_best_model}", 
        second_best_model,
        syn.get_charts(
            second_best_model, 
            set(syn.text_columns) | set(syn.exclude_columns)
            )
    )

    print_charts(
        f"{base_path}/top2", 
        f"top2",
        syn.get_multiple_charts(
            [best_model, second_best_model], 
            set(syn.text_columns) | set(syn.exclude_columns)
            )
    )


    print_charts(
        f"{base_path}/top2+1", 
        f"top2+1",
        syn.get_multiple_charts(
            [best_model, second_best_model, "copulagan"], 
            set(syn.text_columns) | set(syn.exclude_columns)
            )
    )

    # Correlation Maps
    models = [best_model, second_best_model]
    prop_cat = ["name", "top5", "top5_prob"]

    current_metrics = syn.current_metrics()
    fake_metrics = syn.get_metrics_fake()
    columns = list(current_metrics.name.unique())
    relative_path = base_path.replace("../docs/tesis/","")
    dfs = [
        current_metrics.dropna(axis=1, how='all').assign(model="Real")
    ]
    for model_name in models + ["ctgan"]:
        dfs.append(fake_metrics[model_name]\
                   .dropna(axis=1, how='all')\
                    .assign(model=model_name))

    diffdf = pd.concat(dfs)
    stats_tex = open(f"{base_path}/stats.tex", "w")
    for col in columns:
        d = diffdf[ (diffdf["name"] == col) ].copy()
        d = d.drop(columns=["name", "is_categorical"])\
            .rename(columns={"model":"Variable/Modelo"})\
                .set_index("Variable/Modelo").T.dropna()
        f_d = d.style\
            .format(precision=3, escape="latex")\
            .format(precision=0, subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 1000],:] )\
            .format("{:.5e}", subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 10e8],:] )\
            .format_index(escape="latex", precision=3, axis=1)\
            .format_index("\hline {}", escape="latex", precision=3, axis=0)\
            .set_table_styles([
            {'selector': 'toprule', 'props': ':hline\n \\rowcolor[gray]{0.8};'},
            {'selector': 'bottomrule', 'props': ':hline;'}
        ], overwrite=False)\
        .apply(
            max_min_first,
            axis=1
        ).to_latex(
            column_format = "|l|m{10em}|m{10em}|m{10em}|m{10em}|",
            position="H",
            position_float="centering",
            caption = unicode_to_latex(f"Propiedades  estadisticas de variable {col}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
            label = f"table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}",
            clines=None,
        ).replace("\centering", "\\centering\n\\fontsize{8}{14}\\selectfont")
        with open(f"{base_path}/tables/table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}.tex", "w") as stext:
            stext.write(f_d)
        stats_tex.write(f'\input{{{relative_path}/tables/table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}.tex}}\n')
    stats_tex.close()

    stats_tex = open(f"{base_path}/stats-short.tex", "w")
    for col in columns:
        d = diffdf[ (diffdf["name"] == col) ].copy()
        d = d.drop(columns=["name", "is_categorical"])\
            .rename(columns={"model":"Variable/Modelo"})\
                .set_index("Variable/Modelo").T.dropna()
        change = np.array([pd.Series(d.loc[[i], ["Real", "tddpm_mlp", "smote-enc"]].values.reshape(-1)).pct_change().abs().sum() if not i.startswith("top5") else 0.0 for i in d.index  ])
        d = d.loc[change > 0.005,:]
        f_d = d.style\
            .format(precision=3, escape="latex")\
            .format(precision=0, subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 1000],:] )\
            .format("{:.5e}", subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 10e8],:] )\
            .format_index(escape="latex", precision=3, axis=1)\
            .format_index("\hline {}", escape="latex", precision=3, axis=0)\
            .set_table_styles([
            {'selector': 'toprule', 'props': ':hline\n \\rowcolor[gray]{0.8};'},
            {'selector': 'bottomrule', 'props': ':hline;'}
        ], overwrite=False)\
        .apply(
            max_min_first,
            axis=1
        ).to_latex(
            column_format = "|l|m{10em}|m{10em}|m{10em}|m{10em}|",
            position="H",
            position_float="centering",
            caption = unicode_to_latex(f"Propiedades estadisticas de variable {col} con cambio>5%, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
            label = f"table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}-short",
            clines=None,
        ).replace("\centering", "\\centering\n\\fontsize{8}{14}\\selectfont")
        with open(f"{base_path}/tables/table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}-short.tex", "w") as stext:
            stext.write(f_d)
        stats_tex.write(f'\input{{{relative_path}/tables/table-stats-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{col}-short.tex}}\n')
    stats_tex.close()

    if not os.path.exists(f"{base_path}/pairwise/"):
        os.makedirs(f"{base_path}/pairwise/", exist_ok=True)
    pair_tex = open(f"{base_path}/pairwise.tex", "w")
    relative_path = base_path.replace("../docs/tesis/", "")
    for model_name, model_data in syn.fake_data.items():
        model_name_ = model_name.replace("_", "-").split("-")[0].capitalize()
        fig = syn.charts.pair_corr(syn.df, model_data, set(syn.text_columns) | set(syn.exclude_columns), syn.target_column)
        #fig.update_layout(dict(width=1000)).show("png")
        fig.update_layout(
            title=dict(
                text=f"Comparativa entre original y {model_name_}",
                x=0.5,  # x=0.5 centra el título horizontalmente
                yanchor="top",  # Alinea el título en la parte superior
                font=dict(
                    size=20,  # Establecer el tamaño de la fuente
                    family="Arial, sans-serif",  # Establecer la fuente
                )
            )
        )
        fig.write_image(f"{base_path}/pairwise/pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}.svg")
        ecaped_model = model_name.replace("_", "\_")
        with open(f"{base_path}/pairwise/{model_name}.tex", "w") as ltext:
            ltext.write(f"""\\begin{{figure}}[H]
    \\centering
    \\includesvg[scale=.6,inkscapelatex=false]{{{relative_path}/pairwise/pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}.svg}}
    \\caption{{Correlación de conjunto original de entrenamiento y {model_name_}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})}}
    \\label{{pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}}}
\\end{{figure}}""")
            pair_tex.write(f'\input{{{relative_path}/pairwise/{model_name}.tex}}\n')                
        print(f"{base_path}/pairwise/pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}.svg")
        if "description" in model_data.columns:
            c_3 = pd.DataFrame(index=range(5), data={"description": model_data.sample(5)["description"].to_list() })
            #c_3["description"] = model_data.sample(10)["description"].apply(unicode_to_latex) 
            #.format(escape="latex")\
            current_sample_wtext = c_3.style\
            .hide(axis="index")\
            .format("\hline {}", c_3.columns[0], escape="latex")\
            .format_index("\hline {}", escape="latex", axis=0)\
            .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
            {'selector': 'bottomrule', 'props': ':hline;'}
        ], overwrite=False)\
                .to_latex(
                    column_format = "|m{50em}|",
                    position="H",
                    position_float="centering",
                    caption = unicode_to_latex(f"Ejemplos de textos aleatoreos del modelo {model_name_}, conjunto {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
                    label = f"table-sample10-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}-text",
                    clines=None
                ).replace("\centering", "\\centering\n\\fontsize{8}{14}\\selectfont")
            with open(f"{base_path}/tables/table-sample10-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}-text.tex", "w") as stext:
                stext.write(current_sample_wtext) 
    
    # Score Table
    score_table = syn.scores.sort_values("score", ascending=False).loc[:
    ,["type", "score"]].reset_index().pivot(index="name", 
                                            columns=["type"],values="score").sort_values(
    "avg", ascending=False).rename(columns={'avg':'Score'}).loc[:,
    ["Column Pair Trends", "Column Shapes", "Coverage", "Boundaries", "Score"]].reset_index().rename(columns={"name": "Model Name"}).rename(columns={"Score":"\\textbf{Score}"})

    formated_table = score_table.style.hide(axis="index")\
        .format(precision=3)\
        .format("\hline {}", score_table.columns[0], escape="latex")\
        .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n \\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False).highlight_max(
        subset=score_table.columns[1:],
        props='bfseries:;'
    ).to_latex(
        column_format = f"|l|{'r|'*len(score_table.columns[1:])}",
        position="H",
        position_float="centering",
        caption = f"SDMetric Scores {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})",
        label = f"table-score-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    if not os.path.exists(f"{base_path}/tables"):
        os.makedirs(f"{base_path}/tables", exist_ok=True)
    with open(f"{base_path}/tables/table-score-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(formated_table)

    detailed_scores = syn.get_details()
    coverage_score = pd.concat(
    [
        detailed_scores[model_name]['diagnostic']['coverage'].assign(model=model_name) 
        for model_name in models
    ]
    ).pivot(index=["Column","Metric"], 
            values="Diagnostic Score", columns="model").sort_values("smote-enc", ascending=False).reset_index().rename(columns={"Column": "Columna", "Metric":"Metrica"})
    formated_coverage = coverage_score.sort_values("Columna").style.hide(axis="index")\
        .format(precision=3)\
        .format("\hline {}", coverage_score.columns[0:1], escape="latex")\
        .format_index("{}", escape="latex", axis=1)\
        .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False).highlight_max(
        subset=coverage_score.columns[2:],
        props='bfseries:;',
        axis=1
    ).to_latex(
        column_format = f"|l|l|{'r|'*len(coverage_score.columns[2:])}",
        position="H",
        position_float="centering",
        caption = f"Cobertura {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})",
        label = f"table-coverage-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-coverage-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(formated_coverage)

    shape_score = pd.concat(
    [
        detailed_scores[model_name]['report']['column_shape'].assign(model=model_name) 
        for model_name in models
    ]
    ).pivot(index=["Column","Metric"], 
            values="Quality Score", columns="model").sort_values("smote-enc", ascending=False).reset_index().rename(columns={"Column": "Columna", "Metric":"Metrica"})
    formated_shape = shape_score.sort_values("Columna").style.hide(axis="index")\
        .format(precision=3)\
        .format("\hline {}", shape_score.columns[0:1], escape="latex")\
        .format_index("{}", escape="latex", axis=1)\
        .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False).highlight_max(
        subset=shape_score.columns[2:],
        props='bfseries:;',
        axis=1
    ).to_latex(
        column_format = f"|l|l|{'r|'*len(shape_score.columns[2:])}",
        position="H",
        position_float="centering",
        caption = f"Distribución {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})",
        label = f"table-shape-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-shape-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(formated_shape)

    dcr_score = syn.scores[syn.scores["type"] == "avg"].sort_values("score", ascending=False).loc[:,["DCR ST 5th", "DCR SH 5th", "DCR TH 5th","NNDR ST 5th", "NNDR SH 5th", "NNDR TH 5th", "score"]].reset_index().rename(columns={'name':"Modelo", "score": "\\textbf{Score}", "DCR ST 5th":"DCR ST", "DCR SH 5th": "DCR SH", "DCR TH 5th": "DCR TH", "NNDR ST 5th": "NNDR ST", "NNDR SH 5th": "NNDR SH", "NNDR TH 5th": "NNDR TH"})
    formated_dcr = dcr_score.style.hide(axis="index")\
        .format(precision=3)\
        .format("\hline {}", dcr_score.columns[0], escape="latex")\
        .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False).highlight_min(
        subset=dcr_score.columns[1:3],
        props='cellcolor:[rgb]{0.9, 0.54, 0.52};',
        axis=0
    ).highlight_max(
        subset=dcr_score.columns[1:],
        props='bfseries:;',
        axis=0
    ).to_latex(
        column_format = f"|l|l|{'r|'*len(dcr_score.columns[1:])}",
        position="H",
        position_float="centering",
        caption = f"Distancia de registros más cercanos entre conjuntos Sinteticos, \emph{{Train}} y \emph{{Hold}}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()}",
        label = f"table-dcr-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-dcr-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(formated_dcr)
    
    avg = syn.scores[syn.scores["type"] == "avg"]
    avg.sort_values("score", ascending=False).loc[:,:]

    dfs = []
    for model in avg.index:
        model_ = model.replace("_", "-").split("-")[0].capitalize()
        for i in ["min", "1p", "2p", "3p", "4p", "5p"]:
            d = avg.loc[[model],[f"record_ST_{i}"]].iloc[0].to_dict()[f"record_ST_{i}"]
            current_data = pd.DataFrame.from_dict(
                    [d["record"], d["closest"], d["closest_2"]]
                ).assign(distance=['Sintético']+[f'DCR{i+1} d({v:.2e})' for i,v in enumerate(d["dists"])]).rename(columns={"distance": "Variable/Distancia"})
            dfs.append(
                current_data.assign(model=model).assign(level=i)
            )
            c_1 = current_data.drop(columns=[c for c in current_data.columns if c in syn.text_columns or c == "input_text"]).copy()
            c_1 = c_1.set_index(["Variable/Distancia"]).T.sort_index()
            percentil = "minimo" if i == "min" else f"percentil {i[0]}"
            #.format(escape="latex", subset=[c for c in c_1.columns if c in syn.text_columns])\
            current_table = c_1.style\
            .format_index(escape="latex", precision=3, axis=1)\
            .format_index("\hline {}", escape="latex", precision=3, axis=0)\
            .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False)\
            .apply(
                is_same,
                axis=1
            )\
            .to_latex(
                column_format = f"|l|r|r|r|",
                position="H",
                position_float="centering",
                caption = unicode_to_latex(f"Ejemplos para el modelo {model_}, {percentil}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
                label = f"table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}",
                clines=None
            ).replace("\centering", "\\centering\n\\fontsize{10}{14}\\selectfont")
            with open(f"{base_path}/tables/table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}.tex", "w") as stext:
                stext.write(current_table)

            c_2 = current_data[["Variable/Distancia","description"]].copy()
            c_2 = c_2.rename(columns={"Variable/Distancia": "Distancia"})
            c_2["description"] = c_2["description"].apply(unicode_to_latex) 
            #.format(escape="latex")\
            current_table_wtext = c_2.style\
            .hide(axis="index")\
            .format("\hline {}", c_2.columns[0], escape="latex")\
            .format_index("\hline {}", escape="latex", axis=0)\
            .set_table_styles([
        {'selector': 'toprule', 'props': ':hline\n\\rowcolor[gray]{0.8};'},
        {'selector': 'bottomrule', 'props': ':hline;'}
    ], overwrite=False)\
            .to_latex(
                column_format = "|l|m{35em}|",
                position="H",
                position_float="centering",
                caption = unicode_to_latex(f"Ejemplos de texto modelo {model_}, {percentil}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
                label = f"table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}-text",
                clines=None
            ).replace("\centering", "\\centering\n\\fontsize{10}{14}\\selectfont")
            with open(f"{base_path}/tables/table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}-text.tex", "w") as stext:
                stext.write(current_table_wtext)
    
    data_table = pd.concat(dfs).set_index(["model", "level", "Variable/Distancia"])
    latex_table = data_table.style\
        .format_index("{}", escape="latex", axis=1)\
        .format_index("{}", escape="latex", axis=0)\
        .to_latex(
        position="H",
        position_float="centering",
        caption = f"Distancia de registros más cercanos entre conjuntos Sinteticos, \emph{{Train}} y \emph{{Hold}}, {DATASET_NAME.capitalize()}",
        label = f"table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(latex_table)