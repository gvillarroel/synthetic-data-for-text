import pandas as pd
from syntheticml.data.synthetic import Synthetic, MODELS
import os
from pylatexenc.latexencode import unicode_to_latex
import collections.abc

import sys
syn = None
DATASET_VERSION=sys.argv[1]
DATASET_NAME = "King County"

if __name__ == '__main__':
    df = pd.read_csv('../datasets/kingcounty/raw/kc_house_data.csv')
    syn = Synthetic(df,
                    id="id",
                    category_columns=("condition", "floors", "grade", "view",
                                      "waterfront", "zipcode", "bathrooms", "bedrooms",),
                    synthetic_folder=f"../datasets/kingcounty/synth-{DATASET_VERSION}",
                    models=MODELS.keys(),
                    n_sample=21613,
                    max_cpu_pool=1,
                    target_column="price",
                    model_parameters=dict(
                        tddpm_mlp=dict(
                                batch_size=3750,
                                steps=1000000,
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

import numpy as np
def is_same(x):
    """
    subset=coverage_score.columns[2:],
        props='bfseries:;',
        axis=1
    """
    values = [i if type(i) == str else '{:.2e}' if i < 1.0 else round(i,5) for i in x.values]
    return [ 'cellcolor:[rgb]{0.9, 0.54, 0.52};' if i == 0 or values[0] == v else '' for i,v in enumerate(values)]

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



def print_charts(folder_path, model_name, figs):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    #table-score-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}
    model_tex = open(f"{folder_path}/../{model_name}.tex", "w")
    relative_path = folder_path.replace("../docs/tesis/", "")
    for fig in figs:
        if fig:
            file_name = f'{fig.layout.title.text.replace(":","").replace(" ","_").lower()}'
            field_name = ' '.join(map(str.capitalize, file_name.split('_')))
            fig.update_layout(
                title=f"Distribución Variable {field_name}",
                xaxis_title=f"Total <{field_name}>",
                yaxis_title="Frequency Percentage",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.write_image(f"{folder_path}/{file_name}.svg")
            field_name_ = field_name.replace("_", "\_")
            model_name_ = model_name.replace("_", "-").split("-")[0]
            with open(f"{folder_path}/frecuency-{model_name_.lower()}-{field_name_.lower()}.tex", "w") as ltext:
                ltext.write(f"""\\begin{{figure}}[H]
    \\centering
    \\includesvg[scale=.7,inkscapelatex=false]{{{relative_path}/{file_name}.svg}}
    \\caption{{Frecuencia del campo {field_name_.capitalize()} en el modelo real y {model_name_.lower() if model_name_ != "top2" else "Top 2"}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})}}
    \\label{{frecuency-{model_name_.lower()}-{field_name_.lower()}}}
\\end{{figure}}""")
            print(f"{folder_path}/{file_name}.svg")
            model_tex.write(f'\input{{{relative_path}/frecuency-{model_name_.lower()}-{field_name_.lower()}.tex}}\n')
    model_tex.close()
    

if __name__ == '__main__':
    best_model = "tddpm_mlp"
    second_best_model = "smote-enc"

    # Frecuency Maps
    from syntheticml.data.charts import Charts
    base_path = f"../docs/tesis/datasets/{DATASET_NAME.replace(' ','').lower()}-{DATASET_VERSION}"
    print_charts(
        f"{base_path}/{best_model}", 
        best_model,
        syn.get_charts(
            best_model, 
            {'date', 'id', 'zipcode', 'lat', 'long', 'yr_renovated'}
            )
    )

    print_charts(
        f"{base_path}/{second_best_model}", 
        second_best_model,
        syn.get_charts(
            second_best_model, 
            {'date', 'id', 'zipcode', 'lat', 'long', 'yr_renovated'}
            )
    )

    print_charts(
        f"{base_path}/top2", 
        f"top2",
        syn.get_multiple_charts(
            [best_model, second_best_model], 
            {'date', 'id', 'zipcode', 'lat', 'long', 'yr_renovated'}
            )
    )


    print_charts(
        f"{base_path}/top2+1", 
        f"top2+1",
        syn.get_multiple_charts(
            [best_model, second_best_model, "copulagan"], 
            {'date', 'id', 'zipcode', 'lat', 'long', 'yr_renovated'}
            )
    )

    # Correlation Maps
    
    if not os.path.exists(f"{base_path}/pairwise/"):
        os.makedirs(f"{base_path}/pairwise/", exist_ok=True)
    pair_tex = open(f"{base_path}/pairwise.tex", "w")
    relative_path = base_path.replace("../docs/tesis/","")
    for model_name, model_data in syn.fake_data.items():
        model_name_ = model_name.replace("_", "-").split("-")[0].capitalize()
        fig = syn.charts.pair_corr(syn.df, model_data, {'id', 'waterfront', 'yr_renovated'}, "price")
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
        fig.write_image(f"{base_path}/pairwise/{model_name}.svg")
        
        with open(f"{base_path}/pairwise/pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}.tex", "w") as ltext:
            ltext.write(f"""\\begin{{figure}}[H]
    \\centering
    \\includesvg[scale=.6,inkscapelatex=false]{{{relative_path}/pairwise/{model_name}.svg}}
    \\caption{{Correlación de conjunto original de entrenamiento y {model_name_}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})}}
    \\label{{pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}}}
\\end{{figure}}""")
            pair_tex.write(f'\input{{{relative_path}/pairwise/pairwise-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model_name}.tex}}\n')                
        print(f"{base_path}/pairwise/{model_name}.svg")
    
    # Tables
    models = [best_model, second_best_model]
    prop_cat = ["name", "top5", "top5_prob"]

    current_metrics = syn.current_metrics()
    fake_metrics = syn.get_metrics_fake()
    columns = list(current_metrics.name.unique())

    dfs = [
        current_metrics.dropna(axis=1, how='all').assign(model="Real")
    ]
    for model_name in models + ["ctgan"]:
        dfs.append(fake_metrics[model_name]\
                   .dropna(axis=1, how='all')\
                    .assign(model=model_name))

    diffdf = pd.concat(dfs)
    stats_tex = open(f"{base_path}/stats.tex", "w")
    #.format(precision=0, subset=[i for i in d.index if d.loc[i, "Real"] > 1000])\
    for col in columns:
        d = diffdf[ (diffdf["name"] == col) ].copy()
        d = d.drop(columns=["name", "is_categorical"])\
            .rename(columns={"model":"Variable/Modelo"})\
                .set_index("Variable/Modelo").T.dropna()
        f_d = d.style\
            .format(precision=3, escape="latex")\
            .format_index(escape="latex", precision=3, axis=1)\
            .format(precision=0, subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 1000],:] )\
            .format("{:.5e}", subset=pd.IndexSlice[[i for i in d.index if not isinstance(d.loc[i, "Real"], np.ndarray) and np.abs(d.loc[i, "Real"]) > 10e8],:] )\
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
    #dfs = [
    #    current_metrics.loc[(current_metrics.name.isin(columns) & current_metrics.is_categorical),prop_cat].dropna(axis=1, how='all').assign(model="Real")
    #]
    #for model_name in models:
    #    dfs.append(fake_metrics[model_name].loc[(fake_metrics[model_name].name.isin(columns) & fake_metrics[model_name].is_categorical),prop_cat].dropna(axis=1, how='all').assign(model=model_name))
    #for column in diffdf["columns"]
    # Score Table
    score_table = syn.scores.sort_values("score", ascending=False).loc[:
    ,["type", "score"]].reset_index().pivot(index="name", 
                                            columns=["type"],values="score").sort_values(
    "avg", ascending=False).rename(columns={'avg':'Score'}).loc[:,
    ["Column Pair Trends", "Column Shapes", "Coverage", "Boundaries", "Score"]].reset_index().rename(columns={"name": "Model Name"}).rename(columns={"Score":"\\textbf{Score}"})

    formated_table = score_table.style.hide(axis="index").format(precision=3).format("\hline {}", score_table.columns[0], escape="latex").set_table_styles([
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
    #"NNDR ST 5th", "NNDR SH 5th", "NNDR TH 5th"
    #"NNDR ST 5th": "NNDR ST", "NNDR SH 5th": "NNDR SH", "NNDR TH 5th": "NNDR TH"
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
        caption = f"Distancia de registros más cercanos entre conjuntos Sinteticos, \emph{{Train}} y \emph{{Hold}}",
        label = f"table-dcr-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-dcr-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(formated_dcr)

    avg = syn.scores[syn.scores["type"] == "avg"]
    avg.sort_values("score", ascending=False).loc[:,:]

    dfs = []
    for model in avg.index:
        for i in ["min", "1p", "5p"]:
            d = avg.loc[[model],[f"record_ST_{i}"]].iloc[0].to_dict()[f"record_ST_{i}"]
            current_data = pd.DataFrame.from_dict(
                    [d["record"], d["closest"], d["closest_2"]]
                ).assign(distance=['Sintético']+[f'DCR{i+1} d({v:.2e})' for i,v in enumerate(d["dists"])]).rename(columns={"distance": "Variable/Distancia"})
            dfs.append(
                current_data.assign(model=model).assign(level=i)
            )
            c_1 = current_data.set_index(["Variable/Distancia"]).T
            percentil = "minimo" if i == "min" else f"percentil {i[0]}"
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
                caption = unicode_to_latex(f"Ejemplos para el modelo {model}, {percentil}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})"),
                label = f"table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}",
                clines=None
            ).replace("\centering", "\\centering\n\\fontsize{10}{14}\\selectfont")
            with open(f"{base_path}/tables/table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}-{model}-{i}.tex", "w") as stext:
                stext.write(current_table)
            

    data_table = pd.concat(dfs).set_index(["model", "level", "Variable/Distancia"])
    latex_table = data_table.style\
        .format_index("{}", escape="latex", axis=1)\
        .format_index("{}", escape="latex", axis=0)\
        .to_latex(
        position="H",
        position_float="centering",
        caption = f"Distancia de registros más cercanos entre conjuntos Sinteticos, \emph{{Train}} y \emph{{Hold}}, {DATASET_NAME.capitalize()} ({DATASET_VERSION.upper()})",
        label = f"table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}",
        clines=None
    )
    with open(f"{base_path}/tables/table-example-{DATASET_NAME.lower()}-{DATASET_VERSION.lower()}.tex", "w") as stext:
        stext.write(latex_table)