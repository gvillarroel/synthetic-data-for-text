## memoria-uchile
# Data Sintética Privada, Generación Vía Modelo Deep Learning
[Link](https://docs.google.com/document/d/1Y4JAyeCSBADCZPokOMzrGdkT5LfFzUcAx6ybxM0UyXc/edit)


## Como utilizar
Se puede descargar con los siguientes comandos
```
pip install dvc
git clone https://github.com/gvillarroel/synthetic-data-for-text.git
cd synthetic-data-for-text
dvc pull
```
Luego simplemente ejecuta los notebooks

## Notebook Disponibles
- [KingCounty](notebooks/kingcounty.ipynb)

## Estructura
### Parametros

| Nombre | Tipo | Descripción |
|--|--|--|
| df | Pandas DataFrame | El dataset a replicar |
| category_columns | tuple | lista de columnas que serán consideradas categorías |
| id | str | columna única que será considerada llave principal |
| synthetic_folder | str |  directorio base donde se construirá los artefacto sinteticos |
| text_columns | tuple | columnas consideradas como texto |
