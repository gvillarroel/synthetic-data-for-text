---
author:
- Gerardo Jorge Villarroel González
bibliography:
- bibliografia.bib
date: 2023
title: 'Data Sintética Privada, EJECUCIÓN Y EVALUACIONES DE MODELOS'
---

A todos los lectores *no organicos*, que para cuando interiorisen estas
palabras espero hayamos aprendido a ser buenos padres

Introducción
============

Cuando se revise esta tesis, estará desactualizada. Desde AlexNet
[@krizhevsky_imagenet_2012] en 2012, el liderazgo en el problema de
clasificación de imágenes ha cambiado al menos 15 veces
[@noauthor_papers_nodate]. En el campo de texto a imágenes, modelos como
DALL-E 2 [@noauthor_dalle_nodate], Google Imagen
[@noauthor_imagen_nodate] y Stable Diffusion [@noauthor_stable_nodate]
fueron presentados en 2022, mientras que para el 2023 se pronostica el
inicio de una carrera de inteligencia artificial en el campo de los
chatbots entre Google y Microsoft
[@milmo_google_2023; @noauthor_microsoft_2023]. En definitiva, es un
campo actualmente en crecimiento y que seguirá sorprendiendo con nuevas
técnicas y productos, en variedad y calidad.

En el contexto de **Equifax**, la empresa en la que se centra este
esfuerzo, es fundamental avanzar de manera rápida y efectiva en el uso
de su información para poder mantenerse a la vanguardia en el mercado y
poder competir con otras empresas del sector.

Según el libro *Practical synthetic data generation: balancing privacy
and the broad availability of data* [@el_emam_practical_2020] los datos
sintéticos ofrecen dos beneficios principales:

1.  Mayor eficiencia en la disponibilidad de datos, y

2.  Mejora en los análisis realizados.

Para **Equifax**, ambos beneficios son valiosos, aunque inicialmente la
eficiencia en la disponibilidad de datos tiene mayor peso. Como se verá
posteriormente, la empresa ejerce un control total sobre el acceso a la
información y los datos, ya que es necesario proteger su
confidencialidad.

El objetivo general de este trabajo es diseñar un mecanismo para generar
conjuntos de datos sintéticos estructurados, que contengan textos, y
compararlos con sus contrapartes originales utilizando deep learning.

Estructura del documento
------------------------

En este documento se presenta un estudio detallado del desarrollo de un
mecanismo para generar conjuntos de datos sintéticos estructurados que
incluyen textos, y se comparan con sus contrapartes originales
utilizando deep learning.

En la **Introducción** se establecerá el contexto del desafío, se
describirán los objetivos a cumplir y se presentará la estructura del
documento.

En el capítulo 2 se realizará una revisión de la literatura sobre
técnicas de generación de datos sintéticos y deep learning.

En el capítulo 3 se detallará el diseño y la implementación del
mecanismo para generar los conjuntos de datos sintéticos y su
comparación con los conjuntos de datos originales.

En el capítulo 4 se presentarán los resultados de la evaluación
comparativa entre los conjuntos de datos sintéticos y los originales.

Finalmente, en el capítulo 5 se presentarán las conclusiones y las
posibles áreas de mejora del trabajo.

Equifax: contexto y limitaciones
--------------------------------

**Equifax** es un buró de crédito multinacional, que en conjunto a
Transunion y Experian componen los tres más grandes a nivel mundial. La
compañía posee equipos de desarrollo en Estados Unidos, India, Irlanda y
Chile. Asimismo está operativa en más de 24 países. El negocio principal
de Equifax es la información/conocimiento extraído de la data
recolectada, la que incluye información crediticia, servicios básicos,
autos, mercadotecnia, Twitter, revistas, informaciones demográficas
entre otros. El principal desafío tecnológico de la compañía es
resguardar la privacidad. El segundo, realizar toda clase de
predicciones relevantes para el mercado con los datos acumulados. Los
datos son uno de los mayores, si no el mayor activo de la compañía.

**Keying and Linking** es el equipo de Equifax encargado de identificar
entidades y relacionarlas dentro de los diferentes conjuntos de datos,
esta labor debe ser aplicada a cada entidad dentro de la compañía y
zonas geográficas. La tarea de la identificación de entidades, entity
resolution, es el proceso de identificar que dos o más registros de
información, que referencian a un único objeto en el mundo real, esto
puede ser una persona, lugar o cosa. Por ejemplo, Bob Smith, Robert
Smith, Robert S. podría referirse a la misma persona, lo mismo puede
darse con una dirección. Es importante destacar que la información
requerida para este equipo es de identificación personal (PII),
categorizada y protegida con las mayores restricciones dentro de la
compañía, de aquí el delicado uso que se dé a los registros y se
prohíben el uso de datos reales en ambientes de desarrollo.

La propuesta actual se enmarca en la búsqueda de un método alternativo
en la generación de data sintética utilizando inteligencia artificial.
La data sintética es utilizada en las pruebas de nuevo software en
ambientes no productivos en Equifax. Para el equipo de **Keying and
Linking** y la compañía es importante la evaluación de los nuevos
desarrollos, pero es aún más importante resguardar la privacidad y
seguridad de los datos. Es por ello que la privacidad y calidad de estos
datos es relevante.

Los métodos actuales que posee Keying and linking para la generación de
data sintética y así probar sus algoritmos son las siguientes, a))
Anonimización de los registros, este método destruye piezas claves de
los registros, para asegurar que no puede ser identificado el dueño de
la información. b)) Generación de data sintética en base de heurísticas,
utilizando conocimiento sobre la estructura de los registros, por
ejemplo, DOB (date of birth) establecen rangos de fechas, o formatos en
el caso de SSN (Security Social Number) o Tarjetas de créditos. c))
Reemplazo por revuelta de datos, se compone de registros reales, pero
mezcla elementos con heurísticas para que no puedan ser identificados,
por ejemplo, mezclando nombres, segmentos de SSN, fechas de nacimiento y
así con todos los registros involucrados. El sistema de revuelta de
datos es el método utilizado, pero debido a peligro de exponer datos
reales, fue limitado a generar un único dataset.

Sobre la regulación y acceso directo a información personal legible, no
enmascarada en Equifax. Esta se encuentra regulada y solo disponibles
para proyectos categorizados como "Protected Data Zone" (PDZ). Estos
proyectos están administrados por el equipo de Ignite, encargado de la
seguridad y herramientas ofrecidas para dichos espacios de trabajo. Los
permisos de acceso son supervisados y revisados cada 3 meses.

Equifax como AI-First Company, está en una evolución en búsqueda de ser
precursora en inteligencia artificial, utilizando los datos almacenados
durante más de un siglo y su asociación con Google, principal proveedor
de servicios en la nube. El objetivo del año 2022, es poseer capacidades
de entrenar modelos de Deep Learning usando las plataformas analíticas
actuales administradas por Ignite, el producto seleccionado y está en
proceso de implementación es Vertex AI. Equifax está en proceso de
evaluación de empresas que generen data sintética con las condiciones
que la organización requiere. Uno de los evaluados es Tonic IA
<https://www.tonic.ai/>. Esto deja ver la relevancia que los datos
sintéticos en los objetivos de Equifax a mediano plazo.

Contexto Temporal/tecnológico
-----------------------------

Usando ChatGPT en el marzo 2023 [@openai_chatgpt_2023].

Introducción a la relevancia de la generación de datos sintéticos para
una tesis.

La generación de datos sintéticos ha surgido como una técnica innovadora
y prometedora en el ámbito de la inteligencia artificial, la ciencia de
datos y el aprendizaje automático. Esta tesis aborda la relevancia de la
generación de datos sintéticos y su impacto en la investigación y el
desarrollo de soluciones tecnológicas. La generación de datos sintéticos
es esencial debido a diversas razones, entre las que destacan la
privacidad, la escasez de datos y la mejora del rendimiento de los
modelos.\
En primer lugar, la privacidad de los datos es un tema de creciente
preocupación en la era digital. La generación de datos sintéticos
permite abordar este problema al crear datos que imitan las
características y la distribución de los datos reales sin revelar
información sensible o identificable. Esto es especialmente relevante en
campos como la medicina, las finanzas o la investigación social, donde
la protección de la privacidad de los individuos es de suma
importancia.\
En segundo lugar, la escasez de datos es un desafío común en diversas
aplicaciones de aprendizaje automático y ciencia de datos. La generación
de datos sintéticos puede mitigar este problema al complementar
conjuntos de datos limitados o desequilibrados. Esto permite a los
investigadores y profesionales desarrollar y evaluar modelos más sólidos
y precisos, mejorando así la calidad y la confiabilidad de las
soluciones propuestas.\
Además, la generación de datos sintéticos contribuye a la mejora del
rendimiento de los modelos de aprendizaje automático. Al ampliar y
enriquecer conjuntos de datos existentes, los modelos pueden aprender
patrones y relaciones más complejas y generalizables, lo que se traduce
en una mejor capacidad de predicción y clasificación.\
Esta tesis examinará las técnicas y enfoques actuales en la generación
de datos sintéticos, así como las aplicaciones y desafíos asociados a su
implementación en diferentes contextos. También se analizará el papel de
los datos sintéticos en la ética y la privacidad de los datos y su
impacto en la toma de decisiones basada en datos en el mundo real.

Objetivo
--------

**Objetivo General:**

El objetivo general es definir un mecanismo para generar conjuntos de
datos sintéticos estructurados, que incluyen textos y comparar, mediante
modelos generativos y su contraparte original.

**Objetivos Específicos:**

1.  Elaborar modelos generativos para sintetizar nuevos conjuntos de
    datos, a partir de los originales que incluyen textos.

2.  Comparar los conjuntos de datos sintéticos y originales en 2 casos,
    propiedades estadísticas, distribuciones, privacidad y frecuencia de
    palabras para campos de textos.

Revisión Bibliográfica
======================

Tipos de Datos {#tipo-de-datos}
--------------

Los tipos de datos tienen diversas implicaciones en su generación, como
su representación, almacenamiento y procesamiento. Los datos
estructurados se presentan en la .

En 2012, IDC estimó que para 2020, más del 95% de los datos serían no
estructurados [@gantz_digital_2012]. En un análisis posterior, Kiran
Adnan y Rehan Akbar [@adnan_analytical_2019] encontraron que el texto es
el tipo de dato no estructurado que más rápido crece en las
publicaciones, seguido por la imagen, el video y finalmente el audio.

La resume la lista que se encuentra en *Practical Statistics for Data
Scientists* [@bruce_practical_2020].

::: {#tabla-tipo-datos}
  T   Sub tipo                                                                                                            Descripción                                                                Ejemplos
  --- ------------------------------------------------------------------------------------------------------------------- -------------------------------------------------------------------------- ---------------------------
      Datos establecidos como números                                                                                     \-                                                                         
      Continuo                                                                                                            Datos que pueden tomar cualquier valor en un intervalo                     3.14 metros, 1.618 litros
      Discreto                                                                                                            Datos que solo pueden tomar valores enteros                                1 habitación, 73 años
      Datos que pueden tomar solo un conjunto específico de valores que representan un conjunto de categorías posibles.   \-                                                                         
      Binario                                                                                                             Un caso especial de datos categóricos con solo dos categorías de valores   0/1, verdadero/falso
      Ordinal                                                                                                             Datos categóricos que tienen un ordenamiento explícito.                    pequeña/ mediana/ grande

  : Tipos de datos estructurados
:::

Privacidad de Datos
-------------------

La protección de la información es un aspecto fundamental en la
generación de datos sintéticos. Aunque este aspecto puede no ser crucial
cuando los datos corresponden a temas como recetas o automóviles,
resulta esencial cuando se trata de información relacionada con
individuos [@bruce_practical_2020]. Por esta razón, el resguardo de la
información es un tema de importancia para entidades como Equifax, que
gestionan una gran cantidad de conjuntos de datos con contenido
personal.

### Tipo de datos a ser protegidos

Para identificar qué campos de datos son significativos desde el punto
de vista de la privacidad, se puede recurrir a la definición resumida en
la del texto *Data privacy: Definitions and techniques*
[@de_capitani_di_vimercati_data_2012].

::: {#data-relevante}
  Tipo de revelación                Descripción
  --------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Identificadores**               Atributos que identifican de manera única a individuos (por ejemplo, SSN, RUT, DNI).
  **Cuasi-identificadores (QI)**    Atributos que, en combinación, pueden identificar a individuos, o reducir la incertidumbre sobre sus identidades (por ejemplo, fecha de nacimiento, género y código postal).
  **Atributos confidenciales**      Atributos que representan información sensible (por ejemplo, enfermedad).
  **Atributos no confidenciales**   Atributos que los encuestados no consideran sensibles y cuya divulgación es inofensiva (por ejemplo, color favorito).

  : Niveles de revelación y ejemplos
:::

### Tipos de riesgos de divulgación

Los tipos de divulgación definidos en *Practical Synthetic Data
Generation* [@bruce_practical_2020] están resumidos en la .

::: {#relevantes-definiciones}
  Tipo de revelación                     Descripción
  -------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Divulgación de identidad**           Este riesgo se refiere a la posibilidad de que un atacante pueda identificar la información de un individuo a partir de los datos publicados, utilizando técnicas de filtrado para reducir las posibilidades hasta un solo individuo.
  **Divulgación de nueva información**   Este riesgo comprende el riesgo de Divulgación de Identidad, y además, implica la adquisición de información adicional sobre el individuo a partir de los datos publicados.
  **Divulgación de Atributos**           Este riesgo se da cuando, aunque no se pueda identificar a un individuo, se puede descubrir un atributo común en varios registros, lo que permite obtener información sensible acerca de un grupo de individuos.
  **Divulgación Inferencial**            Este riesgo se refiere a la posibilidad de inferir información sensible a partir de los datos publicados, mediante el uso de técnicas de análisis estadístico o de aprendizaje automático. Por ejemplo, si después de filtrar todos los registros, el 80% de los registros con las mismas características tienen cáncer, se podría inferir que el individuo buscado puede tener cáncer.

  : Tipos de Riesgos de Divulgación y sus Descripciones
:::

Adicionalmente se deben establecer dos conceptos relevantes ante el
análisis de revelación de información:

1.  En términos prácticos, normalmente los datos sintéticos buscan tener
    cierta permeabilidad con respecto a la **Divulgación Inferencial**,
    ya que se quiere que estadísticamente sean similares. Además, se
    busca proteger la identidad de los individuos, pero esta no es la
    única condición, también se busca proteger aquellos atributos que
    pueden ser sensibles, como las enfermedades. A todo este conjunto se
    le denomina **Revelación de identidad significativa**. Es
    particularmente riesgoso por la posibilidad de discriminación hacia
    ciertos grupos que cumplen con los atributos criterio.

2.  Los mismos atributos pueden tener más relevancia para ciertos grupos
    de la población que para otros. El ejemplo que se indica en
    [@el_emam_practical_2020] es que, debido a que el número de hijos
    igual a 2 es menos frecuente en una etnia que en otra (40% en la
    primera y 10% en la segunda), ese dato es más relevante en la
    segunda. Esto se debe a que es un factor que filtra mejor y, por lo
    tanto, puede permitir un mejor conocimiento de ese grupo específico.
    A esto se le denomina **Definición de información ganada**.

### Regulación de datos sintéticos

Debido a que los datos sintéticos son basados en datos reales, pueden
ser afectos a las regulaciones de sobre protección de datos
[@bruce_practical_2020]. Los nuevos datos podrían ser afectos por:

1.  [Regulation (EU) 2016/679 of the European Parliament and of the
    Council](https://dvbi.ru/Portals/0/DOCUMENTS_SHARE/RISK_MANAGEMENT/EBA/GDPR_eng_rus.pdf)
    [@regulation_regulation_2016], si el proceso de generación de datos
    sintéticos a menudo implica el uso de datos personales reales como
    entrada. En este caso, el GDPR sería relevante. Las organizaciones
    que utilicen datos personales para generar datos sintéticos deben
    garantizar que este proceso cumple con los principios del GDPR, como
    la minimización de datos (sólo se deben utilizar los datos
    necesarios) y la limitación de la finalidad (los datos sólo se deben
    utilizar para el propósito para el que se recogieron).

2.  [The California consumer privacy act: Towards a European-style
    privacy regime in the United
    States](https://heinonline.org/HOL/LandingPage?handle=hein.journals/jtlp23&div=5&id=&page=)
    [@pardau_california_2018]

3.  [Health insurance portability and accountability act of
    1996](http://www.eolusinc.com/pdf/hipaa.pdf) [@act_health_1996]

### Protección de Privacidad

En la se listas las utilizadas en diferentes publicaciones para
determinar la privacidad efectiva de los conjuntos generados.

::: {#metricas-privacidad}
  Tipo de revelación                              Descripción
  ----------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  ***Distance to Closest Record* (DCR)**          DRC se utiliza para medir la distancia euclidiana entre cualquier registro sintético y su vecino real más cercano. Idealmente, cuanto mayor sea la DCR, menor será el riesgo de violación de la privacidad. Además, se calcula el percentil 5 de esta métrica para proporcionar una estimación robusta del riesgo de privacidad. [@zhao_ctab-gan_2021]
  ***Nearest Neighbour Distance Ratio* (NNDR)**   NNDR mide la relación entre la distancia euclidiana del vecino real más cercano y el segundo más cercano para cualquier registro sintético correspondiente. Esta relación se encuentra dentro del intervalo \[0, 1\]. Los valores más altos indican una mayor privacidad. Los bajos valores de NNDR entre datos sintéticos y reales pueden revelar información sensible del registro de datos reales más cercano. [@zhao_ctab-gan_2021]

  : Metricas de privacidad
:::

Generación de Datos Sintéticos
------------------------------

Los datos sintéticos, aunque no son datos reales, se generan con la
intención de preservar ciertas propiedades de los datos originales. La
utilidad de los datos sintéticos se mide por su capacidad para servir
como un sustituto efectivo de los datos originales
[@bruce_practical_2020]. Basándose en el uso de los datos originales,
los datos sintéticos se pueden clasificar en tres categorías: aquellos
que se basan en datos reales, los que no se basan en datos reales, y los
híbridos.

**Datos basados en datos reales**: utilizan modelos que aprenden la
distribución de los datos originales para generar nuevos puntos de datos
similares.

**Datos no basados en datos reales**: utilizan conocimientos del mundo
real. Por ejemplo, se podría formar un nombre completo seleccionando
aleatoriamente un nombre y un apellido de un conjunto predefinido.

**Híbridos**: estos combinan técnicas de imitación de distribución con
algunos campos que no derivan de los datos reales. Esto puede ser
especialmente útil cuando se intenta desacoplar las distribuciones de
datos que podrían ser sensibles o generar discriminación, como la
información sobre la etnia.

En la , se revisaron los datos estructurados. Si bien cada tipo puede
tener muchas representaciones, por ejemplo, los datos continuos podrían
considerarse como *float*, *datetime* o incluso intervalos
personalizados, como de 0 a 1. Sobre estos datos estructurados, se
pueden generar estructuras para unirlos.

Entre las estructuras más comunes se encuentran las matrices
bidimensionales (datos tabulares) y los arreglos, que permiten matrices
de muchas dimensiones e incluso estructuras complejas que pueden mezclar
todas las estructuras previas.

Debido al objetivo, se detallan solo los modelos que permiten abordar la
generación de datos tabulares y texto basados en datos reales.

### Generación de datos tabulares

En la , se resumen las últimas publicaciones sobre generación de datos
tabulares, indicando la fecha de publicación y si se puede acceder al
código fuente o no, a febrero de 2023.

::: {#tab-sota-tab}
  Nombre                                                                                                                 Fecha $\downarrow$                                                               Código
  -------------------------------------------------------------------------------------------------------------------- -------------------- --------------------------------------------------------------------
  REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers [@solatorio_realtabformer_2023]             2023-02-04               [Github](https://github.com/avsolatorio/REaLTabFormer)
  PreFair: Privately Generating Justifiably Fair Synthetic Data [@pujol_prefair_2022]                                            2022-12-20 
  GenSyn: A Multi-stage Framework for Generating Synthetic Microdata using Macro Data Sources [@acharya_gensyn_2022]             2022-12-08                        [Github](https://github.com/Angeela03/GenSyn)
  TabDDPM: Modelling Tabular Data with Diffusion Models [@kotelnikov_tabddpm_2022]                                               2022-10-30                         [Github](https://github.com/rotot0/tab-ddpm)
  Language models are realistic tabular data generators [@borisov_language_2022]                                                 2022-10-12                      [Github](https://github.com/kathrinse/be_great)
  Ctab-gan+: Enhancing tabular data synthesis [@zhao_ctab-gan_2022]                                                              2022-04-01                  [Github](https://github.com/Team-TUD/CTAB-GAN-Plus)
  Ctab-gan: Effective table data synthesizing [@zhao_ctab-gan_2021]                                                              2021-05-31                       [Github](https://github.com/Team-TUD/CTAB-GAN)
  Modeling Tabular data using Conditional GAN [@xu_modeling_2019]                                                                2019-10-28                             [Github](https://github.com/sdv-dev/SDV)
  SMOTE: synthetic minority over-sampling technique [@chawla_smote_2002]                                                         2002-06-02   [Github](https://github.com/scikit-learn-contrib/imbalanced-learn)

  : Estado del arte en generación de datos tabulares
:::

### Generación de texto en base de datos tabulares

En la , se listan las publicaciones en la generación de texto a partir
de datos estructurados.

::: {#tab-sota-text}
  Nombre                                                                                  Fecha $\downarrow$   Modelo Base
  ------------------------------------------------------------------------------------- -------------------- -------------
  Table-To-Text generation and pre-training with TABT5 [@andrejczuk_table--text_2022]             2022-10-17            T5
  Text-to-text pre-training for data-to-text tasks [@kale_text--text_2020]                        2021-07-09            T5
  TaPas: Weakly supervised table parsing via pre-training [@herzig_tapas_2020]                    2020-04-21          Bert

  : Estado del arte en generación de textos en base a datos
:::

El estado del arte en la generación de texto a partir de datos tabulares
es TabT5. Es importante notar que la tabla mezcla los enfoques de
*Table-To-Text* y *Data-To-Text*. Aunque ninguna de las publicaciones
incluye código asociado, no es necesario, ya que utilizan modelos
abiertos como base (T5 y Bert). Lo más relevante en estos casos es el
proceso de *fine-tuning*. Para completar la tarea de generar nuevos
textos a partir de información inicial, esta información debe ser
codificada para poder ser procesada por el modelo utilizado.

La diferencia entre *Table-To-Text* y *Data-To-Text* radica en el
formato de información de entrada. en *Table-To-Text* es una tabla con
multiples filas y en *Data-To-Text* corresponde a un solo objeto con sus
propiedades. A continuación ejemplos de entradas de los modelos.

En los siguientes ejemplos, se utilizará la para ilustrar cómo se puede
utilizar para generar texto utilizando los modelos de *fine-tuning*
mencionados anteriormente. Esta tabla representa información sobre
películas, incluyendo el nombre de la película, el director, el año de
lanzamiento y el género, y se utilizará para generar preguntas y
respuestas a partir de la información proporcionada.

::: {#tabla-ejemplo-inputs}
  Nombre de la Película            Director       Año de Lanzamiento   Género
  -------------------------------- -------------- -------------------- -----------------
  Star Wars: Una Nueva Esperanza   George Lucas   1977                 Ciencia ficción

  : Ejemplo de tabla de entrada
:::

Para los modelos TabT5 y TaPas, se utiliza el mismo preprocesamiento
para convertir la tabla de entrada en una pregunta/tarea y respuesta
[@andrejczuk_table--text_2022; @herzig_tapas_2020]. En este ejemplo, la
tabla representa información sobre películas, y se utiliza para generar
una pregunta y respuesta sobre el director de la película \"Star Wars:
Una Nueva Esperanza\". La pregunta se construye a partir de la
información de la tabla, y la respuesta se espera que sea el nombre del
director. Una vez que se ha generado la pregunta y la respuesta, se
puede utilizar un modelo de *fine-tuning* como TabT5 o TaPas para
generar texto a partir de la información proporcionada. En resumen, el
proceso de generación de texto a partir de datos tabulares implica la
conversión de información tabular en preguntas y respuestas, y luego la
utilización de modelos de *fine-tuning* para generar texto a partir de
estas preguntas y respuestas.

Table: Películas Nombre de la Película \| Director \| Año de Lanzamiento
\| Género Star Wars: Una Nueva Esperanza \| George Lucas \| 1977 \|
Ciencia ficción

¿Qué director dirigió la película Star Wars: Una Nueva Esperanza?

George Lucas

En cambio, el modelo *Text-to-text pre-training for data-to-text tasks*
[@kale_text--text_2020] utiliza una entrada diferente, que consiste en
una serie de tuplas que representan las propiedades de la entidad y sus
valores correspondientes. Se espera que el modelo identifique la tupla
relevante y genere una pregunta y respuesta correspondientes. Una vez
generada la pregunta y respuesta, se puede utilizar el modelo de
fine-tuning correspondiente para generar texto a partir de ellas. En
conclusión, la generación de texto a partir de datos tabulares implica
una conversión adecuada de la información de entrada en un formato
apropiado para cada modelo, la identificación de la pregunta o tarea
relevante y la utilización del modelo correspondiente para generar el
texto resultante.

\<Star Wars: Una Nueva Esperanza, Director, George Lucas\>,\
\<Star Wars: Una Nueva Esperanza, Año de Lanzamiento, 1977\>,\
\<Star Wars: Una Nueva Esperanza, Género, Ciencia ficción\>

¿Qué director dirigió la película Star Wars: Una Nueva Esperanza?

George Lucas

Metricas de evaluación
----------------------

Es importante destacar que no todas estas métricas son aplicables a
todos los tipos de datos y modelos, y que la selección de las métricas a
utilizar debe ser cuidadosamente considerada en función de las
necesidades y objetivos específicos de cada caso de estudio. A
continuación presentan algunas de las posibles a considerar para medir
la similitud, privacidad y utilidad en la evaluación de los conjuntos de
datos sintéticos generados.

### SDMetrics

SDMetrics es una herramienta que proporciona un conjunto de métricas
para la evaluación de conjuntos sintéticos. La herramienta utiliza dos
métodos de cálculo: el método de Reporte y el método de Diagnóstico.

#### SDMetrics Report

El informe de SDMetrics genera una puntuación de evaluación para un
conjunto sintético al compararlo con el conjunto real. La puntuación
utiliza *KSComplement* en tablas numéricas y *TVComplement* en caso de
campos categóricos. El promedio de las columnas compone la métrica
*Column Shapes*. Además, se utiliza *CorrelationSimilarity* en campos
numéricos y *ContingencySimilarity* en campos categóricos o
combinaciones entre campos categóricos y numéricos.

Estas cuatro métricas, TVComplement, KSComplement, CorrelationSimilarity
y ContingencySimilarity, son utilizadas en la biblioteca de Python de
código abierto *SDMetrics* para evaluar datos sintéticos tabulares. Cada
una de estas métricas tiene un enfoque diferente para evaluar la calidad
de los datos sintéticos.

TVComplement se enfoca en la similitud entre una columna real y una
columna sintética en términos de sus formas, mientras que KSComplement
utiliza la estadística de Kolmogorov-Smirnov para calcular la máxima
diferencia entre las funciones de distribución acumulativa de dos
distribuciones numéricas. Por otro lado, CorrelationSimilarity mide la
correlación entre un par de columnas numéricas y calcula la similitud
entre los datos reales y sintéticos, comparando las tendencias de las
distribuciones bidimensionales. Finalmente, ContingencySimilarity mide
la similitud entre dos variables categóricas utilizando la tabla de
contingencia y la estadística del coeficiente de contingencia,
proporcionando una medida de la dependencia entre las dos variables.

Cada una de estas métricas tiene una forma diferente de evaluar la
calidad de los datos sintéticos y, por lo tanto, proporciona información
valiosa sobre diferentes aspectos de la calidad de los datos.
TVComplement se enfoca en la distribución marginal o el histograma
unidimensional de la columna, mientras que KSComplement se centra en la
diferencia entre las funciones de distribución acumulativa de dos
distribuciones numéricas. CorrelationSimilarity mide la similitud entre
los datos reales y sintéticos basándose en la correlación entre un par
de columnas numéricas, y ContingencySimilarity mide la similitud entre
dos variables categóricas utilizando la tabla de contingencia y la
estadística del coeficiente de contingencia. Juntas, estas métricas
proporcionan una evaluación más completa de la calidad de los datos
sintéticos.

#### SDMetrics Diagnostic

Esta herramienta utiliza una variedad de métricas para proporcionar
información valiosa sobre la calidad de los datos, incluyendo
RangeCoverage, BoundaryAdherence y CategoryCoverage.

RangeCoverage es una métrica que mide la proporción del rango de valores
posibles para una característica que está cubierta por los datos. Se a
como la relación entre el rango de valores observados y el rango de
valores posibles para esa característica. Esta métrica puede ayudar a
identificar si los datos tienen una cobertura adecuada en términos de la
variedad de valores posibles que podría tomar la característica.

BoundaryAdherence es una métrica que mide la proporción de puntos de
datos que caen dentro de los límites especificados para una
característica. Se calcula como la relación entre el número de puntos de
datos que caen dentro de los límites y el número total de puntos de
datos. Esta métrica es útil para evaluar si los datos se ajustan a los
límites especificados para la característica, lo que puede ser
importante en situaciones donde se espera que la característica tenga
ciertos valores o límites específicos.

CategoryCoverage es una métrica que mide la proporción de categorías
posibles para una característica categórica que está cubierta por los
datos. Se calcula como la relación entre el número de categorías
observadas y el número total de categorías posibles para esa
característica. Esta métrica puede ayudar a identificar si los datos
tienen una cobertura adecuada en términos de la variedad de categorías
posibles que podría tomar la característica categórica.

En resumen, *SDMetrics Diagnostic* utiliza RangeCoverage,
BoundaryAdherence y CategoryCoverage para evaluar la calidad de los
datos tabulares. Estas métricas proporcionan información valiosa sobre
la cobertura de los datos en términos de rango de valores, límites y
categorías posibles, lo que puede ayudar a identificar problemas en la
calidad de los datos.

### Conjuntos Estadísticos

::: {#tab-stats}
  Nombre                                              Descripción
  --------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Nombre                                              Descripción
                                                      
  Media (Mean)                                        La suma de todos los valores dividido por el número total de valores
  Mediana (Median)                                    El valor que se encuentra en el centro de un conjunto de datos ordenados de menor a mayor. Es decir, la mitad de los valores son mayores que la mediana y la otra mitad son menores
  Moda (Mode)                                         El valor que aparece con mayor frecuencia en un conjunto de datos
  Mínimo (Min)                                        El valor más pequeño en un conjunto de datos
  Máximo (Max)                                        El valor más grande en un conjunto de datos
  Percentil (25, 75) (Percentile)                     El valor tal que P (25 o 75) por ciento de los datos son menores que él, y el restante (100 - P) por ciento son mayores. Cuando P = 50, el percentil es la mediana
  Media Truncada (Trimmed Mean)                       El promedio de todos los valores, una vez que se han eliminado un porcentaje de los valores más bajos y un porcentaje de los valores más altos
  Outlier                                             Un valor que se encuentra muy lejos de la mayoría de los valores en un conjunto de datos
  Desviación (Deviation)                              La diferencia entre un valor observado y la estimación de ese valor
  Varianza (Variance)                                 La medida de cuán dispersos están los valores en un conjunto de datos. Es la suma de los cuadrados de las desviaciones desde la media dividido por n - 1, donde n es el número de valores
  Desviación Estándar (SD)                            La raíz cuadrada de la varianza
  Desviación Absoluta Media (MAD)                     La media de los valores absolutos de las desviaciones desde la media
  Rango (Range)                                       La diferencia entre el valor más grande y el valor más pequeño en un conjunto de datos
  Tablas de Frecuencia (Frequency Tables)             Un método para resumir los datos al contar cuántas veces ocurre cada valor en un conjunto de datos
  Probabilidad (Probability)                          La medida de la posibilidad de que un evento ocurra. Se establece como el número de ocurrencias de un valor dividido por el número total de ocurrencias
  Tabla de Contingencia (Contingency Table)           Una tabla que muestra la distribución conjunta de dos o más variables categóricas
  Correlación                                         Una medida estadística que indica cómo dos variables numéricas están relacionadas entre sí. Puede variar entre -1 y 1
  Distribución Estratificada                          Una comparación de la distribución de datos para diferentes estratos
  Comparación de Modelos Predictivos Multivariables   Un método para comparar varios modelos predictivos que involucran múltiples variables. Implica la construcción de modelos separados para cada variable objetivo y comparar la curva ROC (Receiver Operating Characteristic) para cada modelo
  Distinguibilidad                                    Un método para evaluar la calidad de los conjuntos de datos sintéticos. Implica la creación de un modelo que intenta distinguir entre conjuntos de datos reales y sintéticos. Un buen conjunto sintético es aquel que el modelo no puede distinguir de los datos reales
  Kullback-Leibler                                    Una medida de la divergencia entre dos distribuciones de probabilidad
  Pairwise Correlation                                Una medida de la similitud entre dos conjuntos de datos que compara las correlaciones de cada par de variables en los conjuntos de datos
  Log-Cluster                                         Un método para evaluar la calidad de los conjuntos de datos sintéticos que compara la estructura de los conjuntos de datos reales y sintéticos mediante el uso de clustering
  Cobertura de Soporte (Support Coverage)             Una medida de qué tan bien los datos sintéticos representan la distribución de los datos reales. Se mide como la proporción de variables en el conjunto de datos real que están representadas en el conjunto de datos sintéticos
  Cross-Classification                                Un método para evaluar la calidad de los conjuntos de datos sintéticos que compara la precisión de los modelos predictivos construidos a partir de los conjuntos de datos reales y sintéticos
  Métrica de Revelación Involuntaria                  Una medida de qué tan bien se protege la privacidad de los datos en un conjunto de datos sintético. Se mide como la tasa de predicciones correctas de atributos sensibles de un individuo en un conjunto de datos sintético

  : Listado de conjunto estadísticos
:::

Desarrollo
==========

Recursos disponibles
--------------------

### Conjuntos de datos

A continuación se listan y detallan los conjuntos de datos utilizados en
los experimentos.

#### King County

El conjunto de datos King County [@kaggle_house_2015] contiene
información sobre precios de venta y características de 21,613 viviendas
en Seattle y King County de los años 2014 y 2015. El conjunto de datos
incluye información como el número de habitaciones, el número de baños,
la superficie del terreno y la superficie construida, así como
información sobre la ubicación de la propiedad, como la latitud y la
longitud. Este conjunto de datos es comúnmente utilizado para tareas de
regresión y predicción de precios de viviendas. Sus campos se listan en
.

::: {#data-county}
  Variable      Descripción
  ------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  id            Identificación
  date          Fecha de venta
  price         Precio de venta
  bedrooms      Número de dormitorios
  bathrooms     Número de baños
  sqft\_liv     Tamaño del área habitable en pies cuadrados
  sqft\_lot     Tamaño del terreno en pies cuadrados
  floors        Número de pisos
  waterfront    '1' si la propiedad tiene vista al mar, '0' si no
  view          Índice del 0 al 4 de la calidad de la vista de la propiedad
  condition     Condición de la casa, clasificada del 1 al 5
  grade         Clasificación por calidad de construcción que se refiere a los tipos de materiales utilizados y la calidad de la mano de obra. Los edificios de mejor calidad (grado más alto) cuestan más construir por unidad de medida y tienen un valor más alto. Información adicional en: KingCounty
  sqft\_above   Pies cuadrados sobre el nivel del suelo
  sqft\_basmt   Pies cuadrados debajo del nivel del suelo
  yr\_built     Año de construcción
  yr\_renov     Año de renovación. '0' si nunca se ha renovado
  zipcode       Código postal de 5 dígitos
  lat           Latitud
  long          Longitud
  sqft\_liv15   Tamaño promedio del espacio habitable interior para las 15 casas más cercanas, en pies cuadrados
  sqft\_lot15   Tamaño promedio de los terrenos para las 15 casas más cercanas, en pies cuadrados
  Shape\_leng   Longitud del polígono en metros
  Shape\_Area   Área del polígono en metros

  : Conjunto de datos King County
:::

#### Economicos

Economicos.cl es un sitio web chileno que se dedica a la publicación de
avisos clasificados en línea, principalmente en las categorías de bienes
raíces, vehículos, empleos, servicios y productos diversos. El conjunto
de datos corresponde a un *Web Scraping* realizado en 2020, contiene
22.059 observaciones.

::: {#data-economicos}
  Variable            Descripción
  ------------------- -----------------------------------------------
  url                 URL de la publicación
  Descripción         Descripción de la publicación
  price               Precio de venta, en dolares, UF o pesos
  property\_type      Tipo de propiedad: Casa, Departamento, ETC
  transaction\_type   Tipo de transactión Arriendo, Venta
  state               Región de la publicación
  county              Comuna de la publicación
  publication\_date   Día de la publicación
  rooms               Número de dormitorios
  bathrooms           Número de baños
  m\_built            Tamaño del área habitable en metros cuadrados
  m\_size             Tamaño del terreno en metros cuadrados
  source              Diario de la publicación
  title               Titulo de la publicación
  address             Dirección de la publicación
  owner               Publicante
  \_price             Precio traspasado a UF

  : Conjunto de datos Economicos.cl
:::

### Computación y Software

Para llevar a cabo los experimentos, se utilizó un computador con las
siguientes especificaciones técnicas, como se muestra en la . El
procesador empleado fue un AMD Ryzen 9 7950X 16-Core Procesadores, con
cuatro modulos de 32 GB para una memoria total de 128 GB DDR5. La
tarjeta gráfica empleada fue una NVIDIA GeForce RTX 4090, y se contó con
dos discos duros de 500 GB SSD. La utilización de un equipo con estas
características permitió una ejecución eficiente de los modelos de
generación de datos, asegurando la viabilidad de los experimentos. Es
importante destacar que la elección de los componentes del computador
fue cuidadosamente considerada para asegurar que los resultados
obtenidos no se vieran limitados por un hardware insuficiente.

En relación al software utilizado, se trabajó con el sistema operativo
Ubuntu 20.04.2 LTS y se empleó el lenguaje de programación Python 3.10
para el desarrollo de los modelos de generación de datos. Se utilizaron
diversas bibliotecas, incluyendo DVC, SDV y PyTorch, cuya lista completa
se puede encontrar en el [repositorio en
Github](https://github.com/gvillarroel/synthetic-data-for-text/blob/main/freeze.txt).
La elección de estas herramientas se basó en la compatibilidad con el
modelo TabDDPM, el cual fue utilizado en algunos de los experimentos.

::: {#tabla-componentes-pc}
  Componente        Descripción
  ----------------- -------------------------------------
  Procesador        AMD Ryzen 9 7950X 16-Core Processor
  Memoria RAM       128 GB DDR5
  Tarjeta gráfica   NVIDIA GeForce RTX 4090
  Disco duro        1 TB SSD

  : Computador Usado
:::

En favor de la reproducibilidad, se utilizó *devcontainer*, el cual
establece el entorno de desarrollo y pruebas mediante una imagen de
*Docker* replicable. Los experimentos pueden ser replicados utilizando
el contenedor descrito en el repositorio.

El código fuente de los modelos de generación de datos, así como los
scripts de análisis y visualización de los resultados, se encuentra
disponible en un repositorio público de Github:
[gvillarroel/synthetic-data-for-text](https://github.com/gvillarroel/synthetic-data-for-text)

Desarrollo del flujo de procesamiento
-------------------------------------

A continuación se describe el flujo de procesamiento utilizado para
generar nuevos datos sintéticos. Este flujo se basa en el propuesto por
Synthetic Data Vault (SDV), con algunas modificaciones para guardar
etapas intermedias.

SDV es un ecosistema de bibliotecas de generación de datos sintéticos
que permite a los usuarios aprender conjuntos de datos de una sola
tabla, de múltiples tablas y de series de tiempo, y luego generar nuevos
datos sintéticos con las mismas propiedades estadísticas y el mismo
formato que los conjuntos de datos originales. Para ello, SDV utiliza
diferentes técnicas, como modelos generativos y redes neuronales, para
aprender la distribución subyacente de los datos y generar nuevos datos
que sigan dicha distribución
[@kotelnikov_overview_nodate; @patki_synthetic_2016].

A continuación se describe el proceso de generación de datos sintéticos
para una tabla única utilizando la biblioteca Synthetic Data Vault
(SDV), seguido de las modificaciones realizadas para extender el proceso
y agregar nuevos modelos.

En la se muestran los pasos necesarios para generar un conjunto de datos
sintéticos utilizando SDV:

1.  **Create Metadata**: Se crea un diccionario que define los campos
    del conjunto de datos y los tipos de datos que posee. Esto permite a
    SDV aprender la estructura del conjunto de datos original y
    utilizarla para generar nuevos datos sintéticos con la misma
    estructura.

2.  **Create Model**: Se selecciona el modelo de generación de datos a
    utilizar. SDV ofrece varios modelos, incluyendo GaussianCopula,
    CTGAN, CopulaGAN y TVAE, que se adaptan a diferentes tipos de datos
    y distribuciones.

3.  **Fit Model**: El modelo seleccionado se entrena con el conjunto de
    datos original para aprender sus distribuciones y patrones
    estadísticos.

4.  **Generate Synthetic Dataset**: Con el modelo ya entrenado, se
    generan nuevos datos sintéticos con la misma estructura y
    características estadísticas que el conjunto original. Este nuevo
    conjunto de datos puede ser utilizado para diversas aplicaciones,
    como pruebas de software o análisis de datos sensibles.

Es importante destacar que el proceso de generación de datos sintéticos
con SDV es escalable y puede utilizarse con conjuntos de datos de una
sola tabla, múltiples tablas y series de tiempo. Además, en este
proyecto se realizaron algunas modificaciones al flujo para extender el
proceso y permitir la inyección de nuevos modelos.

En el proceso de generación de datos sintéticos con SDV extendido, se
incluyen dos nuevas etapas para poder guardar los modelos intermedios y
los resultados de la evaluación. El proceso completo se muestra en la y
consta de los siguientes pasos:

1.  **Create Metadata**: Crea un diccionario que define los campos del
    conjunto de datos y los tipos de datos que posee.

2.  **Create Model**: Se selecciona el modelo a utilizar. SDV permite
    GaussianCopula, CTGAN, CopulaGAN y TVAE.

3.  **Fit Model**: El modelo seleccionado toma el conjunto original para
    entrenar el modelo y aprender sus distribuciones.

4.  **Save Model**: El modelo entrenado se guarda en un archivo para su
    uso posterior.

5.  **Generate Synthetic Dataset**: Genera un nuevo conjunto de datos
    usando el modelo entrenado.

6.  **Evaluate & Save Metrics**: Evalúa y guarda el conjunto de datos
    sintético generado mediante métricas como la correlación, el error
    absoluto medio y el error cuadrático medio.

Con estas nuevas etapas, se pueden guardar los modelos intermedios y los
resultados de la evaluación, lo que permite una mayor flexibilidad en el
proceso y la capacidad de utilizar los modelos y los resultados en
posteriores experimentos.

Modelos
-------

Los modelos de generación de datos tabulares utilizan como base la
metodología propuesta por *Synthetic Data Vault* (SDV), mientras que
para los modelos de generación de texto se utiliza la biblioteca Hugging
Face para cargar, realizar *fine-tuning* con nuevas tareas y evaluar el
modelo basado en mT5.

### Modelos para datos tabulares

Para que un modelo pueda ser utilizado con el SDV, es necesario que
implemente los siguientes métodos:

1.  **load**: Carga el modelo desde un archivo

2.  **fit**: Entrena el modelo, utilizando un pandas dataframe como
    entrada

3.  **save**: Guarda el modelo en un archivo

4.  **sample**: Genera un conjunto de registros nuevos utilizando el
    modelo entrenado.

Como consideración adicional, se recomienda ejecutar el proceso
utilizando un script en lugar de un notebook, ya que se ha observado que
el notebook puede fallar con algunos modelos debido a limitaciones de
memoria. A continuación, se detallan los pasos a seguir para la
ejecución del proceso:

1.  Crear un archivo de configuración que contenga la información
    necesaria para la generación de datos sintéticos, como la ruta del
    conjunto de datos original y la configuración de los modelos a
    utilizar.

2.  Crear un script que cargue la configuración, ejecute el proceso de
    generación de datos sintéticos y guarde el conjunto de datos
    sintético resultante.

3.  Ejecutar el script creado en el paso anterior.

De esta manera, se puede ejecutar el proceso de generación de datos
sintéticos de forma automatizada y con una mayor capacidad de
procesamiento, lo que puede mejorar el desempeño del proceso y reducir
los tiempos de ejecución. Vea

La clase *Synthetic* es una implementación que permite configurar los
modelos a utilizar en el proceso de generación de datos sintéticos. Esta
clase encapsula los métodos comunes de los modelos, como *load*, *fit*,
*save* y *sample*, permitiendo así una configuración general de las
entradas y la selección de modelos.

En el ejemplo mostrado en el Código , se instancia la clase *Synthetic*
con un pandas dataframe previamente pre-procesado. Se especifican las
columnas que se considerarán como categorías, las que se considerarán
como texto y las que se excluirán del análisis. Además, se indica el
directorio donde se guardarán los archivos temporales, se seleccionan
los modelos a utilizar, se establece el número de registros sintéticos
deseados y se define una columna objetivo para realizar pruebas con
machine learning y estratificar los conjuntos parciales de datos que se
utilizarán. De esta manera, se configura de manera flexible el proceso
de generación de datos sintéticos según las necesidades específicas del
usuario.

La presenta las opciones para la instancia de la clase *Synthetic*:

::: {#synthetic-input}
  **Variable**        **Descripción**
  ------------------- --------------------------------------------------------------------------------------------------------------------------------------------
  df                  Pandas DataFrame a utilizar
  Id                  Nombre de la columna a ser usada como identificadora
  category\_columns   Listado de columnas categóricas
  text\_columns       Listado de columnas de texto
  exclude\_columns    Listado de columnas que deben ser excluidas
  synthetic\_folder   Carpeta donde se guardarán los documentos intermedios y finales
  models              Listado de modelos a utilizar
  n\_sample           Número de registros a generar
  target\_column      Columna a utilizar como objetivo para modelos de machine learning en las evaluaciones y separación cuando se deba estratificar los campos.

  : Variables de entrada para *Synthetic*
:::

En la se detallan los modelos actualmente soportados en la clase
*Synthetic* y su origen.

::: {#modelos-tab-soportados}
  Nombre Modelo    Fuente
  ---------------- -----------------------------------
  copulagan        SDV [@kotelnikov_overview_nodate]
  tvae             SDV [@kotelnikov_overview_nodate]
  gaussiancopula   SDV [@kotelnikov_overview_nodate]
  ctgan            SDV [@kotelnikov_overview_nodate]
  tablepreset      SDV [@kotelnikov_overview_nodate]
  smote-enc        tabDDPM [@akim_tabddpm_2023]
  tddpm\_mlp       tabDDPM [@akim_tabddpm_2023]

  : Modelos Tabulares Soportados
:::

Al ejecutar el script de generación de datos sintéticos, se crearán
múltiples archivos en una carpeta. En la se muestra un ejemplo de los
archivos generados y su formato. El nombre del modelo utilizado se
indica en el campo **\<model\>**, y en caso de haberse aplicado
*Differential Privacy* para generar una versión con ruido. El campo
**\<n\_sample\>** indica el número de registros sintéticos generados, y
finalmente el campo **\<type\_comparison\>** especifica si se trata de
una comparación entre los datos sintéticos y los datos de entrenamiento
(*Synthetic vs Train*, abreviado como ST) o entre los datos sintéticos y
los datos de validación (*Synthetic vs Hold*, abreviado como SH).
Adicionalmente se encuentran los archivos de esquema (*metadata.json*) y
una separación del dataset inicial en el conjunto de entrenamiento y
test (hold).

### Modelos para textos

Como se mencionó anteriormente, se utilizó el modelo **mT5** que se
entrenó para una nueva tarea utilizando la estrategia presentada en el
artículo *Text-to-Text Pre-Training for Data-to-Text Tasks*
[@kale_text--text_2020]. Para ilustrar el proceso, se presenta un
ejemplo del texto pre-procesado, el segmento de la pregunta y la
respuesta esperada para un registro del conjunto de datos
*economicos.cl*.

\<fecha, 2022-01-01\>\
\<precio, \$ 105.000.000\>\
\<tipo, Departamento\>\
\<transacción, Venta\>\
\<región, Metropolitana de Santiago\>\
\<comuna, Santiago\>\
\<dormitorios, 3.0\>\
\<baños, 3.0\>\
\<construidos, 47.0\>\
\<terreno, 47.0\>\
\<precio\_real, 3387.4540447373292\>\
\<titulo, Departamento en Venta en Santiago 3 dormitorios 1 baño\>\
\<dirección, DEPARTAMENTO EN EL CORAZON DE LO BARNECHEA Santiago,
Metropolitana de Santiago\>

descripción de esta publicación

Kazona Propiedades Vende Departamento de 47m2, 3 dormitorios, 1 baño,
cocina, living comedor , Paredes con Cerámica y Tabiquería en techo con
madera barnizada timbrada, ventanas nuevas de PVC y vidrio
termolaminado, sistema eléctrico actualizado, departamento ubicado en el
3er nivel (sin ascensor) , bajo gasto común. Excelentes conectividades y
ubicación en Pleno Centro De Lo Barnechea, como colegios privados y
públicos, supermercados, Mall Portal La Dehesa, locomoción, entre
otros.\
Podemos destacar de la propiedad:\
Pleno Centro Lo Barnechea\
100 metros de locomoción a Escuela Militar , Bilbao, Stgo Centro,
Mapocho\
200 metros colegios Montessori Nido de Águila, San Rafael , otros\
200 metros Mall Portal La Dehesa\
200 metros Sta. Isabel\
300 metros carabineros\
Gastos comunes bajos \$10.000\
Estacionamiento comunitario\
No paga contribuciones\
Contactanos al telefono Kazona 569 56031154

Obtención de Métricas
---------------------

Se han automatizado la mayoría de las métricas para evaluar los
conjuntos de datos sintéticos mediante el módulo *metrics*. Estas
métricas se aplican a los tres conjuntos de datos para su evaluación, lo
que permite calcular estadísticas y comparativas para el conjunto de
datos real utilizado para el entrenamiento (train dataset), el conjunto
de datos reservado para la evaluación (hold) y el conjunto de datos
sintético generado por los diferentes modelos (synthetic). Se pueden
recolectar ejecutando el ejemplo de código proporcionado en Código .

En la se muestra las metricas recolectadas para campos numericos.

::: {#metricas-numericas}
  Campo                                                                Ejemplos
  -------------------------------------------------------------------- ------------------------------------------------------------
  Campo                                                                Ejemplos
                                                                       
  Nombre del campo (name)                                              sqft\_living
  Valores del Top 5 (top5)                                             \[1400 1300 1720 1250 1540\]
  Frecuencia Top 5 (top5\_frec)                                        \[109 107 106 106 105\]
  Probabilidades de Top 5 (top5\_prob)                                 \[0.00630422 0.00618855 0.00613071 0.00613071 0.00607287\]
  Elementos observados (nobs)                                          17290
  Nulos (missing)                                                      0
  Promedio (mean)                                                      2073.894910
  Desviación Estándar (std)                                            907.297963
  Error estándar de la media (std\_err)                                6.900053
  Intervalo de confianza superior (upper\_ci)                          2087.418766
  Intervalo de confianza inferior (lower\_ci)                          2060.371055
  Rango intercuartílico (iqr)                                          1110
  Rango intercuartílico normalizado (iqr\_normal)                      822.844231
  Desviación absoluta de la mediana (mad)                              693.180169
  Desviación absoluta de la mediana normalizada (mad\_normal)          868.772506
  Coeficiente de variación (coef\_var)                                 0.437485
  Rango (range)                                                        11760
  Valor máximo (max)                                                   12050
  Valor mínimo (min)                                                   290
  Sesgo (skew)                                                         1.370859
  Curtosis (kurtosis)                                                  7.166622
  Test de normalidad de Jarque-Bera (jarque\_bera)                     17922.347382
  Valor p del test de normalidad de Jarque-Bera (jarque\_bera\_pval)   0
  Moda (mode)                                                          1400
  Frecuencia de la moda (mode\_freq)                                   0.006304
  Mediana (median)                                                     1910
  Percentil 0.1%                                                       522.890000
  Percentil 1%                                                         720
  Percentil 5%                                                         940
  Percentil 25%                                                        1430
  Percentil 75%                                                        2540
  Percentil 95%                                                        3740
  Percentil 99%                                                        4921.100000
  Percentil 99.9%                                                      6965.550000

  : Metricas para campos numericos
:::

En la se muestran los datos calculados para campos categóricos.

::: {#metricas-categoricas}
  Nombre del campo (name)                waterfront
  -------------------------------------- ---------------------------
  Valores del Top 5 (top5)               \[0 1\]
  Frecuencia Top 5 (top5\_freq)          \[17166 124\]
  Probabilidades de Top 5 (top5\_prob)   \[0.99282822 0.00717178\]
  Elementos observados (nobs)            17290.0
  Nulos (missing)                        17290.0

  : Métricas para campos categóricos
:::

En el Código , se muestra cómo se calcula y se muestra el Score promedio
para una selección específica de modelos. El código utiliza la función
\"sort\_values\" para ordenar los resultados en orden descendente según
el puntaje. Luego, se filtran los resultados para incluir solo los
modelos seleccionados y las columnas que muestran el puntaje y la
Distancia al registro más cercano (DCR) en los tres umbrales *Synthetic
vs Train* (ST), *Synthetic vs Hold* (SH) y *Train vs Hold* TH.

``` {.python .numberLines linenos="true" frame="lines" framesep="2mm" baselinestretch="1.2"}
avg = syn.scores[syn.scores["type"] == "avg"]
avg.sort_values("score", ascending=False).loc[ ["tddpm_mlp","smote-enc","gaussiancopula","tvae","gaussiancopula", "copulagan","ctgan"], ["score", "DCR ST 5th", "DCR SH 5th", "DCR TH 5th"]]
```

El Score calculado se obtiene a través de SDV y se basa en cuatro
métricas: KSComplement, TVComplement que conforman *Column Shapes*,
ContingencySimilarity y CorrelationSimilarity conforman *Column Pair
Trends*. Además, para mostrar los resultados, se proporciona un ejemplo
de código en el Código y un ejemplo de resultado en la .

::: {#tabla-show-score}
  Nombre          Column Pair Trends   Column Shapes     **Score** $\downarrow$   DCR ST   DCR SH   DCR TH
  --------------- -------------------- --------------- ------------------------ -------- -------- --------
  ntddpm\_mlp     0.954                0.971                              0.962    0.084    0.104    0.035
  nsmote-enc      0.941                0.967                              0.954    0.058    0.090    0.035
  **\<model\>**   0.941                0.967                              0.954    0.058    0.090    0.035

  : Ejemplo de scores promedios
:::

Resultados
==========

Este proyecto se ha centrado en la generación de datos sintéticos a
través de diversos métodos de preprocesamiento y modelos de aprendizaje
automático. Los resultados se examinan en base al rendimiento de los
modelos, los cuales fueron entrenados con los datos sintéticos, y se
valoran respecto a la similitud, privacidad y utilidad de los datos
generados.

Cabe mencionar que los resultados son inherentes a cada conjunto de
datos y modelo utilizado. Por lo tanto, se proporciona un análisis
exhaustivo de los resultados en cada escenario específico. Esto permite
una mejor comprensión de la eficacia de los métodos utilizados en la
generación de datos sintéticos y su comparación con los datos
originales.

A continuación, se presentan los conjuntos de datos de King County y
Económicos. El conjunto Económicos se subdivide en dos subconjuntos, que
difieren en su tratamiento de los datos nulos durante el
preprocesamiento.

King County
-----------

### Reportes

La Tabla [4.1](#table-score-king county-a){reference-type="ref"
reference="table-score-king county-a"} presenta los puntajes alcanzados
por los distintos patrones empleados en este estudio. Se observa que los
patrones con calificaciones superiores, tales como tddpm\_mlp y
smote-enc, exhiben una mayor correspondencia con el conjunto original de
datos. Por otro lado, aquellos patrones con calificaciones inferiores,
como ctgan, muestran una correspondencia significativamente reducida con
el conjunto original. Se proporciona el promedio ± desviación en base de
las 3 ejecuciones.

::: {#table-score-king county-a}
  Model Name         Column Pair Trends       Column Shapes            Coverage          Boundaries           **Score**
  ---------------- -------------------- ------------------- ------------------- ------------------- -------------------
  tddpm\_mlp          **0.94±3.80e-03**   **0.97±1.48e-03**   **0.97±4.96e-03**       1.00±0.00e+00       0.95±2.36e-03
  smote-enc               0.94±2.60e-04       0.96±3.06e-04       0.84±8.31e-03   **1.00±1.02e-05**   **0.95±2.45e-04**
  ctgan                   0.81±1.40e-02       0.84±2.67e-02       0.86±2.25e-03       1.00±0.00e+00       0.82±2.02e-02
  tablepreset             0.84±0.00e+00       0.84±1.36e-16       0.75±0.00e+00       1.00±0.00e+00       0.84±7.85e-17
  copulagan               0.76±4.93e-03       0.81±4.70e-03       0.84±1.74e-02       1.00±0.00e+00       0.79±2.92e-03
  gaussiancopula          0.76±0.00e+00       0.81±0.00e+00       0.75±7.85e-17       1.00±0.00e+00       0.79±0.00e+00
  tvae                    0.71±1.19e-02       0.77±1.22e-02       0.45±1.63e-02       1.00±0.00e+00       0.74±1.18e-02

  : Evaluación de Métricas de Rendimiento para Diversos Modelos de
  Aprendizaje Automático, King County
:::

Aunque los patrones TDDPM y SMOTE logran calificaciones prometedoras en
términos generales, existe una diferencia notable entre ambos en lo que
respecta a cobertura y límites. SMOTE no abarca la diversidad del
conjunto de datos, lo cual se manifiesta en su calificación de cobertura
*(Coverage)*, que es considerablemente inferior a la de TDDPM, así como
en su calificación de límites *(Boundaries)*.

#### Correlación pairwise

Este resultado se puede apreciar en el Anexo , donde se muestran las
diferencias entre los datos reales y los datos generados por cada
modelo. Se puede observar que, en general, los modelos con puntajes más
altos tienen una mayor similitud visual con los datos reales. Por
ejemplo, las imágenes y muestran la comparación entre los datos reales y
los datos generados por los modelos gaussiancopula y copulagan. A pesar
de que estos modelos tienen puntajes similares, el modelo gaussiancopula
tiene una mayor similitud visual con los datos reales que el modelo
copulagan.

Es importante destacar que, entre los modelos con puntajes superiores al
90%, puede ser difícil evaluar visualmente cuál es el mejor. Esto se
debe a que, a medida que el puntaje aumenta, la similitud visual entre
los datos reales y los datos generados también aumenta. Esto se puede
observar en las figuras y , donde se comparan los datos reales con los
datos generados por los modelos smote-enc y tddpm\_mlp, respectivamente.
Ambos modelos tienen puntajes superiores al 90%, y la similitud visual
entre los datos reales y los datos generados es muy alta en ambos casos.

En la evaluación de SDMetrics y en la comparación visual utilizando la
correlación de Wise, los mejores modelos encontrados son TDDPM y SMOTE.
Estos modelos han obtenido los puntajes más altos en ambas métricas y
también se han demostrado tener una mayor similitud visual con los datos
reales. Por lo tanto, se puede concluir que estos modelos son los más
efectivos para generar datos sintéticos útiles para este conjunto de
datos específico.

#### Revisión de Columnas

La muestra la superioridad del modelo TDDPM al cubrir los diferentes
valores en general, aunque existen casos en los que ambos modelos fallan
en cubrir los valores. Por ejemplo, en las columnas de *bathrooms* o
*bedrooms*, donde TDDPM solo sobrepasa el 70% de cobertura, pero aún así
es mejor que SMOTE. En cambio, SMOTE tiene algunos atributos que solo
alcanzan un 40% de cobertura.

::: {#table-coverage-king county-a}
  Columna          Metrica                          smote-enc              tddpm\_mlp
  ---------------- ------------------ ----------------------- -----------------------
  bathrooms        CategoryCoverage         6.56e-01±3.85e-02   **8.11e-01±3.85e-02**
  bedrooms         CategoryCoverage         5.13e-01±4.44e-02   **7.18e-01±4.44e-02**
  condition        CategoryCoverage     **9.33e-01±1.15e-01**       1.00e+00±0.00e+00
  date             CategoryCoverage     **9.64e-01±6.77e-03**       9.44e-01±9.69e-03
  floors           CategoryCoverage         8.33e-01±0.00e+00   **9.44e-01±9.62e-02**
  grade            CategoryCoverage         7.50e-01±0.00e+00   **8.61e-01±4.81e-02**
  id               RangeCoverage            9.93e-01±4.54e-04   **1.00e+00±7.79e-04**
  lat              RangeCoverage            9.65e-01±8.31e-03   **1.00e+00±0.00e+00**
  long             RangeCoverage            9.91e-01±5.54e-03   **1.00e+00±2.45e-04**
  price            RangeCoverage            5.72e-01±1.02e-01   **1.00e+00±1.25e-05**
  sqft\_above      RangeCoverage            7.88e-01±2.98e-02   **1.00e+00±1.45e-05**
  sqft\_basement   RangeCoverage            7.47e-01±2.02e-01   **1.00e+00±0.00e+00**
  sqft\_living     RangeCoverage            7.03e-01±4.89e-02   **1.00e+00±1.33e-05**
  sqft\_living15   RangeCoverage            8.49e-01±5.19e-02   **1.00e+00±5.14e-05**
  sqft\_lot        RangeCoverage            5.86e-01±7.02e-03   **1.00e+00±3.49e-06**
  sqft\_lot15      RangeCoverage        **8.30e-01±2.80e-01**       1.00e+00±4.72e-05
  view             CategoryCoverage     **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**
  waterfront       CategoryCoverage     **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**
  yr\_built        RangeCoverage        **1.00e+00±4.11e-05**       1.00e+00±0.00e+00
  yr\_renovated    RangeCoverage        **1.00e+00±9.76e-05**       1.00e+00±0.00e+00
  zipcode          CategoryCoverage     **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**

  : Evaluación de Cobertura Categorı́a-Rango para Modelos SMOTE-ENC y
  TDDPM\_MLP, King County
:::

En general, la distribución en ambos modelos es cercana a la real, en
casi todos los casos por encima del 90%. La única excepción es SMOTE en
*bathrooms*.

::: {#table-shape-king county-a}
  Columna          Metrica                      smote-enc              tddpm\_mlp
  ---------------- -------------- ----------------------- -----------------------
  bathrooms        TVComplement         8.84e-01±5.09e-03   **9.46e-01±6.18e-03**
  bedrooms         TVComplement         9.18e-01±7.87e-04   **9.50e-01±5.73e-03**
  condition        TVComplement         9.33e-01±1.23e-03   **9.61e-01±5.43e-03**
  date             TVComplement     **9.38e-01±1.73e-03**       9.26e-01±2.29e-03
  floors           TVComplement         9.66e-01±1.12e-03   **9.68e-01±4.38e-03**
  grade            TVComplement         9.58e-01±6.82e-04   **9.64e-01±1.19e-03**
  id               KSComplement     **9.86e-01±6.51e-04**       9.75e-01±2.95e-03
  lat              KSComplement     **9.89e-01±1.69e-03**       9.83e-01±8.10e-04
  long             KSComplement     **9.88e-01±2.22e-03**       9.78e-01±1.98e-03
  price            KSComplement     **9.81e-01±6.63e-04**       9.72e-01±7.86e-03
  sqft\_above      KSComplement         9.72e-01±1.42e-03   **9.77e-01±8.75e-03**
  sqft\_basement   KSComplement         9.35e-01±3.60e-03   **9.75e-01±3.87e-03**
  sqft\_living     KSComplement     **9.81e-01±2.50e-03**       9.73e-01±5.59e-03
  sqft\_living15   KSComplement     **9.81e-01±1.63e-03**       9.76e-01±4.34e-03
  sqft\_lot        KSComplement     **9.83e-01±4.81e-03**       9.58e-01±8.34e-03
  sqft\_lot15      KSComplement     **9.84e-01±3.16e-03**       9.62e-01±8.15e-03
  view             TVComplement         9.36e-01±9.73e-04   **9.52e-01±4.70e-03**
  waterfront       TVComplement         9.94e-01±1.22e-04   **9.95e-01±6.04e-04**
  yr\_built        KSComplement     **9.83e-01±4.71e-04**       9.76e-01±6.80e-03
  yr\_renovated    KSComplement     **9.92e-01±4.17e-04**       9.91e-01±1.00e-03
  zipcode          TVComplement     **9.74e-01±1.57e-03**       9.50e-01±4.11e-04

  : Evaluación de Similitud de Distribución para Modelos SMOTE-ENC y
  TDDPM\_MLP, King County
:::

En la revisión por columnas de los conjuntos de datos completos, como se
puede observar en la lista , se aprecia una similitud entre los tres
conjuntos analizados: Real, Smote y TDDPM. Sin embargo, existen
diferencias notables entre ellos. Cabe destacar que los datos generados
son alrededor de un 20% más grandes que el conjunto real.

En varias columnas, la distribución entre los tres conjuntos es similar,
como es el caso de bathrooms, sqft\_lot, sqft\_above, price,
sqft\_living, sqft\_basement, yr\_built, sqft\_living15 y grade. En
muestra un ejemplo de esto.

La distribución de los atributos bedrooms, condition, view y floors
contiene más elementos menos frecuentes en el conjunto de datos generado
por el modelo TDDPM. Por ejemplo, en se puede observar que en la columna
*bedrooms* la distribución de valores en el conjunto TDDPM es distinta a
SMOTE. Presenta más registros en valor 6 y 1.

En contraste, en el caso de la columna *sqft\_lot15*, el modelo SMOTE
tiene una distribución más cercana a la del conjunto real. Esto se puede
observar en .

#### Privacidad

En el análisis de los registros más cercanos entre los conjuntos reales
usados para entrenamiento, los generados por los modelos y el conjunto
real almacenado, se presentan sus distancias en la tabla.

::: {#table-dcr-king county-a}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       5.79e-02±6.08e-04             7.65e-02±1.23e-03   3.57e-02±0.00e+00   **9.52e-01±2.36e-03**          
  smote-enc        7.04e-03±2.77e-04             3.69e-02±6.21e-04   3.57e-02±0.00e+00       9.53e-01±2.45e-04          
  ctgan            2.15e-01±1.32e-02             2.38e-01±1.32e-02   3.57e-02±0.00e+00       8.24e-01±2.02e-02          
  tablepreset      1.80e-01±0.00e+00             2.00e-01±0.00e+00   3.57e-02±0.00e+00       8.37e-01±7.85e-17          
  copulagan        **3.75e-01±9.42e-03**     **4.12e-01±7.08e-03**   3.57e-02±0.00e+00       7.89e-01±2.92e-03          
  gaussiancopula   2.63e-01±0.00e+00             3.06e-01±0.00e+00   3.57e-02±0.00e+00       7.88e-01±0.00e+00          
  tvae             8.09e-02±3.59e-04             9.86e-02±5.62e-04   3.57e-02±0.00e+00       7.38e-01±1.18e-02          

  : Distancia de registros más cercanos, percentil 5, datos king county
:::

::: {#table-dcr-king county-a}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       1.34e-02±3.06e-03             1.99e-02±1.70e-03   0.00e+00±0.00e+00   **9.52e-01±2.36e-03**          
  smote-enc        0.00e+00±0.00e+00             1.24e-03±1.14e-04   0.00e+00±0.00e+00       9.53e-01±2.45e-04          
  ctgan            8.76e-02±2.88e-03             1.06e-01±1.12e-02   0.00e+00±0.00e+00       8.24e-01±2.02e-02          
  tablepreset      7.90e-02±0.00e+00             8.53e-02±1.39e-17   0.00e+00±0.00e+00       8.37e-01±7.85e-17          
  copulagan        **2.09e-01±2.71e-02**     **2.34e-01±3.43e-02**   0.00e+00±0.00e+00       7.89e-01±2.92e-03          
  gaussiancopula   7.88e-02±9.81e-18             1.27e-01±0.00e+00   0.00e+00±0.00e+00       7.88e-01±0.00e+00          
  tvae             3.26e-02±2.42e-03             3.48e-02±7.64e-03   0.00e+00±0.00e+00       7.38e-01±1.18e-02          

  : Distancia de registros más cercanos, minimo, datos king county
:::

::: {#table-dcr-king county-a}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       6.12e-01±2.28e-03             6.03e-01±4.17e-03   3.76e-01±0.00e+00   **9.52e-01±2.36e-03**          
  smote-enc        1.98e-01±3.57e-03             4.09e-01±6.38e-03   3.76e-01±0.00e+00       9.53e-01±2.45e-04          
  ctgan            8.09e-01±8.59e-03             8.15e-01±4.89e-03   3.76e-01±0.00e+00       8.24e-01±2.02e-02          
  tablepreset      8.25e-01±1.11e-16             8.18e-01±0.00e+00   3.76e-01±0.00e+00       8.37e-01±7.85e-17          
  copulagan        **8.30e-01±5.92e-03**     **8.24e-01±3.49e-03**   3.76e-01±0.00e+00       7.89e-01±2.92e-03          
  gaussiancopula   7.53e-01±1.11e-16             7.52e-01±1.36e-16   3.76e-01±0.00e+00       7.88e-01±0.00e+00          
  tvae             7.32e-01±5.74e-03             7.04e-01±4.22e-03   3.76e-01±0.00e+00       7.38e-01±1.18e-02          

  : Proporción entre el más cercano y el segundo más cercano, percentil
  5, datos king county
:::

::: {#table-dcr-king county-a}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       1.23e-01±1.54e-02             1.57e-01±3.36e-02   0.00e+00±0.00e+00   **9.52e-01±2.36e-03**          
  smote-enc        0.00e+00±0.00e+00             1.10e-02±5.41e-03   0.00e+00±0.00e+00       9.53e-01±2.45e-04          
  ctgan            4.25e-01±3.23e-02             3.91e-01±4.06e-02   0.00e+00±0.00e+00       8.24e-01±2.02e-02          
  tablepreset      4.51e-01±6.80e-17             3.58e-01±5.55e-17   0.00e+00±0.00e+00       8.37e-01±7.85e-17          
  copulagan        **5.48e-01±1.58e-02**     **5.32e-01±3.85e-02**   0.00e+00±0.00e+00       7.89e-01±2.92e-03          
  gaussiancopula   3.90e-01±3.93e-17             4.08e-01±0.00e+00   0.00e+00±0.00e+00       7.88e-01±0.00e+00          
  tvae             3.44e-01±1.91e-02             3.43e-01±1.63e-02   0.00e+00±0.00e+00       7.38e-01±1.18e-02          

  : Proporción entre el más cercano y el segundo más cercano, minimo,
  datos king county
:::

En la solo se consideran los modelos TDDPM y SMOTE para su comparación.
Se ve que en ambos casos existe una distancia mayor a cero, pero que en
el caso de TDDPM es mayor, por lo que se considera que es un mejor
conjunto en términos de privacidad.

Economicos
----------

El conjunto de economicos, a diferencia de kingcounty que fue filtrado y
preprocesado para evitar valores nulos. Este dataset economicos.cl
contiene nulos. A continuación se mostrará dos tipos de tratamientos de
los elementos nulos. El primero simplemente quita todos los registros
que contiene un registro vacio con dropna, se muestra en Código , se
considerará . El Código se considerará .

``` {.python .numberLines linenos="true" frame="lines" framesep="2mm" baselinestretch="1.2"}
df_converted = df.dropna().astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
basedate = pd.Timestamp('2017-12-01')
dtime = df_converted.pop("publication_date")
df_converted["publication_date"] = dtime.apply(lambda x: (x - basedate).days)
```

``` {.python .numberLines linenos="true" frame="lines" framesep="2mm" baselinestretch="1.2"}
df_converted = df.fillna(dict(
            property_type = "None",
            transaction_type = "None",
            state = "None",
            county = "None",
            rooms = -1,
            bathrooms = -1,
            m_built = -1,
            m_size = -1,
            source = "None"
    )).fillna(-1).astype({k: 'str' for k in ("description", "price", "title", "address", "owner",)})
basedate = pd.Timestamp('2017-12-01')
dtime = df_converted.pop("publication_date")
df_converted["publication_date"] = dtime.apply(lambda x: (x - basedate).days)
```

### Reportes - Conjunto A

**Conjunto A** [\[ds-conjunto-a\]]{#ds-conjunto-a label="ds-conjunto-a"}

Para el conjunto A, como la muestra los

::: {#table-score-economicos-a}
  Model Name            Column Pair Trends           Column Shapes                Coverage              Boundaries               **Score**
  ---------------- ----------------------- ----------------------- ----------------------- ----------------------- -----------------------
  tddpm\_mlp         **9.73e-01±2.21e-03**   **9.84e-01±3.63e-04**       7.91e-01±5.31e-02   **1.00e+00±0.00e+00**   **9.79e-01±1.27e-03**
  smote-enc              9.62e-01±1.52e-03       9.76e-01±4.01e-04       6.67e-01±2.79e-02   **1.00e+00±0.00e+00**       9.69e-01±6.71e-04
  copulagan              7.46e-01±3.30e-02       7.90e-01±2.63e-02       6.80e-01±2.57e-03   **1.00e+00±0.00e+00**       7.68e-01±2.96e-02
  ctgan                  7.44e-01±1.96e-02       6.53e-01±4.72e-02       6.75e-01±1.75e-03   **1.00e+00±0.00e+00**       6.98e-01±2.63e-02
  gaussiancopula         6.96e-01±0.00e+00       6.88e-01±0.00e+00       5.65e-01±0.00e+00   **1.00e+00±0.00e+00**       6.92e-01±0.00e+00
  tvae                   5.83e-01±1.02e-02       6.41e-01±4.66e-02   **8.59e-02±1.28e-02**   **1.00e+00±0.00e+00**       6.12e-01±2.50e-02

  : Evaluación de Métricas de Rendimiento para Diversos Modelos de
  Aprendizaje Automático, Economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       4.48e-09±2.32e-10             3.59e-08±2.38e-09   1.28e-08±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        3.15e-11±3.01e-12             4.22e-08±2.49e-09   1.28e-08±0.00e+00       9.69e-01±6.71e-04          
  copulagan        1.37e-06±1.76e-07             2.86e-06±3.82e-07   1.28e-08±0.00e+00       7.68e-01±2.96e-02          
  ctgan            **1.49e-05±5.01e-06**     **2.42e-05±9.67e-06**   1.28e-08±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   5.28e-06±0.00e+00             8.21e-06±0.00e+00   1.28e-08±0.00e+00       6.92e-01±0.00e+00          
  tvae             3.90e-07±1.08e-07             7.80e-07±2.49e-07   1.28e-08±0.00e+00       6.12e-01±2.50e-02          

  : Distancia de registros más cercanos, percentil 5, datos economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       1.46e-10±3.86e-12             1.44e-09±1.01e-10   0.00e+00±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        0.00e+00±0.00e+00             1.54e-09±5.32e-13   0.00e+00±0.00e+00       9.69e-01±6.71e-04          
  copulagan        1.97e-07±4.64e-08             4.53e-07±9.95e-08   0.00e+00±0.00e+00       7.68e-01±2.96e-02          
  ctgan            **3.18e-06±4.34e-07**     **5.23e-06±1.44e-06**   0.00e+00±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   7.84e-07±0.00e+00             1.75e-06±0.00e+00   0.00e+00±0.00e+00       6.92e-01±0.00e+00          
  tvae             1.48e-07±9.24e-08             2.35e-07±1.18e-07   0.00e+00±0.00e+00       6.12e-01±2.50e-02          

  : Distancia de registros más cercanos, percentil 1, datos economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00       9.69e-01±6.71e-04          
  copulagan        5.88e-09±2.05e-09             1.21e-08±3.19e-09   0.00e+00±0.00e+00       7.68e-01±2.96e-02          
  ctgan            **2.83e-08±3.88e-08**     **6.05e-08±2.56e-08**   0.00e+00±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   1.13e-08±0.00e+00             1.75e-08±0.00e+00   0.00e+00±0.00e+00       6.92e-01±0.00e+00          
  tvae             5.65e-09±3.07e-09             2.56e-08±3.04e-08   0.00e+00±0.00e+00       6.12e-01±2.50e-02          

  : Distancia de registros más cercanos, minimo, datos economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       6.88e-02±1.16e-03             9.85e-02±2.09e-03   1.31e-02±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        7.17e-04±1.27e-05             1.12e-01±3.32e-03   1.31e-02±0.00e+00       9.69e-01±6.71e-04          
  copulagan        2.74e-01±3.32e-02             3.03e-01±4.80e-02   1.31e-02±0.00e+00       7.68e-01±2.96e-02          
  ctgan            2.65e-01±1.35e-02             2.71e-01±5.93e-02   1.31e-02±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   2.93e-01±0.00e+00             2.76e-01±0.00e+00   1.31e-02±0.00e+00       6.92e-01±0.00e+00          
  tvae             **3.67e-01±6.93e-02**     **4.31e-01±1.08e-01**   1.31e-02±0.00e+00       6.12e-01±2.50e-02          

  : Proporción entre el más cercano y el segundo más cercano, percentil
  5, datos economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       3.00e-03±9.91e-05             1.04e-02±2.95e-04   0.00e+00±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        0.00e+00±0.00e+00             2.47e-03±2.53e-04   0.00e+00±0.00e+00       9.69e-01±6.71e-04          
  copulagan        1.37e-02±3.97e-03             1.31e-02±1.78e-03   0.00e+00±0.00e+00       7.68e-01±2.96e-02          
  ctgan            4.84e-02±1.75e-02             3.67e-02±7.08e-03   0.00e+00±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   2.67e-02±2.45e-18             2.95e-02±4.25e-18   0.00e+00±0.00e+00       6.92e-01±0.00e+00          
  tvae             **5.44e-02±4.44e-02**     **1.95e-01±6.70e-02**   0.00e+00±0.00e+00       6.12e-01±2.50e-02          

  : Proporción entre el más cercano y el segundo más cercano, percentil
  1, datos economicos
:::

::: {#table-dcr-economicos-a}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00   **9.79e-01±1.27e-03**          
  smote-enc        0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00       9.69e-01±6.71e-04          
  copulagan        1.22e-04±7.03e-05             1.84e-04±1.12e-04   0.00e+00±0.00e+00       7.68e-01±2.96e-02          
  ctgan            4.21e-04±2.19e-04             1.32e-03±1.54e-03   0.00e+00±0.00e+00       6.98e-01±2.63e-02          
  gaussiancopula   4.99e-05±0.00e+00             7.59e-06±8.47e-22   0.00e+00±0.00e+00       6.92e-01±0.00e+00          
  tvae             **8.11e-04±1.77e-04**     **7.24e-03±3.14e-03**   0.00e+00±0.00e+00       6.12e-01±2.50e-02          

  : Proporción entre el más cercano y el segundo más cercano, minimo,
  datos economicos
:::

::: {#table-coverage-economicos-a}
  Columna             Metrica                          smote-enc              tddpm\_mlp
  ------------------- ------------------ ----------------------- -----------------------
  \_price             RangeCoverage        **9.68e-01±5.48e-02**       9.66e-01±3.30e-02
  bathrooms           CategoryCoverage     **8.63e-01±3.40e-02**       6.76e-01±2.94e-02
  county              CategoryCoverage         5.97e-01±3.73e-03   **7.87e-01±2.27e-02**
  m\_built            RangeCoverage            5.52e-01±3.16e-01   **7.71e-01±3.97e-01**
  m\_size             RangeCoverage            1.79e-02±8.52e-03   **3.36e-01±4.53e-02**
  property\_type      CategoryCoverage         6.67e-01±5.56e-02   **9.07e-01±3.21e-02**
  publication\_date   RangeCoverage            9.70e-01±5.80e-03   **9.81e-01±2.86e-03**
  rooms               CategoryCoverage         7.40e-01±1.41e-02   **7.80e-01±6.45e-02**
  state               CategoryCoverage         7.92e-01±3.61e-02   **9.58e-01±3.61e-02**
  transaction\_type   CategoryCoverage         5.00e-01±0.00e+00   **7.50e-01±2.50e-01**

  : Evaluación de Cobertura Categorı́a-Rango para Modelos SMOTE-ENC y
  TDDPM\_MLP, Economicos
:::

::: {#table-shape-economicos-a}
  Columna             Metrica                      smote-enc              tddpm\_mlp
  ------------------- -------------- ----------------------- -----------------------
  \_price             KSComplement         9.90e-01±1.16e-03   **9.88e-01±3.17e-03**
  bathrooms           TVComplement     **9.96e-01±5.34e-04**       9.86e-01±4.99e-04
  county              TVComplement         9.20e-01±1.01e-03   **9.65e-01±2.54e-03**
  m\_built            KSComplement         9.87e-01±7.24e-04   **9.87e-01±1.32e-03**
  m\_size             KSComplement         9.74e-01±1.11e-03   **9.85e-01±8.91e-04**
  property\_type      TVComplement         9.68e-01±1.75e-03   **9.81e-01±2.30e-03**
  publication\_date   KSComplement         9.79e-01±2.43e-03   **9.86e-01±3.20e-03**
  rooms               TVComplement         9.78e-01±1.42e-03   **9.82e-01±3.04e-03**
  state               TVComplement         9.67e-01±3.85e-03   **9.84e-01±1.79e-04**
  transaction\_type   TVComplement     **9.99e-01±8.21e-04**       9.96e-01±2.73e-03

  : Evaluación de Similitud de Distribución para Modelos SMOTE-ENC y
  TDDPM\_MLP, Economicos
:::

### Reportes - Conjunto B {#ds-conjunto-b}

::: {#table-dcr-economicos-b}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       9.12e-15±1.09e-15             9.99e-15±8.14e-16   9.00e-17±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        9.19e-15±6.41e-16             1.17e-14±6.96e-16   9.00e-17±0.00e+00       9.43e-01±4.67e-04          
  copulagan        2.65e-16±1.60e-16             2.84e-16±1.73e-16   9.00e-17±0.00e+00       7.74e-01±2.02e-02          
  tvae             1.00e-09±1.74e-09             1.00e-09±1.74e-09   9.00e-17±0.00e+00       7.38e-01±1.48e-02          
  ctgan            **7.29e-09±8.52e-09**     **7.35e-09±8.45e-09**   9.00e-17±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   9.23e-13±0.00e+00             1.02e-12±0.00e+00   9.00e-17±0.00e+00       6.31e-01±0.00e+00          

  : Distancia de registros más cercanos, percentil 5, datos economicos
:::

::: {#table-dcr-economicos-b}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       4.63e-16±2.28e-17             4.16e-16±3.23e-17   0.00e+00±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        1.83e-16±9.41e-18             2.54e-16±1.73e-17   0.00e+00±0.00e+00       9.43e-01±4.67e-04          
  copulagan        9.00e-17±1.01e-26             9.00e-17±1.30e-26   0.00e+00±0.00e+00       7.74e-01±2.02e-02          
  tvae             2.41e-16±4.18e-16             2.38e-16±4.12e-16   0.00e+00±0.00e+00       7.38e-01±1.48e-02          
  ctgan            1.57e-16±2.72e-16             1.87e-16±3.25e-16   0.00e+00±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   **1.15e-15±0.00e+00**     **1.37e-15±0.00e+00**   0.00e+00±0.00e+00       6.31e-01±0.00e+00          

  : Distancia de registros más cercanos, percentil 1, datos economicos
:::

::: {#table-dcr-economicos-b}
  Modelo           DCR ST                                   DCR SH              DCR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00       9.43e-01±4.67e-04          
  copulagan        4.57e-19±3.77e-21         **5.21e-19±1.82e-22**   0.00e+00±0.00e+00       7.74e-01±2.02e-02          
  tvae             8.99e-20±0.00e+00             8.99e-20±0.00e+00   0.00e+00±0.00e+00       7.38e-01±1.48e-02          
  ctgan            8.99e-20±0.00e+00             8.99e-20±0.00e+00   0.00e+00±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   **5.23e-19±0.00e+00**         5.09e-19±0.00e+00   0.00e+00±0.00e+00       6.31e-01±0.00e+00          

  : Distancia de registros más cercanos, minimo, datos economicos
:::

::: {#table-dcr-economicos-b}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       **3.03e-01±4.42e-03**     **2.96e-01±1.27e-02**   1.15e-07±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        2.47e-01±3.63e-03             2.60e-01±6.24e-03   1.15e-07±0.00e+00       9.43e-01±4.67e-04          
  copulagan        1.07e-05±4.91e-06             2.27e-05±1.82e-05   1.15e-07±0.00e+00       7.74e-01±2.02e-02          
  tvae             4.28e-04±2.75e-04             4.49e-04±2.88e-04   1.15e-07±0.00e+00       7.38e-01±1.48e-02          
  ctgan            2.10e-03±7.18e-04             7.23e-03±1.01e-02   1.15e-07±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   1.52e-02±0.00e+00             1.38e-02±0.00e+00   1.15e-07±0.00e+00       6.31e-01±0.00e+00          

  : Proporción entre el más cercano y el segundo más cercano, percentil
  5, datos economicos
:::

::: {#table-dcr-economicos-b}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       **3.14e-02±4.92e-03**     **3.08e-02±3.94e-03**   0.00e+00±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        2.52e-03±1.07e-03             3.47e-03±2.68e-04   0.00e+00±0.00e+00       9.43e-01±4.67e-04          
  copulagan        5.33e-09±1.38e-09             1.15e-07±1.65e-07   0.00e+00±0.00e+00       7.74e-01±2.02e-02          
  tvae             3.02e-05±4.14e-05             3.04e-05±4.15e-05   0.00e+00±0.00e+00       7.38e-01±1.48e-02          
  ctgan            1.21e-04±1.18e-04             1.35e-04±1.66e-04   0.00e+00±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   6.43e-06±0.00e+00             6.43e-06±0.00e+00   0.00e+00±0.00e+00       6.31e-01±0.00e+00          

  : Proporción entre el más cercano y el segundo más cercano, percentil
  1, datos economicos
:::

::: {#table-dcr-economicos-b}
  Modelo           NNDR ST                                 NNDR SH             NNDR TH               **Score**          
  ---------------- ----------------------- ----------------------- ------------------- ----------------------- -- -- -- --
  tddpm\_mlp       0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00   **9.84e-01±1.85e-03**          
  smote-enc        0.00e+00±0.00e+00             0.00e+00±0.00e+00   0.00e+00±0.00e+00       9.43e-01±4.67e-04          
  copulagan        6.76e-13±2.95e-13             1.49e-12±5.43e-13   0.00e+00±0.00e+00       7.74e-01±2.02e-02          
  tvae             1.51e-12±1.68e-13             4.64e-12±1.76e-13   0.00e+00±0.00e+00       7.38e-01±1.48e-02          
  ctgan            **2.46e-12±1.48e-12**     **3.61e-12±2.05e-12**   0.00e+00±0.00e+00       7.34e-01±5.42e-03          
  gaussiancopula   5.50e-14±0.00e+00             1.81e-12±0.00e+00   0.00e+00±0.00e+00       6.31e-01±0.00e+00          

  : Proporción entre el más cercano y el segundo más cercano, minimo,
  datos economicos
:::

::: {#table-coverage-economicos-b}
  Columna             Metrica                          smote-enc              tddpm\_mlp
  ------------------- ------------------ ----------------------- -----------------------
  \_price             RangeCoverage        **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**
  bathrooms           CategoryCoverage     **7.59e-01±3.45e-02**       4.77e-01±3.59e-02
  county              CategoryCoverage         8.19e-01±9.11e-03   **8.66e-01±1.49e-02**
  m\_built            RangeCoverage            8.78e-02±1.49e-02   **1.00e+00±0.00e+00**
  m\_size             RangeCoverage            2.53e-01±2.92e-01   **1.00e+00±0.00e+00**
  property\_type      CategoryCoverage         7.28e-01±4.28e-02   **9.01e-01±5.66e-02**
  publication\_date   RangeCoverage        **9.66e-01±5.52e-02**       1.00e+00±0.00e+00
  rooms               CategoryCoverage         4.23e-01±2.28e-02   **4.93e-01±1.49e-02**
  state               CategoryCoverage     **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**
  transaction\_type   CategoryCoverage     **1.00e+00±0.00e+00**   **1.00e+00±0.00e+00**

  : Evaluación de Cobertura Categorı́a-Rango para Modelos SMOTE-ENC y
  TDDPM\_MLP, Economicos
:::

::: {#table-shape-economicos-b}
  Columna             Metrica                      smote-enc              tddpm\_mlp
  ------------------- -------------- ----------------------- -----------------------
  \_price             KSComplement         9.85e-01±1.94e-04   **9.93e-01±8.05e-04**
  bathrooms           TVComplement     **9.98e-01±3.13e-04**       9.95e-01±4.98e-04
  county              TVComplement         9.10e-01±5.37e-04   **9.84e-01±2.56e-03**
  m\_built            KSComplement         8.56e-01±1.32e-03   **9.91e-01±1.44e-03**
  m\_size             KSComplement         5.51e-01±8.46e-07   **9.90e-01±2.65e-03**
  property\_type      TVComplement         9.79e-01±7.12e-04   **9.89e-01±3.27e-03**
  publication\_date   KSComplement         9.66e-01±9.67e-05   **9.91e-01±5.41e-03**
  rooms               TVComplement         9.87e-01±9.57e-04   **9.95e-01±7.29e-04**
  state               TVComplement         9.78e-01±4.57e-04   **9.90e-01±1.06e-03**
  transaction\_type   TVComplement         9.94e-01±1.97e-04   **9.97e-01±1.53e-03**

  : Evaluación de Similitud de Distribución para Modelos SMOTE-ENC y
  TDDPM\_MLP, Economicos
:::

Conclusiones
============

Discusión
=========

Anexos
======

Código de entrenamiento de economicos
-------------------------------------

Lista completa de figura pairwise kingcounty {#A-pairwise-kingcounty-top2-a-1}
--------------------------------------------

Smote y TDDPM en KingCounty Graficas por Columnas
-------------------------------------------------

Tabla de comparación de Top5 KingCounty
---------------------------------------

Figuras de correlación Economicos - Conjunto A {#pairwise-full-a}
----------------------------------------------

Figuras de correlación Economicos - Conjunto B {#pairwise-full-a}
----------------------------------------------

### Conjunto A
