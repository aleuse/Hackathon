# Hackathon

## Dataset
Se escogió el dataset: students' dropout and academic success.
Este dataset contiene datos de instituciones de educacion superior relacionada con alumnos matriculados en pregrados; en la información hay variables demográficas, sociales, económicas y del rendimiento de los estudiantes.Esto para el final del primer y del segundo semestre de cada uno.

## Objetivo
Identificar a los estudiantes que tienen un mayor riesgo de abandono. Es decir, es un problema de clasificación entre 3 clases: dropout, enrolled y graduated.

## Herramientas utilizadas
Plotly, Plotly.express, cufflinks, ydata_profiling, Numpy, Pandas, Matplotlib, Seaborn, Plotly, Scikit.Learn, Pycaret, MLFLow, FastAPI, Gradio.

## Abordando el reto
La primera forma de abordarlo fue a través de un clustering, es decir, agrupar a los estudiantes en función de sus características. Los datos se normalizaron con minmax para que quedaran entre 0 y 1 y no realizamos imputaciones, pues las variables que contenían ceros eran variables numéricas relacionadas con los créditos que tomaban los estudiantes, por lo que esta información no era nula necesariamente.
Se entrenaron modelos de clustering de KMeans, Hierarchical Clustering y DBSCAN inicialmente, luego se entrenaron dos modelos de KMeans con 3 y 4 clusters respectivamente. El desempeño fue pobre respecto al coeficiente de silueta, no superando el 0.06. Nos quedamos con el KMeans de 4 clusters y al realizar las predicciones nos encontramos con que uno de los grupos estaba formado enteramente de estudiantes "enrolled", otro grupo era de estudiantes "graduated" y los otros dos grupos estaban compensados entre estudiantes "graduate" y "dropout".

Después procedimos con la metodología de clasificación.
Tampoco utilizamos imputación por lo mencionado anteriormente. Normalizamos con minmax y utilizamos SMOTE para el desbalanceo de los datos, además de un tamaño de train de 0.8

Se compararon 15 modelos y el mejor de ellos fue Light Gradient Boosting Machine con 
- Accuracy: 0.7861												
- AUC: 0.9086
- Recall: 0.7861
- Prec: 0.7802
- F1: 0.7805
- Kappa: 0.6447
- MCC: 0.6475

El accuracy, que es el porcentaje de predicciones correctas. No nos guiamos de él pues estamos trabajando con datos desbalanceados.
AUC, área bajo la curva, que nos dice que la relación entre la tasa de verdaderos positivos y de falsos positivos, este valor fue muy alto lo que indica que el modelo es bueno clasificando los verdaderos positivos.
El recall o sensibilidad, que mide la proporción de ejemplos positivos reales que el modelo predijo correctamente.
La precisión, que nos dice la tasa de predicciones positivas que son verdaderamente positivos.
El F1 es muy importante en este caso, ya que trabajamos con datos desbalanceados. De esta manera, también lo es el MCC
La variable objetivo quedó: Dropout: 0, Enrolled: 1, Graduate: 2

Evaluando el modelo vimos que el modelo tenía una buena precisión y sensibilidad a la hora de clasificar a los "graduate" de 0.81 y 0.87
En cuanto al grupo que más nos interesa, los que tienen riesgo de abandono, para estos la precisión y sensibilidad fue: 0.77 y 0.70
