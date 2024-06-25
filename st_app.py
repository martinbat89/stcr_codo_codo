import streamlit as st
import pandas as pd
import numpy as np
import time

# Preprocesado y modelado
# ==============================================================================
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import max_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Latitud y longitud por barrio
lat_long_barrio = {'latitud': {'AGRONOMIA': -34.59331,
  'ALMAGRO': -34.6072645,
  'BALVANERA': -34.607742,
  'BARRACAS': -34.647335,
  'BELGRANO': -34.559736,
  'BOCA': -34.634533,
  'BOEDO': -34.62837588,
  'CABALLITO': -34.618299,
  'CHACARITA': -34.586453,
  'COGHLAN': -34.56118,
  'COLEGIALES': -34.5735565,
  'CONSTITUCION': -34.626049,
  'FLORES': -34.631799,
  'FLORESTA': -34.62923,
  'LINIERS': -34.64162,
  'MATADEROS': -34.657909,
  'MONSERRAT': -34.612553,
  'MONTE CASTRO': -34.619938,
  'NUEVA POMPEYA': -34.650126,
  'NUÑEZ': -34.548484,
  'PALERMO': -34.583308,
  'PARQUE AVELLANEDA': -34.6480265,
  'PARQUE CHACABUCO': -34.635863,
  'PARQUE CHAS': -34.585026,
  'PARQUE PATRICIOS': -34.636325,
  'PATERNAL': -34.600045,
  'PUERTO MADERO': -34.61189811,
  'RECOLETA': -34.593239,
  'RETIRO': -34.591533,
  'SAAVEDRA': -34.549439,
  'SAN CRISTOBAL': -34.623627,
  'SAN NICOLAS': -34.603668,
  'SAN TELMO': -34.620664,
  'VELEZ SARSFIELD': -34.632517,
  'VERSALLES': -34.631197,
  'VILLA CRESPO': -34.5987685,
  'VILLA DEL PARQUE': -34.603903,
  'VILLA DEVOTO': -34.60454,
  'VILLA GRAL. MITRE': -34.610095,
  'VILLA LUGANO': -34.675263,
  'VILLA LURO': -34.638019,
  'VILLA ORTUZAR': -34.580512,
  'VILLA PUEYRREDON': -34.583016,
  'VILLA REAL': -34.6190145,
  'VILLA RIACHUELO': -34.690079,
  'VILLA SANTA RITA': -34.616212,
  'VILLA SOLDATI': -34.66192455,
  'VILLA URQUIZA': -34.572254},
 'longitud': {'AGRONOMIA': -58.492907,
  'ALMAGRO': -58.420518,
  'BALVANERA': -58.404636,
  'BARRACAS': -58.381361,
  'BELGRANO': -58.455826,
  'BOCA': -58.362239,
  'BOEDO': -58.418227,
  'CABALLITO': -58.440134,
  'CHACARITA': -58.451847,
  'COGHLAN': -58.473886,
  'COLEGIALES': -58.449775,
  'CONSTITUCION': -58.382976,
  'FLORES': -58.460710000000006,
  'FLORESTA': -58.480823,
  'LINIERS': -58.522869,
  'MATADEROS': -58.50207,
  'MONSERRAT': -58.38049,
  'MONTE CASTRO': -58.506494,
  'NUEVA POMPEYA': -58.416487,
  'NUÑEZ': -58.465273,
  'PALERMO': -58.4239825,
  'PARQUE AVELLANEDA': -58.475662,
  'PARQUE CHACABUCO': -58.43815,
  'PARQUE CHAS': -58.478133,
  'PARQUE PATRICIOS': -58.402115,
  'PATERNAL': -58.467119,
  'PUERTO MADERO': -58.362888,
  'RECOLETA': -58.399526,
  'RETIRO': -58.37837,
  'SAAVEDRA': -58.483677,
  'SAN CRISTOBAL': -58.401106,
  'SAN NICOLAS': -58.380869,
  'SAN TELMO': -58.371795,
  'VELEZ SARSFIELD': -58.491507,
  'VERSALLES': -58.523033,
  'VILLA CRESPO': -58.43997811,
  'VILLA DEL PARQUE': -58.490997,
  'VILLA DEVOTO': -58.514020200000004,
  'VILLA GRAL. MITRE': -58.468278,
  'VILLA LUGANO': -58.4742678,
  'VILLA LURO': -58.5022875,
  'VILLA ORTUZAR': -58.4676405,
  'VILLA PUEYRREDON': -58.503482,
  'VILLA REAL': -58.526837,
  'VILLA RIACHUELO': -58.472598,
  'VILLA SANTA RITA': -58.481367,
  'VILLA SOLDATI': -58.447466,
  'VILLA URQUIZA': -58.486612}}

#Definimos diccionarios para barrios y franjas
dict_barrios = {'AGRONOMIA': 0, 'ALMAGRO': 1, 'BALVANERA': 2, 'BARRACAS': 3, 'BELGRANO': 4, 'BOCA': 5, 'BOEDO': 6, 'CABALLITO': 7, 'CHACARITA': 8, 'COGHLAN': 9, 'COLEGIALES': 10, 'CONSTITUCION': 11, 'FLORES': 12, 'FLORESTA': 13, 'LINIERS': 14, 'MATADEROS': 15, 'MONSERRAT': 16, 'MONTE CASTRO': 17, 'NUEVA POMPEYA': 18, 'NUÑEZ': 19, 'PALERMO': 20, 'PARQUE AVELLANEDA': 21, 'PARQUE CHACABUCO': 22, 'PARQUE CHAS': 23, 'PARQUE PATRICIOS': 24, 'PATERNAL': 25, 'PUERTO MADERO': 26, 'RECOLETA': 27, 'RETIRO': 28, 'SAAVEDRA': 29, 'SAN CRISTOBAL': 30, 'SAN NICOLAS': 31, 'SAN TELMO': 32, 'VELEZ SARSFIELD': 33, 'VERSALLES': 34, 'VILLA CRESPO': 35, 'VILLA DEL PARQUE': 36, 'VILLA DEVOTO': 37, 'VILLA GRAL. MITRE': 38, 'VILLA LUGANO': 39, 'VILLA LURO': 40, 'VILLA ORTUZAR': 41, 'VILLA PUEYRREDON': 42, 'VILLA REAL': 43, 'VILLA RIACHUELO': 44, 'VILLA SANTA RITA': 45, 'VILLA SOLDATI': 46, 'VILLA URQUIZA': 47}
dict_tipos = {'Crimenes No Violentos': 0, 'Crimenes Viales': 1, 'Crimenes Violentos': 2}


# MODELADO
# ==============================================================================

#Función de reversión de resultados
def rev_log(x):
    return np.exp(x) - 1


def entrena_predice():
    
    progress_text = "Procesando datos..."
    mi_bar = st.sidebar.progress(10, text=progress_text)
    
    #Lectura Data_
    df_crimenes_group = pd.read_csv('./data/delitos_mes_anio55.csv')

    #Normalizamos la cantidad de crimenes
    df_crimenes_group['cantidad_log'] = np.log(df_crimenes_group['cantidad_crimenes'] + 1)

    #Encod de variables categóricas
    le = LabelEncoder()
    df_crimenes_group['tipo_encode']= le.fit_transform(df_crimenes_group['tipo_gral'])

    progress_text = "Procesando datos..."
    mi_bar.progress(20, text=progress_text)
    time.sleep(1)

    le = LabelEncoder()
    df_crimenes_group['bario_encode']= le.fit_transform(df_crimenes_group['barrio'])

    le_name_mapping = dict(zip(le.classes_,le.transform(le.classes_)))

    #Empieza seccion modelo RFRegg
    rfr_regg = df_crimenes_group.copy()


    # División de los datos en train y test
    # ==============================================================================
    X_rfr = rfr_regg[['bario_encode', 'tipo_encode', 'franja_grupo']]
    Y_rfr = rfr_regg.cantidad_log.to_frame()

    progress_text = "División datos de entrenamiento..."
    mi_bar.progress(40, text=progress_text)
    time.sleep(1)

    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(
        X_rfr, #.reshape(-1,1) cuando necesite organizar los datos segun lo envio a la funcion train_test_split
        Y_rfr,
        train_size = muestra / 100.00, # test_size puede ser tambien, corresponde con 1 menos test_train
        random_state= 1234,
        shuffle=shuff
    )

    
    progress_text = "Preparando Pipeline..."
    mi_bar.progress(60, text=progress_text)
    time.sleep(1)

    # Definimos el pipeline
    #Mejores hiperparametros de random: 
    pipeline_rfregg = Pipeline([
        ('scaler', StandardScaler()),  # Normalización opcional de los datos
        ('pca', PCA(n_components=3)),  # Reducción de dimensionalidad con PCA
        ('regressor', RandomForestRegressor(random_state=123, 
                                            n_estimators=n_estimators, 
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            max_features=max_features,
                                            max_depth=max_depth,
                                            bootstrap=boostrap))  # Modelo de regresión de árbol de decisión
    ])

    progress_text = "Entrenando el modelo..."
    mi_bar.progress(70, text=progress_text)
    time.sleep(1)
                   
    #Aplanamos el array Y_train2
    Y_train2 = np.ravel(Y_train2)
    # Entrenamos el modelo utilizando el pipeline
    pipeline_rfregg.fit(X_train2, Y_train2)

    Y_pred2 = pipeline_rfregg.predict(X_test2)

    #Cargamos errores principales del modelo
    dict_errores_datos2 = {'Error cuadratico medio (MSE)' : [mean_squared_error(Y_test2, Y_pred2)],
    'Error absoluto medio (MAE)' : [mean_absolute_error(Y_test2, Y_pred2)],
    'Error máximo (M)' : [max_error(Y_test2, Y_pred2)],
    'Suma de residuos al cuadrado (RSS)' : [round(mean_squared_error(Y_test2, Y_pred2)*len(Y_pred2),2)],
    'Raiz cuadrada del error cuadratico medio (RMSE)' : [mean_squared_error(Y_test2, Y_pred2,squared=False)],
    'Exactitud' : [pipeline_rfregg.score(X_test2, Y_test2)]}
    df_errores_datos2 = pd.DataFrame.from_dict(dict_errores_datos2, orient='index',columns=['Valor'])
    exactitud = f'{round(pipeline_rfregg.score(X_test2, Y_test2)*100)} %'      

    progress_text = "Predicciones..."
    mi_bar.progress(80, text=progress_text)
    time.sleep(1)

    #Definimos diccionarios para barrios y franjas
    dict_barrios = {'AGRONOMIA': 0, 'ALMAGRO': 1, 'BALVANERA': 2, 'BARRACAS': 3, 'BELGRANO': 4, 'BOCA': 5, 'BOEDO': 6, 'CABALLITO': 7, 'CHACARITA': 8, 'COGHLAN': 9, 'COLEGIALES': 10, 'CONSTITUCION': 11, 'FLORES': 12, 'FLORESTA': 13, 'LINIERS': 14, 'MATADEROS': 15, 'MONSERRAT': 16, 'MONTE CASTRO': 17, 'NUEVA POMPEYA': 18, 'NUÑEZ': 19, 'PALERMO': 20, 'PARQUE AVELLANEDA': 21, 'PARQUE CHACABUCO': 22, 'PARQUE CHAS': 23, 'PARQUE PATRICIOS': 24, 'PATERNAL': 25, 'PUERTO MADERO': 26, 'RECOLETA': 27, 'RETIRO': 28, 'SAAVEDRA': 29, 'SAN CRISTOBAL': 30, 'SAN NICOLAS': 31, 'SAN TELMO': 32, 'VELEZ SARSFIELD': 33, 'VERSALLES': 34, 'VILLA CRESPO': 35, 'VILLA DEL PARQUE': 36, 'VILLA DEVOTO': 37, 'VILLA GRAL. MITRE': 38, 'VILLA LUGANO': 39, 'VILLA LURO': 40, 'VILLA ORTUZAR': 41, 'VILLA PUEYRREDON': 42, 'VILLA REAL': 43, 'VILLA RIACHUELO': 44, 'VILLA SANTA RITA': 45, 'VILLA SOLDATI': 46, 'VILLA URQUIZA': 47}
    dict_tipos = {'Crimenes No Violentos': 0, 'Crimenes Viales': 1, 'Crimenes Violentos': 2}

    barrio_list = []
    tipo_list = []
    franja_list = []
    cantidad_list = []
    lat = []
    long = []

    

    for barrio in dict_barrios.keys():
        for tipo in dict_tipos.keys():
            for franja in range(1, 5):
                df_features = pd.DataFrame([[dict_barrios[barrio], dict_tipos[tipo], franja]], columns=['bario_encode', 'tipo_encode', 'franja_grupo'])
                pred_puntual = rev_log(pipeline_rfregg.predict(df_features))
                barrio_list.append(barrio)
                tipo_list.append(tipo)
                franja_list.append(franja)
                cantidad_list.append(round(pred_puntual[0]))
                lat.append(lat_long_barrio['latitud'][barrio])
                long.append(lat_long_barrio['longitud'][barrio])

    dict_estimaicion = {
                    'barrio': barrio_list,
                    'tipo': tipo_list,
                    'franja': franja_list,
                    'cantidad': cantidad_list,
                    'lat': lat,
                    'lon': long
                }

    df_estimacion = pd.DataFrame(dict_estimaicion)
    #df para el mapa
    df_map_barrios = df_estimacion.groupby(['lat', 'lon']).agg({'cantidad': 'sum'}).reset_index()
    #df para hbar chart por tipo y franja
    df_franja_tipo = df_estimacion.groupby(['franja', 'tipo']).agg({'cantidad': 'sum'}).reset_index()
    df_por_tipo = df_estimacion.groupby(['tipo']).agg({'cantidad': 'sum'}).reset_index()
    
    
    progress_text = "Visualizaciones generales..."
    mi_bar.progress(90, text=progress_text)
    time.sleep(1)
    # Mostrar algunos datos de estimación de forma global
    st.header('Predicción global de crímenes')
    st.caption('La cantidad de crímenes se expresan en términos mensuales.')
    st.subheader('Resultados del modelo')
    cola_1, cola_2 = st.columns(2)
    cola_1.metric(label="Exactitud Modelo", value=f"{exactitud}")
    cola_2.metric(label="Crímenes Estimados", value=f"{df_estimacion['cantidad'].sum()}")
    st.write(df_errores_datos2)
    st.subheader('Mapa de Barrios Mas Peligrosos')
    st.caption('Un marcador mas grande indica mayor peligro.')
    st.map(df_map_barrios, size='cantidad')
    st.subheader('Resumen por tipo y franja')
    st.caption('⏰1: 0 a 6 ⏰2: 6 a 12 ⏰3: 12 a 18 ⏰4: 18 a 24' )
    st.bar_chart(data=df_franja_tipo, x='franja', y="cantidad", color="tipo", horizontal=True)
    st.subheader('Resumen por tipo')
    st.bar_chart(data=df_por_tipo, x='tipo', y="cantidad", color="tipo")

    st.subheader('Detalle por barrio, tipo y franja horaria')
    df_detalle = df_estimacion.copy()
    st.dataframe(df_detalle, column_config={'lat': None, 
                                            'lon': None, 
                                            'cantidad': st.column_config.NumberColumn(
                                                "Crímenes Mensuales",
                                                help= "Cantidad de crímenes mensuales",
                                                format="%d 🕵️"
                                            ),
                                            'barrio': 'Barrio',
                                            'tipo': 'Tipo',
                                            'franja': 'Franja'})
    st.write('__________________________________________')
    
    progress_text = "Visualizaciones barrio puntual..."
    mi_bar.progress(95, text=progress_text)
    time.sleep(1)
    #Mostrar solo para el barrio puntual
    #primero copiamos el df
    df_barrio_punt = df_estimacion[df_estimacion['barrio'] == barrio_punt]
    df_barrio_punt_tipos = df_barrio_punt.groupby(['tipo']).agg({'cantidad': 'sum'}).reset_index()
    st.header(f'Predicción para el barrio {barrio_punt}')
    st.caption('La cantidad de crímenes se expresan en términos mensuales.')
    st.subheader('Resultados del modelo')
    cola_1, cola_2 = st.columns(2)
    cola_1.metric(label="Exactitud Modelo", value=f"{exactitud}")
    cola_2.metric(label="Crímenes Estimados", value=f"{df_barrio_punt['cantidad'].sum()}")
    st.subheader(f'Resumen por tipo y franja para el barrio {barrio_punt}')
    st.caption('Franjas ➡️ ⏰1: 0 a 6 ⏰2: 6 a 12 ⏰3: 12 a 18 ⏰4: 18 a 24' )
    st.bar_chart(data=df_barrio_punt, x='franja', y="cantidad", color="tipo", horizontal=True)
    st.subheader(f'Resumen por tipo para el barrio {barrio_punt}')
    st.bar_chart(data=df_barrio_punt_tipos, x='tipo', y="cantidad", color="tipo")

    st.subheader(f'Detalle tipo y franja horaria para el barrio {barrio_punt}')
    st.caption('Franjas ➡️ ⏰1: 0 a 6 ⏰2: 6 a 12 ⏰3: 12 a 18 ⏰4: 18 a 24' )
    st.dataframe(df_barrio_punt, column_config={'lat': None, 
                                                'lon': None, 
                                                'cantidad': st.column_config.NumberColumn(
                                                    "Crímenes Mensuales",
                                                    help= "Cantidad de crímenes mensuales",
                                                    format="%d 🕵️"
                                                ),
                                                'barrio': 'Barrio',
                                                'tipo': 'Tipo',
                                                'franja': 'Franja'})


    mi_bar.progress(100, text=progress_text)
    time.sleep(1)
    mi_bar.empty()
    st.toast('Modelo entrenado. Predicción realizada!', icon="✅")
    container1.empty()

# APP STREAMLIT
# ==============================================================================
container1 = st.container(border=True)
container1.title("Predicción Crimenes En CABA 🤖")
container1.caption('La aplicación permite evaluar el funcionamiendo de un modelo de ML de tipo Random Forest Regressor para estimar la cantidad de crímenes MENSUALES en CABA por barrio, tipo y franja horaria.')
container1.write("Para realizar predicciones seguir los siguientes pasos:")
container1.write('1. Define el porcentaje de datos de entrenamiento.')   
container1.write('2. Definie los hiperparámetros del modelo.')
container1.write('3. La predicción permite analizar un barrio puntual. Selecciona el barrio de tu interés.')
container1.write('4. Dale click al boton predicción y esperá el resultado.')
container1.info('Recordá que el entrenamiento y predicción demora al menos un minuto.', icon="ℹ")
ayuda_exp = container1.expander('Mas información sobre el modelo y sus parámetros.')
# Añadir contenido al expander
with ayuda_exp:
    st.write("""
    ### Modelo de Predicción de Crímenes
    Este modelo utiliza un `RandomForestRegressor` dentro de un `Pipeline` para predecir la cantidad de crímenes en diferentes barrios de la ciudad, por tipo y franja horaria. A continuación, se detallan los componentes y parámetros del modelo:

    **1. Normalización de Datos**
    - `StandardScaler`: Se utiliza para normalizar los datos, asegurando que cada característica tenga una media de 0 y una desviación estándar de 1.

    **2. Reducción de Dimensionalidad**
    - `PCA (Análisis de Componentes Principales)`: Se reduce la dimensionalidad de los datos a 3 componentes principales, lo cual puede mejorar la eficiencia del modelo y reducir el ruido.

    **3. Modelo de Regresión**
    - `RandomForestRegressor`: Este es un modelo de aprendizaje supervisado que utiliza múltiples árboles de decisión para hacer predicciones precisas. Los parámetros clave son:
      - `n_estimators=510`: Número de árboles en el bosque.
      - `min_samples_split=10`: Número mínimo de muestras requeridas para dividir un nodo.
      - `min_samples_leaf=2`: Número mínimo de muestras que debe tener un nodo hoja.
      - `max_features="sqrt"`: Número de características a considerar al buscar la mejor división.
      - `max_depth=30`: Profundidad máxima de los árboles.
      - `bootstrap=True`: Si se utilizan muestras con reemplazo al construir los árboles.

    Este enfoque permite capturar relaciones complejas en los datos y hacer predicciones robustas sobre la cantidad de crímenes en distintas áreas de la ciudad.
    """)
st.sidebar.header('Definí tus parámetros')
st.sidebar.caption('💡Los parámetros que aparecen por defecto optimizan el rendimiento del modelo.')
st.sidebar.subheader('Datos de entrenamiento')
muestra = st.sidebar.slider(
"Porcentaje Datos Entrenamiento",
30.0, 100.0, 70.0, 1.0)


on = st.sidebar.toggle("Shuffle (Barajar Datos)", value=True)
if on:
    shuff = True
else:
    shuff = False


st.sidebar.subheader('Definir Hiperparámetros')

n_estimators = st.sidebar.number_input("n_estimators", 1, 550, 510)
min_samples_split = st.sidebar.number_input("min_samples_split", 1, 15, 10)
min_samples_leaf= st.sidebar.number_input("min_samples_leaf", 1, 15, 2)
max_features = st.sidebar.selectbox("max_features",( 'sqrt',None, 'log2'))

max_depth=st.sidebar.number_input("max_depth", 0, 50, 30)

bootsrapinp = st.sidebar.toggle("Boostrap", value=True)

if bootsrapinp:
    boostrap = True
else:
    boostrap = False

nombres_barrios = tuple(dict_barrios.keys())
st.sidebar.subheader('Barrio para estimación puntual')
st.sidebar.write("Selecciona un barrio de CABA para la estimación por barrio puntual.")
barrio_punt = st.sidebar.selectbox("Barrio Puntual", nombres_barrios)

if st.sidebar.button("Predicción", 
            type="primary", 
            help="Presioná el boton para entrenar el modelo y predecir."
            ):
    entrena_predice()

st.sidebar.info('Recordá que el entrenamiento y predicción demora al menos un minuto.', icon="ℹ️")
st.sidebar.write('_____________________________________')
st.sidebar.caption('👨‍🎓👩‍🎓 Proyecto realizado durante el curso Codo a Codo 4.0 - Fundamentos de la Ciencia de Datos - 2024 - Equipo D.')
   

