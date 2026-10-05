# Estimador del puntaje global de Saber Pro

Proyecto integrador de Ciencia de Datos (metodología CRISP-DM). Modelo de **regresión supervisada** que estima el
**puntaje global de Saber Pro** (suma de los 5 módulos genéricos, de 0 a 1.500) a partir del **perfil del estudiante**:
familia, situación económica, programa e institución. No usa ningún resultado de la prueba.

**Aplicación en línea:** (pegar aquí la URL pública de Streamlit)

## Datos

- Fuente: [Resultados únicos Saber Pro - ICFES, Datos Abiertos Colombia](https://www.datos.gov.co/resource/u37r-hjmu.json) (id `u37r-hjmu`).
- 1.217.482 filas originales -> 1.146.476 tras la limpieza -> muestra de trabajo de 24.000 filas para modelar
  (y muestra ampliada de 120.000 para la curva de aprendizaje y la validación en el tiempo). Años 2018 a 2022.
- Variable objetivo: `puntaje_global` = suma de los puntajes de razonamiento cuantitativo, lectura crítica, competencias ciudadanas,
  comunicación escrita e inglés.

## Modelo

- Algoritmo final: **XGBoost** dentro de un `Pipeline` de scikit-learn (ordinal, one-hot y target encoding + XGBRegressor).
- Hiperparámetros: {'n_estimators': 537, 'learning_rate': 0.0221, 'max_depth': 6, 'min_child_weight': 51, 'subsample': 0.7172, 'colsample_bytree': 0.6206, 'reg_lambda': 17.5597}
- Ajuste con `RandomizedSearchCV` (validación cruzada de 5 particiones sobre el 70 % de entrenamiento), métrica RMSE.
- Resultados en el 30 % de prueba: **MAE 71.9, RMSE 91.1 puntos, R2 0.408** (línea base que predice el promedio: RMSE 118.4).

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación de Streamlit |
| `modelo_saberpro.joblib` | Pipeline completo serializado |
| `opciones_app.json` | Listas de opciones y metadatos para la aplicación |
| `requirements.txt` | Dependencias con versiones fijas |
| `datos_nuevos_ejemplo.csv` | Registros de ejemplo para probar el modelo |

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones

- El perfil de contexto explica alrededor del 41 % de la varianza del puntaje; el resto depende de factores que el modelo no ve.
  El error típico (RMSE) es de unos 91 puntos.
- El modelo refleja desigualdades de la población (estrato, institución). Es descriptivo y **no debe usarse para decidir sobre personas**.
- Es menos preciso en estratos altos, en grupos pequeños (p. ej. instituciones tecnológicas) y comprime las predicciones hacia el promedio.
- Entrenado con datos de 2018 a 2022; para años posteriores el error esperado es mayor (~5 %).
- Al usar el modelo con datos propios, las columnas categóricas con valores faltantes deben ir como `NaN` dentro de columnas de tipo `object` (texto).
  Si una columna entera queda como `float`, el codificador de variables categóricas falla. La aplicación de Streamlit siempre envía valores completos.
