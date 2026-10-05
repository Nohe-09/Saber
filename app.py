"""Aplicación de Streamlit: estimación del puntaje global de Saber Pro a partir del perfil del estudiante.

El puntaje global es la suma de los cinco módulos genéricos (razonamiento cuantitativo, lectura crítica,
competencias ciudadanas, comunicación escrita e inglés), así que va de 0 a 1.500.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Estimador de puntaje Saber Pro", page_icon="🎓", layout="wide")

BASE = Path(__file__).parent


@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo_saberpro.joblib")


@st.cache_data
def cargar_opciones():
    with open(BASE / "opciones_app.json", encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
op = cargar_opciones()


def miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def predecir(perfil: dict) -> float:
    return float(modelo.predict(pd.DataFrame([perfil]))[0])


st.title("🎓 Estimador del puntaje global de Saber Pro")
st.write(
    "Este modelo estima el **puntaje global** (suma de los 5 módulos, de 0 a 1.500) a partir del **contexto** "
    "del estudiante: familia, situación económica, programa e institución. **No** usa ningún resultado de la prueba. "
    "Se entrenó con resultados oficiales del ICFES publicados en "
    "[Datos Abiertos Colombia](https://www.datos.gov.co/resource/u37r-hjmu.json)."
)
st.info(
    "Es un modelo estadístico sobre la población de evaluados, no una predicción sobre una persona. "
    "El contexto explica solo una parte del resultado (ver la sección de limitaciones)."
)

with st.sidebar:
    st.header("Perfil del estudiante")
    st.subheader("Persona")
    genero = st.selectbox("Género", op["generos"])
    edad = st.slider("Edad al presentar la prueba", 17, 60, 23)
    depto = st.selectbox("Departamento de residencia", op["deptos"], index=op["deptos"].index(op["depto_defecto"]))

    st.subheader("Familia y hogar")
    estrato = st.selectbox("Estrato de la vivienda", op["estrato"], index=3)
    edu_madre = st.selectbox("Nivel educativo de la madre", op["educacion"], index=4)
    edu_padre = st.selectbox("Nivel educativo del padre", op["educacion"], index=4)
    internet = st.radio("¿El hogar tiene internet?", ["Si", "No"], horizontal=True)
    computador = st.radio("¿El hogar tiene computador?", ["Si", "No"], horizontal=True)
    lavadora = st.radio("¿El hogar tiene lavadora?", ["Si", "No"], horizontal=True)
    automovil = st.radio("¿El hogar tiene automóvil?", ["No", "Si"], horizontal=True)

    st.subheader("Situación económica")
    horas = st.selectbox("Horas de trabajo a la semana", op["horas"], index=0)
    matricula = st.selectbox("Valor de la matrícula", op["matricula"], index=3)
    st.caption("¿Cómo paga la matrícula?")
    pago_beca = st.checkbox("Con beca", value=False)
    pago_credito = st.checkbox("Con crédito", value=False)
    pago_padres = st.checkbox("Con ayuda de los padres", value=True)
    pago_propio = st.checkbox("Con recursos propios", value=False)

    st.subheader("Programa e institución")
    institucion = st.selectbox("Institución", op["instituciones"], index=op["instituciones"].index(op["inst_defecto"]))
    programa = st.selectbox("Programa académico", op["programas"], index=op["programas"].index(op["prog_defecto"]))
    metodo = st.selectbox("Metodología del programa", op["metodos"])

# la institución y el programa determinan estas variables: se completan automáticamente
info_inst = op["inst_info"][institucion]
nucleo = op["prog_nucleo"][programa]

perfil = {
    "FAMI_ESTRATOVIVIENDA": estrato,
    "ESTU_HORASSEMANATRABAJA": horas,
    "ESTU_VALORMATRICULAUNIVERSIDAD": matricula,
    "FAMI_EDUCACIONPADRE": edu_padre,
    "FAMI_EDUCACIONMADRE": edu_madre,
    "ESTU_GENERO": genero,
    "INST_CARACTER_ACADEMICO": info_inst["caracter"],
    "ESTU_METODO_PRGM": metodo,
    "INST_ORIGEN": info_inst["origen"],
    "ESTU_PAGOMATRICULABECA": "Si" if pago_beca else "No",
    "ESTU_PAGOMATRICULACREDITO": "Si" if pago_credito else "No",
    "ESTU_PAGOMATRICULAPADRES": "Si" if pago_padres else "No",
    "ESTU_PAGOMATRICULAPROPIO": "Si" if pago_propio else "No",
    "FAMI_TIENEAUTOMOVIL": automovil,
    "FAMI_TIENELAVADORA": lavadora,
    "FAMI_TIENECOMPUTADOR": computador,
    "FAMI_TIENEINTERNET": internet,
    "INST_NOMBRE_INSTITUCION": institucion,
    "ESTU_PRGM_ACADEMICO": programa,
    "ESTU_NUCLEO_PREGRADO": nucleo,
    "ESTU_DEPTO_RESIDE": depto,
    "EDAD": float(edad),
}

pred = predecir(perfil)
prom = op["promedio_global"]
rmse = op["rmse_prueba"]

col1, col2, col3 = st.columns(3)
col1.metric("Puntaje global estimado", f"{pred:,.0f}".replace(",", "."), f"{pred - prom:+.0f} frente al promedio", delta_color="off")
col2.metric("Rango típico (± 1 RMSE)", f"{pred - rmse:,.0f} – {pred + rmse:,.0f}".replace(",", "."))
col3.metric("Promedio de la población", f"{prom:,.0f}".replace(",", "."))
st.caption(
    f"El error típico del modelo (RMSE) en datos de prueba es de unos {rmse:.0f} puntos; es decir, alrededor de dos de cada tres "
    "estudiantes reales con este perfil quedarían dentro del rango mostrado, y el resto fuera. "
    f"Se completó automáticamente: carácter académico = {info_inst['caracter'].title()}, origen = {info_inst['origen'].title()}, "
    f"núcleo de conocimiento = {nucleo.title()}."
)

st.progress(min(max(pred / 1500, 0.0), 1.0), text="Posición en la escala de 0 a 1.500")

tab1, tab2, tab3 = st.tabs(["Sensibilidad", "Cómo se calcula", "Limitaciones"])

with tab1:
    st.write("¿Cómo cambia la estimación si se modifica **una sola** característica y se dejan las demás igual? "
             "Es un análisis descriptivo del modelo, no una relación de causa y efecto.")
    st.caption("Ojo: la institución pesa mucho más que cualquier otra característica y, mientras se mantiene fija, los efectos de las demás "
               "son pequeños y pueden no ser ordenados (por ejemplo, subir de estrato no siempre sube la estimación). Esto pasa porque el "
               "estrato, la educación de los padres y la institución están muy relacionados entre sí: una parte de lo que se asocia con el "
               "estrato ya está contenida en la institución elegida.")
    variable = st.selectbox(
        "Característica a variar",
        ["Estrato de la vivienda", "Nivel educativo de la madre", "Horas de trabajo a la semana", "Valor de la matrícula", "Edad"],
    )
    claves = {"Estrato de la vivienda": ("FAMI_ESTRATOVIVIENDA", op["estrato"]),
              "Nivel educativo de la madre": ("FAMI_EDUCACIONMADRE", op["educacion"]),
              "Horas de trabajo a la semana": ("ESTU_HORASSEMANATRABAJA", op["horas"]),
              "Valor de la matrícula": ("ESTU_VALORMATRICULAUNIVERSIDAD", op["matricula"]),
              "Edad": ("EDAD", [float(e) for e in range(18, 46)])}
    clave, valores = claves[variable]
    filas = pd.DataFrame([{**perfil, clave: v} for v in valores])
    resultado = pd.DataFrame({variable: [str(int(v)) if isinstance(v, float) else v for v in valores],
                              "Puntaje estimado": modelo.predict(filas).round(0)})
    st.bar_chart(resultado, x=variable, y="Puntaje estimado", sort=False)   # sort=False respeta el orden de las categorías
    with st.expander("Ver tabla"):
        st.dataframe(resultado, hide_index=True, width="stretch")

with tab2:
    st.markdown(
        f"""
- **Variable que se predice:** puntaje global = suma de los 5 módulos de Saber Pro (0 a 1.500).
- **Modelo:** {op['modelo_nombre']}, dentro de un pipeline que codifica las variables (ordinales, one-hot y target encoding) y las usa como entrada.
- **Datos:** {miles(op['n_total'])} resultados oficiales limpios del ICFES (2018-2022); el modelo se entrenó con una muestra de {miles(op['n_train'])} estudiantes.
- **Desempeño en datos de prueba:** RMSE de {op['rmse_prueba']:.1f} puntos, MAE de {op['mae_prueba']:.1f} y R² de {op['r2_prueba']:.3f}.
"""
    )

with tab3:
    st.markdown(
        f"""
- **El contexto explica una parte del resultado, no todo.** El R² en prueba es de {op['r2_prueba']:.2f}: eso significa que cerca del {op['r2_prueba']*100:.0f} % de las diferencias
  entre estudiantes se explica con estas variables y el resto depende de otros factores que el modelo no ve (lo que cada persona sabe, su preparación, su motivación).
- **Refleja desigualdades de la población.** Las diferencias por estrato, educación de los padres o tipo de institución son patrones estadísticos
  de los datos, no una medida del potencial de las personas. No debe usarse para decidir admisiones, becas ni otras decisiones sobre individuos.
- **Datos de 2018 a 2022** y solo estudiantes de pregrado universitario que presentaron la prueba. No se garantiza que funcione igual para otros años o poblaciones.
- Las instituciones o programas con pocos estudiantes en los datos tienen estimaciones menos fiables.
"""
    )
