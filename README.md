# 🧠 Análisis y Predicción de Salarios usando Machine Learning

Este proyecto tiene como objetivo analizar y predecir los **salarios anuales** de trabajadores/as en función de variables como experiencia, título, empresa, antigüedad y salario base, utilizando técnicas de **ciencia de datos** y **machine learning**.

---

## 📁 Dataset

El dataset utilizado proviene de la plataforma [Levels.fyi](https://www.levels.fyi/) y contiene información detallada de salarios anuales en empresas del sector tecnológico.  
**Variables principales:**

- `timestamp`: Fecha de carga de la información.
- `company`: Empresa empleadora.
- `level`: Nivel o rango dentro de la empresa.
- `title`: Cargo o título profesional.
- `totalyearlycompensation`: Salario total anual.
- `basesalary`: Salario base.
- `bonus`: Bono anual.
- `yearsofexperience`: Años de experiencia.
- `yearsatcompany`: Años en la empresa.
- `gender`: Género (limpiado).

*(Otros campos fueron omitidos por irrelevancia o ruido)*

---

## 🧹 Limpieza de Datos

Se realizaron los siguientes pasos de preprocesamiento:

1. **Eliminación de columnas irrelevantes:** `Race`, `Education`, `otherdetails`.
2. **Imputación de valores faltantes:**
   - Relleno con valores por defecto en columnas categóricas (`level`, `tag`, `company`).
   - Conversión y completado de `dmaid` como valor numérico promedio.
   - Reemplazo de valores incorrectos en la columna `gender`.
3. **Tratamiento de outliers:**  
   Se eliminaron registros con un Z-Score > 0.7 en la variable `totalyearlycompensation` (aproximadamente el 25% del dataset).
4. **Codificación de variables categóricas:**  
   Se utilizó `LabelEncoder` y `get_dummies` según el modelo.

---

## 📊 Visualización de Datos

Se generaron visualizaciones para comprender la distribución y relaciones entre variables:

- **Boxplots**: Comparación de salarios según `gender` y `title`.
- **Line plots (Plotly)**: Evolución temporal del salario de Data Scientists (con 0 y 1 año de experiencia).
- **Pie chart**: Proporción de cargos con ≤1 año de experiencia y antigüedad.
- **Bar chart**: Promedio de salario por cargo (`title`).
- **Heatmap de correlación**: Relación entre variables numéricas.

---

## 📈 Modelado Predictivo

Se probaron distintos modelos de regresión para predecir `totalyearlycompensation`.

### 🔍 Primer enfoque (features seleccionadas):

- `company`, `title`, `tag`, `yearsofexperience`

Utilizando el paquete **LazyPredict**, se compararon distintos modelos:

- `LinearRegression`, `Lasso`, `Ridge`, `RandomForest`, `ExtraTrees`, `LGBMRegressor`, `HuberRegressor`

> 🔹 **Mejor modelo**: `LGBMRegressor`, aunque con bajo rendimiento inicial.

---

## 🔧 Ajuste del Modelo

Se mejoró el modelo utilizando nuevas features:

- `basesalary`, `yearsofexperience`, `yearsatcompany`, `title`

Luego:

- Se creó una nueva variable `Antiguedad` combinando experiencia y años en la empresa.
- Se eliminaron registros con `basesalary = 0`.
- Se aplicó **StandardScaler** para normalizar los datos.

Modelo final entrenado:

```python
model = lgb.LGBMRegressor(force_col_wise=True)
model.fit(X_train, Y_train)
