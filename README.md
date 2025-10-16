
---
## Assignment: Logistic Regression and Multiclass Extensions

# TAREA2_ML_LLARO_RIOS_VARGAS
TAREA  - MACHINE LEARNING

# Grupo
- LLARO CASTRO, DIEGO RENATO
- RIOS MEZA, JENNIFER SASKIA
- VARGAS FLORES, JOHANNA ANTONELLA

**Deadline:** Monday, October 13th, 2025, 23:59
**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ucimlrepo`.
---


# Solución
1. **How the gradient differs between binary, OvA, and multinomial forms**
    - En la ***regresión logística binaria*** (Heart Disease dataset), el modelo alcanzó un ***84.6% de exactitud***, con alta precisión (92.3%) y buen recall (76.6%), muy similar al modelo de scikit-learn. Esto confirma que el gradiente descendente se implementó correctamente y que la convergencia fue estable tras más de 16 000 iteraciones.

    - En el enfoque ***One-vs-All (OvA)*** aplicado al dataset Wine, la exactitud fue ***98.15%***, igual al resultado de scikit-learn. Cada clasificador binario logró separar correctamente su clase, aunque los coeficientes mostraron ***normas L2 mayores***, lo que indica un ajuste más sensible por la falta de regularización conjunta.

    - En el modelo ***multinomial (Softmax)***, también con Wine, se obtuvo el mismo ***98.15% de exactitud***, demostrando que el gradiente conjunto actualiza de forma coherente todas las clases y mantiene la misma capacidad predictiva que OvA, pero con un equilibrio interno más estable.


2. **How numerical stability issues may arise in softmax**
La implementación del modelo Softmax incorporó una corrección numérica para evitar overflow, restando el valor máximo de los logits antes de calcular las exponenciales.
Esta medida garantizó estabilidad durante el entrenamiento, sin errores y con métricas idénticas a scikit-learn. La coincidencia en la matriz de confusión evidencia que el modelo fue ***numéricamente estable y preciso***.


3. **When OvA and multinomial approaches diverge in predictions**
En los resultados obtenidos, ***OvA y Multinomial alcanzaron la misma exactitud y matriz de confusión***, lo que refleja que las clases del dataset Wine están bien separadas.
Sin embargo, los ***coeficientes más grandes en OvA*** indican mayor esfuerzo individual de cada modelo por separar su clase. En problemas con solapamiento o desbalance, el enfoque ***multinomial*** suele comportarse mejor, ya que optimiza todas las clases simultáneamente y mantiene probabilidades coherentes.


4. **Conclusión General**
Los tres modelos demostraron alta eficacia y coherencia con scikit-learn.
El modelo ***binario*** logró una buena clasificación en un problema médico; ***OvA*** fue efectivo pero con pesos más amplios; y ***Softmax*** ofreció el mismo rendimiento con mejor estabilidad y coherencia probabilística.
Esto confirma la correcta implementación del gradiente y el dominio del proceso de aprendizaje en modelos de clasificación.

