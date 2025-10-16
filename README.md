
---
## Assignment: Logistic Regression and Multiclass Extensions

# TAREA2_ML_LLARO_RIOS_VARGAS
TAREA 2 - MACHINE LEARNING

# Grupo
- LLARO CASTRO, DIEGO RENATO
- RIOS MEZA, JENNIFER SASKIA
- VARGAS FLORES, JOHANNA ANTONELLA

**Deadline:** Monday, October 13th, 2025, 23:59
---

**Environment:** Python, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ucimlrepo`.
---


# Solución
1. **How the gradient differs between binary, OvA, and multinomial forms**

    El gradiente para la regresión logística y binaria es el mismo, la diferencia radica en que en la RL binaria se ajusta un único conjunto de pesos para estimar la probabilidad de pertenecer a una clase, es un único escalar que se obtiene a partir de la función sigmoide sobre la probabilidad de pertenecer a una de las dos clases. Donde el gradiente indica cómo deben actualizarse los parámetros para minimizar el error entre las predicciones y los valores reales.

    Para el método One-vs-All (OvA), se entrenan varios modelos binarios independientes, uno por cada clase. Cada modelo aprende a distinguir una clase frente a las demás. Aunque este enfoque es simple y efectivo, sus decisiones pueden no ser consistentes cuando las clases se superponen o los datos están desequilibrados.

    En el modelo multinomial (Softmax), todas las clases se entrenan simultáneamente dentro de un mismo modelo. Las actualizaciones de los pesos están interrelacionadas ya que como se vio en el código en la función softmax en el denominador se encuentran todas las clases de 1 hasta K, logrando una estructura probabilística más coherente.

    Dentro de nuestros resultados, tenemos que:

    - En la ***regresión logística binaria*** (Heart Disease dataset), el modelo alcanzó un ***84.6% de exactitud***, con alta precisión (92.3%) y buen recall (76.6%), muy similar al modelo de scikit-learn. Esto confirma que el gradiente descendente se implementó correctamente y que la convergencia fue estable tras más de 16 000 iteraciones.

    - En el enfoque ***One-vs-All (OvA)*** aplicado al dataset Wine, la exactitud fue ***98.15%***, igual al resultado de scikit-learn. Cada clasificador binario logró separar correctamente su clase, aunque los coeficientes mostraron ***normas L2 mayores***, lo que indica un ajuste más sensible por la falta de regularización conjunta.

    - En el modelo ***multinomial (Softmax)***, también con Wine, se obtuvo el mismo ***98.15% de exactitud***, demostrando que el gradiente conjunto actualiza de forma coherente todas las clases y mantiene la misma capacidad predictiva que OvA, pero con un equilibrio interno más estable.


2. **How numerical stability issues may arise in softmax**

    La función Softmax traslada al código puede sufrir problemas de estabilidad numérica debido al uso de exponentes muy grandes o pequeños, esto crea errores de desbordamiento (overflow), por ello para evitar estos problemas, se utilizó una versión ***numéricamente estable y precisa*** de la función, que consiste en restar el valor máximo del vector `(z -= np.max(z, axis=1, keepdims=True)` antes de aplicar la exponencial.
    Esta técnica mantiene los resultados dentro de un rango seguro sin alterar la distribución final. Gracias a esto, el modelo Softmax conserva la precisión y evita que los cálculos se distorsionen, garantizando la estabilidad durante el entrenamiento, sin errores y con métricas idénticas a scikit-learn.


3. **When OvA and multinomial approaches diverge in predictions**

    En los resultados obtenidos, ***OvA y Multinomial alcanzaron la misma exactitud y matriz de confusión***, lo que refleja que las clases del dataset Wine están bien separadas.
    Sin embargo, los ***coeficientes más grandes en OvA*** indican mayor esfuerzo individual de cada modelo por separar su clase. En problemas con solapamiento o desbalance, el enfoque ***multinomial*** suele comportarse mejor, ya que optimiza todas las clases simultáneamente y mantiene probabilidades coherentes.


4. **Conclusión General**

    Los tres modelos demostraron alta eficacia y coherencia con scikit-learn.
    El modelo ***binario*** logró una buena clasificación en un problema médico; ***OvA*** fue efectivo pero con pesos más amplios; y ***Softmax*** ofreció el mismo rendimiento con mejor estabilidad y coherencia probabilística.
    Esto confirma la correcta implementación del gradiente y el dominio del proceso de aprendizaje en modelos de clasificación.
