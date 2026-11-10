# Guion del video - Entrega 2 (Grupo 36)

Máximo 5 minutos. Ritmo natural de unas 140 palabras por minuto.

**Total: 574 palabras, unos 4.1 minutos.**

## Slide 1 - Pronóstico de rendimiento por cuartel en kiwi Hayward
_36 palabras, unos 15 s_

Somos el Grupo 36. En la Entrega 1 caracterizamos un huerto de kiwi Hayward en Chimbarongo. Hoy proponemos cómo pronosticar su rendimiento por cuartel, cómo sabremos si funciona y cómo se usará para planificar la cosecha.

## Slide 2 - Hoy el huerto planifica con un promedio que falla por 17 %
_57 palabras, unos 24 s_

Breve recordatorio: el huerto planifica con el promedio histórico de cada cuartel, y ese método falla en promedio un 17 % en el total. El rendimiento salta hasta 59 % entre años y varía al doble entre cuarteles. Nuestro objetivo es estimar al 31 de enero el rendimiento de cada cuartel con menos error que ese promedio.

## Slide 3 - Tres hallazgos del EDA definen la metodología
_54 palabras, unos 23 s_

Tres hallazgos del EDA condicionan la metodología. El panel es pequeño, 77 observaciones, por eso usamos pocos predictores. Cada cuartel mantiene su nivel, así que el modelo necesita un efecto por cuartel. Y la canopia acumulada hasta enero se correlaciona 0,66 con el rendimiento. El clima solo tiene cuatro temporadas completas: queda como sensibilidad.

## Slide 4 - Regresión supervisada sobre un panel cuartel-temporada
_54 palabras, unos 23 s_

Es un problema de regresión sobre un panel cuartel-temporada, con el rendimiento en toneladas por hectárea como objetivo. Entran la canopia, el cuartel y su superficie. Sale el rendimiento con intervalo, los kilos y bins del huerto, el rango de la temporada y los cuarteles en riesgo. Elegimos regresión porque las decisiones necesitan toneladas.

## Slide 5 - Comparamos cuatro alternativas con supuestos distintos
_39 palabras, unos 17 s_

Comparamos cuatro alternativas. El modelo mixto y Ridge son lineales, interpretables y adecuados para paneles pequeños; el mixto además entrega intervalos. Random Forest y Gradient Boosting captan no linealidades, pero no extrapolan y con 77 casos tienden a sobreajustar.

## Slide 6 - El modelo mixto reduce casi a la mitad el error del huerto
_49 palabras, unos 21 s_

Probamos las alternativas fuera de muestra, prediciendo cada año entre 2022 y 2026 con los años anteriores. El modelo mixto baja el error del huerto de 17,4 a 9,6 %. Ridge queda casi igual y los árboles rinden peor. Por eso elegimos el modelo mixto, con Ridge como respaldo.

## Slide 7 - Métricas elegidas según el costo de equivocarse
_53 palabras, unos 23 s_

La métrica principal es el MAE, porque se lee en toneladas por hectárea. Para la operación importa el error del total del huerto. El RMSE castiga errores grandes y el MAPE compara cuarteles. Subestimar deja fruta sin cosechar; sobreestimar compromete costos. Cerca del equilibrio el error cambia la decisión, por eso reportaremos intervalos.

## Slide 8 - Criterio de éxito, fijado antes de implementar
_52 palabras, unos 22 s_

El criterio de éxito tiene tres condiciones: error del huerto de 10 % o menos, al menos 10 % menos MAE que el promedio histórico, y ganarle en cuatro de cinco temporadas. Corregimos la Entrega 1: un MAPE bajo 10 % por cuartel no es realista, el mejor modelo llega a 19.

## Slide 9 - Validación temporal: el pasado predice el año siguiente
_46 palabras, unos 20 s_

Validamos con ventana expansiva: entrenamos con el pasado y probamos el año siguiente, porque un split aleatorio sería optimista. Controlamos sobreajuste con pocos predictores, fuga usando solo datos previos al corte, desbalance reportando el error por año, y representatividad tratando aparte los tres cuarteles sin canopia.

## Slide 10 - Baseline y momento del pronóstico
_40 palabras, unos 17 s_

El baseline es la regla actual: el promedio histórico del cuartel. En diciembre el modelo ya baja el error a 10,3 % y en enero llega a 9,6. Febrero no mejora, por eso proponemos el corte al 31 de enero.

## Slide 11 - Del satélite a la decisión de cosecha
_51 palabras, unos 22 s_

La solución recibe la canopia, la cosecha y el clima, y produce el pronóstico por cuartel. Lo usan el administrador y el jefe técnico en un tablero. En diciembre se reservan bins y cuadrillas, en enero se cierran ventas y presupuesto, y en mayo se mide el error y se reentrena.

## Slide 12 - Próximos pasos
_43 palabras, unos 18 s_

Para la entrega final queda confirmar los polígonos satelitales, ajustar el modelo final, construir intervalos con bootstrap, integrar el punto de equilibrio y prototipar el tablero. En síntesis: un modelo mixto con canopia satelital, validado en el tiempo contra la práctica actual. Gracias.
