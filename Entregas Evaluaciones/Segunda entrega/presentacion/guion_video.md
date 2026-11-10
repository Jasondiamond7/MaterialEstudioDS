# Guion del video - Entrega 2 (Grupo 36)

Máximo 5 minutos. Ritmo natural de unas 140 palabras por minuto.

**Total: 623 palabras, unos 4,5 minutos.** Ensayar con cronómetro: la meta es terminar antes de 4:45.

## Slide 1 - Pronóstico de rendimiento por cuartel en kiwi Hayward
_32 palabras, unos 14 s_

Somos el Grupo 36. En la Entrega 1 caracterizamos un huerto de kiwi Hayward en Chimbarongo. Hoy proponemos cómo pronosticar su rendimiento por cuartel, cómo sabremos si funciona y cómo se usará.

## Slide 2 - Hoy el huerto planifica con un promedio que falla por 17 %
_43 palabras, unos 18 s_

Breve recordatorio: el huerto planifica con el promedio histórico de cada cuartel, y ese método falla un 17 % en el total del huerto. Nuestro objetivo es estimar al 31 de enero el rendimiento de cada cuartel con menos error que ese promedio.

## Slide 3 - Tres hallazgos del EDA definen la metodología
_53 palabras, unos 23 s_

Tres hallazgos del EDA definen la metodología. La canopia hasta enero se correlaciona 0,66 con el rendimiento, casi igual que la temporada completa de la Entrega 1: podemos pronosticar antes de cosechar. El panel es pequeño, 77 casos; cada cuartel mantiene su nivel; y el clima, con solo cuatro temporadas, queda como sensibilidad.

## Slide 4 - Regresión supervisada sobre un panel cuartel-temporada
_43 palabras, unos 18 s_

Es un problema de regresión sobre un panel cuartel-temporada. Entran la canopia, el cuartel y su superficie; sale el rendimiento con intervalo, los kilos del huerto, el rango de la temporada y los cuarteles en riesgo. Elegimos regresión porque las decisiones necesitan toneladas.

## Slide 5 - Comparamos cuatro alternativas con los criterios del caso
_51 palabras, unos 22 s_

Comparamos cuatro alternativas con criterios que salen del caso: panel pequeño, nivel propio por cuartel, años récord, intervalos para decidir cerca del equilibrio e interpretabilidad. El modelo mixto cumple todos. Ridge es casi igual, pero sin intervalos directos. Los árboles no extrapolan a un año récord y con 77 casos sobreajustan.

## Slide 6 - El modelo mixto reduce casi a la mitad el error del huerto
_40 palabras, unos 17 s_

Probamos todo fuera de muestra, prediciendo cada año con los anteriores. El modelo mixto baja el error del huerto de 17,4 a 9,6 %. Ridge queda casi igual y los árboles rinden peor. Elegimos el mixto, con Ridge como respaldo.

## Slide 7 - Métricas elegidas según el costo de equivocarse
_46 palabras, unos 20 s_

La métrica principal es el MAE, porque se lee en toneladas por hectárea. Para la operación importa el error del total del huerto. El RMSE castiga los errores grandes y el MAPE compara cuarteles. Cerca del equilibrio un error cambia la decisión: por eso reportaremos intervalos.

## Slide 8 - Criterio de éxito, fijado antes de implementar
_67 palabras, unos 29 s_

El éxito tiene cuatro condiciones, fijadas antes de implementar: error del huerto de 10 % o menos, al menos 10 % menos MAE que el promedio, ganarle en cuatro de cinco años, y que la mejora no sea ruido. Hoy se cumplen tres; la cuarta, el intervalo bootstrap, todavía roza el cero. Además corregimos la Entrega 1: un MAPE bajo 10 % por cuartel no es realista.

## Slide 9 - Validación temporal: el pasado predice el año siguiente
_54 palabras, unos 23 s_

Validamos con ventana expansiva: entrenamos con el pasado y probamos el año siguiente. Elegimos modelo y variable con 2022 a 2025, y dejamos 2026 como prueba ciega. Controlamos el sobreajuste con pocos predictores, la fuga con datos previos al corte, el desbalance con el error por año, y la representatividad tratando aparte tres cuarteles.

## Slide 10 - El modelo aporta justo cuando el promedio falla
_51 palabras, unos 22 s_

Por temporada, el modelo gana en cuatro de cinco años: justo cuando el promedio erraba entre 16 y 25 %. Solo pierde en 2025, cuando el promedio ya acertaba. Y en la prueba ciega de 2026, elegido sin mirar ese año, bajó el error del huerto de 21,6 a 7,0 %.

## Slide 11 - Baseline y momento del pronóstico
_50 palabras, unos 21 s_

El baseline es la regla actual, el promedio del cuartel, y es justo: con toda la historia desde 2019 da 16,8 %. En diciembre el modelo ya baja el error a 10,3 % y en enero a 9,6; febrero no mejora. Por eso el corte es el 31 de enero.

## Slide 12 - Del satélite a la decisión de cosecha
_48 palabras, unos 21 s_

La solución recibe canopia, cosecha y clima, y produce el pronóstico por cuartel. Lo usan el administrador y el jefe técnico en un tablero. En diciembre se reservan bins y cuadrillas, en enero se cierran ventas y presupuesto, y en mayo se mide el error y se reentrena.

## Slide 13 - Próximos pasos
_45 palabras, unos 19 s_

Para la entrega final queda confirmar los polígonos, ajustar el modelo, construir intervalos, confirmar que la mejora no es ruido, integrar el punto de equilibrio y prototipar el tablero. En síntesis: un modelo mixto con canopia satelital, validado a ciegas contra la práctica actual. Gracias.
