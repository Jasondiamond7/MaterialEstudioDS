# Guion del video por integrante · Entrega 2 · Grupo 36

**Duración estimada: 3:57** (540 palabras a 140 palabras por minuto, más los cambios de turno). Límite de la pauta: 5:00. Meta al ensayar: terminar antes de 4:45.

Cada integrante responde una de las cuatro preguntas guía de la pauta, en el orden de las diapositivas.

| Integrante | Rol | Pregunta guía | Diapositivas | Tiempo |
|---|---|---|---|---|
| Matías | Ingeniero agrónomo | ¿Qué problema estamos resolviendo? | 1–3 | ~58 s |
| Jason | Ingeniero informático | ¿Cómo pensamos resolverlo? | 4–6 | ~57 s |
| Kelly | Ingeniera agroindustrial | ¿Cómo sabremos si funciona? | 7–10 | ~61 s |
| Nadia | Contadora | ¿Cómo se usaría el resultado? | 11–13 | ~55 s |

## Matías · ¿Qué problema estamos resolviendo?

_Ingeniero agrónomo. Conoce el cultivo: explica el problema de planificación y lee los hallazgos del EDA desde lo agronómico (vigor, nivel propio de cada cuartel, frío invernal)._

**Diapositiva 1 · Portada** (~13 s)

Hola, somos el Grupo 36: Nadia, Kelly, Jason y yo, Matías. En la Entrega 1 caracterizamos un huerto de kiwi Hayward en Chimbarongo. Hoy proponemos cómo pronosticar su rendimiento por cuartel.

**Diapositiva 2 · Problema y objetivo** (~23 s)

Hoy el huerto planifica con el promedio histórico de cada cuartel, y ese método falla un 17 % en el total del huerto: el rendimiento saltó 59 % entre 2023 y 2024. Nuestro objetivo es estimar al 31 de enero las toneladas por hectárea de cada cuartel, con menos error que ese promedio.

**Diapositiva 3 · Hallazgos del EDA** (~20 s)

Tres hallazgos definen el método. La canopia satelital, que refleja el vigor del huerto, ya se correlaciona 0,66 con el rendimiento al 31 de enero. Cada cuartel mantiene su propio nivel año a año. Y el clima tiene solo cuatro inviernos completos: todavía no alcanza para modelarlo.

> Pase: «Jason explica cómo lo resolvemos.»


## Jason · ¿Cómo pensamos resolverlo?

_Ingeniero informático. Construyó el análisis: justifica el tipo de problema, compara los modelos y presenta los resultados de la validación._

**Diapositiva 4 · Enfoque analítico** (~14 s)

Es un problema de regresión: necesitamos un número, las toneladas por hectárea, porque bins, cuadrillas y equilibrio se calculan en toneladas. Entran la canopia y el cuartel; sale el rendimiento con un intervalo.

**Diapositiva 5 · Alternativas** (~23 s)

Comparamos cuatro modelos con criterios del caso. Con 77 observaciones, un nivel propio por cuartel y un año récord, el modelo mixto cumple todo: pocos parámetros, un efecto por cuartel, extrapola, entrega intervalos y el agrónomo lo puede interpretar. Los árboles no predicen sobre lo ya visto y con tan pocos datos sobreajustan.

**Diapositiva 6 · Resultados** (~17 s)

Prediciendo cada año con los anteriores, el modelo mixto baja el error del huerto de 17,4 a 9,6 %, casi a la mitad. Ridge queda cerca, como respaldo, y los árboles rinden peor, así que con estos datos los descartamos.

> Pase: «Kelly cuenta cómo sabremos si funciona.»


## Kelly · ¿Cómo sabremos si funciona?

_Ingeniera agroindustrial. Mira la operación y el control de calidad: traduce las métricas al costo de equivocarse en cosecha (bins, cuadrillas, fruta sin cosechar) y explica cómo se comprueba el modelo._

**Diapositiva 7 · Métricas** (~16 s)

Elegimos las métricas según lo que cuesta equivocarse. Subestimar deja fruta sin cosechar; sobreestimar compromete costos sin fruta. La principal es el MAE, en toneladas por hectárea, y para la logística medimos el error del total del huerto.

**Diapositiva 8 · Criterio de éxito** (~18 s)

Fijamos el éxito antes de implementar: error del huerto de 10 % o menos, 10 % menos error que el promedio, ganarle en cuatro de cinco años y que la mejora no sea casualidad. Cumplimos tres; la cuarta queda para el modelo final.

**Diapositiva 9 · Validación** (~12 s)

Validamos como se usaría en la realidad: entrenamos con el pasado y probamos el año siguiente. Así evitamos la fuga de información, y dejamos 2026 como prueba ciega.

**Diapositiva 10 · Resultado por año** (~12 s)

El modelo gana en cuatro de cinco años, justo cuando el promedio más falla, y en la prueba ciega bajó el error del huerto de 21,6 a 7 %.

> Pase: «Nadia explica cómo se usaría.»


## Nadia · ¿Cómo se usaría el resultado?

_Contadora. Conecta el pronóstico con costos, presupuesto y punto de equilibrio, y cierra con los pendientes para la entrega final._

**Diapositiva 11 · Baseline y fecha de corte** (~15 s)

Siempre comparamos contra la regla actual, el promedio de cada cuartel. Y elegimos el 31 de enero porque ahí el modelo logra su mejor error y coincide con el cierre de contratos y presupuesto de cosecha.

**Diapositiva 12 · Implementación** (~27 s)

El administrador y el jefe técnico verán el pronóstico en un tablero tres veces al año: en diciembre, para reservar bins y cuadrillas; en enero, para cerrar ventas; y en mayo, para medir el error y reentrenar. Con un equilibrio de 19,1 toneladas por hectárea, anticipa qué cuarteles no cubrirían sus costos. Y si un año el error supera 15 %, se recalibra.

**Diapositiva 13 · Próximos pasos** (~13 s)

Para la entrega final queda ajustar el modelo, construir los intervalos e integrar el equilibrio al tablero. En síntesis: un modelo mixto con canopia satelital, validado a ciegas. Muchas gracias.

## Cómo grabar

1. Grabar en una reunión de Zoom o Teams con la presentación compartida en pantalla completa. Una sola persona avanza las diapositivas.
2. Cada integrante habla en su turno; la cámara es opcional, pero si se usa, que sea para todos.
3. Ensayar dos veces con cronómetro antes de grabar. Si se pasa de 4:45, recortar frases, no velocidad.
4. Decir el guion con palabras propias: la pauta descuenta por «IA sin personalizar». Las cifras sí deben quedar exactas.
5. Una sola persona sube el video y el PDF antes del 30 de noviembre.
