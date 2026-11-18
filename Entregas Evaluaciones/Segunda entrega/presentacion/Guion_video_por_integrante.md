# Guion del video por integrante · Entrega 2 · Grupo 36

**Duración estimada: 4:30** (617 palabras a 140 palabras por minuto, más los cambios de turno). Límite de la pauta: 5:00. Meta al ensayar: terminar antes de 4:45.

Cada integrante responde una de las cuatro preguntas guía de la pauta, en el orden de las diapositivas. Cada uno cierra su parte con una pregunta para el siguiente, así el video se escucha como una conversación y no como cuatro exposiciones pegadas.

| Integrante | Rol | Pregunta guía | Diapositivas | Tiempo |
|---|---|---|---|---|
| Matías | Ingeniero agrónomo | ¿Qué problema estamos resolviendo? | 1–3 | ~69 s |
| Jason | Ingeniero informático | ¿Cómo pensamos resolverlo? | 4–6 | ~63 s |
| Kelly | Ingeniera agroindustrial | ¿Cómo sabremos si funciona? | 7–10 | ~68 s |
| Nadia | Contadora | ¿Cómo se usaría el resultado? | 11–13 | ~64 s |

## Matías · ¿Qué problema estamos resolviendo?

_Ingeniero agrónomo. Conoce el cultivo: cuenta el problema de planificación y lee los hallazgos del EDA desde el campo (vigor, nivel propio de cada cuartel, frío invernal)._

**Diapositiva 1 · Portada** (~16 s)

Hola, ¿cómo están? Somos el Grupo 36: Nadia, Kelly, Jason y yo, Matías. En la primera entrega estudiamos un huerto de kiwi Hayward en Chimbarongo. Hoy les mostramos cómo anticipar cuánto rendirá cada cuartel antes de la cosecha.

_[Mira a cámara. Cuando nombras a cada uno, esa persona saluda con la mano.]_

**Diapositiva 2 · Problema y objetivo** (~26 s)

Partamos por el problema. Hoy el huerto planifica con el promedio histórico de cada cuartel, y ese promedio se equivoca 17 % en el total. ¿Por qué? Porque los años no se repiten: entre 2023 y 2024 el rendimiento saltó 59 %. Queremos estimar al 31 de enero las toneladas por hectárea de cada cuartel con menos error que ese promedio.

_[Haz una pausa corta después de «¿Por qué?».]_

**Diapositiva 3 · Hallazgos del EDA** (~24 s)

En los datos hay tres pistas. Primero, la canopia satelital, el vigor visto desde el satélite, al 31 de enero se correlaciona 0,66 con el rendimiento. Segundo, cada cuartel tiene su propio nivel: el que rinde bien, suele rendir bien siempre. Y tercero, con solo cuatro inviernos completos, el efecto del clima aún no es medible.

_[Cuenta las tres pistas con los dedos.]_

> **Pase** (Matías lo dice mirando a cámara): «Con esas pistas, Jason, ¿cómo lo resolvemos?»


## Jason · ¿Cómo pensamos resolverlo?

_Ingeniero informático. Construyó el análisis: justifica el tipo de problema, compara los modelos y presenta los resultados de la validación._

**Diapositiva 4 · Enfoque analítico** (~17 s)

Gracias, Matías. Primero definimos el tipo de problema. Necesitamos un número, toneladas por hectárea, porque bins, cuadrillas y equilibrio se calculan en toneladas. Es una regresión: entran la canopia y el cuartel, y sale el rendimiento con su intervalo.

_[Señala con el cursor la entrada y la salida del esquema.]_

**Diapositiva 5 · Alternativas** (~26 s)

Probamos cuatro modelos frente a lo que exige este caso: solo 77 observaciones, un nivel propio por cuartel y un año récord. El modelo mixto cumple todo: pocos parámetros, un punto de partida por cuartel, extrapola, da intervalos y Matías lo puede interpretar. Los árboles no predicen más allá de lo que vieron y, con tan pocos datos, los memorizan.

_[Recorre la fila del modelo mixto en la matriz.]_

**Diapositiva 6 · Resultados** (~17 s)

¿Y cómo le fue? Predijimos cada año solo con los anteriores, como en la realidad. El modelo mixto bajó el error del huerto de 17,4 a 9,6 %: casi la mitad. Ridge quedó cerca, así que queda de respaldo.

_[Pausa antes de decir «casi la mitad».]_

> **Pase** (Jason lo dice mirando a cámara): «Pero, ¿funciona de verdad? Eso lo cuenta Kelly.»


## Kelly · ¿Cómo sabremos si funciona?

_Ingeniera agroindustrial. Mira la operación y el control de calidad: traduce las métricas al costo de equivocarse en cosecha (bins, cuadrillas, fruta sin cosechar) y explica cómo se comprueba el modelo._

**Diapositiva 7 · Métricas** (~21 s)

Gracias, Jason. Yo lo miro desde la operación, porque equivocarse en cosecha cuesta plata. Si nos quedamos cortos, queda fruta en la planta; si nos pasamos, pagamos bins y cuadrillas que sobran. Por eso medimos el MAE, en toneladas por hectárea, y, para la logística, el error del huerto.

**Diapositiva 8 · Criterio de éxito** (~19 s)

Y definimos el éxito antes de ver resultados: error del huerto de 10 % o menos, 10 % menos error que el promedio, ganarle en cuatro de cinco años y que la mejora no sea casualidad. Cumplimos tres; la cuarta, con el modelo final.

_[Señala los tres criterios cumplidos y después el pendiente.]_

**Diapositiva 9 · Validación** (~11 s)

Validamos como en la vida real: entrenamos con el pasado y probamos con el año siguiente. Y el 2026 lo guardamos sin mirar, como prueba ciega.

**Diapositiva 10 · Resultado por año** (~13 s)

Miren este gráfico: el modelo gana en cuatro de cinco años, justo cuando el promedio más se equivocó. Y en la prueba ciega, el error bajó de 21,6 a 7 %.

_[Pasa el cursor sobre las barras de 2026.]_

> **Pase** (Kelly lo dice mirando a cámara): «Nadia, ¿y esto en qué le sirve al huerto?»


## Nadia · ¿Cómo se usaría el resultado?

_Contadora. Conecta el pronóstico con costos, presupuesto y punto de equilibrio, y cierra con los pendientes para la entrega final._

**Diapositiva 11 · Baseline y fecha de corte** (~21 s)

Buena pregunta. Siempre comparamos contra lo que el huerto hace hoy, el promedio de cada cuartel; aunque le demos toda la historia desde 2019, solo baja a 16,8 %. Y cortamos al 31 de enero porque ahí el modelo logra su mejor error y se cierran contratos y presupuesto.

**Diapositiva 12 · Implementación** (~26 s)

El administrador y el jefe técnico verán el pronóstico en un tablero tres veces al año: en diciembre, para reservar bins y cuadrillas; en enero, para cerrar ventas; y en mayo, para comparar con lo cosechado y reentrenar. Y lo que más me gusta: con un equilibrio de 19,1 toneladas por hectárea, sabremos a tiempo qué cuarteles no cubrirán sus costos.

_[Recorre el flujo de izquierda a derecha.]_

**Diapositiva 13 · Próximos pasos** (~17 s)

Para la entrega final queda ajustar el modelo, construir los intervalos y sumar el equilibrio al tablero. En resumen: un modelo mixto con canopia satelital, validado a ciegas, que casi reduce a la mitad el error de hoy. ¡Muchas gracias!

_[Todos se despiden juntos.]_

## Para que suene natural

1. No lo lean palabra por palabra. Usen el guion para ensayar y, al grabar, tengan a mano solo las ideas clave y las cifras.
2. Hablen como si se lo contaran a alguien del huerto, no a un jurado. Si una frase no les sale natural, cámbienla por la suya.
3. Las preguntas de cambio de turno («Jason, ¿cómo lo resolvemos?») son el hilo del video: díganlas mirando a cámara, como una conversación.
4. Hagan una pausa breve antes de cada cifra importante (17,4 → 9,6 %; 21,6 → 7 %). La pausa la destaca más que subir la voz.
5. Las indicaciones en cursiva no se dicen: son gestos o movimientos del cursor para que el video no se vea estático.
6. Las cifras sí deben quedar exactas: la pauta descuenta por «IA sin personalizar», no por usar palabras propias.

## Cómo grabar

1. Grabar en una reunión de Zoom o Teams con la presentación compartida en pantalla completa. Una sola persona avanza las diapositivas, atenta a quién está hablando.
2. Cámara de todos encendida, o de nadie. Si se usa, que quien habla esté destacado.
3. Ensayar dos veces con cronómetro. Si se pasa de 4:45, recortar frases, no hablar más rápido.
4. Una sola persona sube el video y el PDF antes del 30 de noviembre.
