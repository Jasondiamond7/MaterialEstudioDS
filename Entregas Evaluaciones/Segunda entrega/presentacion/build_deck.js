// Genera la presentacion de la Entrega 2 (propuesta metodologica) del Grupo 36.
const path = require('path');
const fs = require('fs');
const pptxgen = require('pptxgenjs');

// Local copy of the theme helper so the deck builds on any machine (needs only pptxgenjs).
const { applyTheme } = require('./lib/apply_theme.js');

const SALIDAS = path.join(__dirname, '..', 'analisis', 'salidas');
// Robustness results written by analisis/Entrega2_Propuesta_metodologica.ipynb, section 4
// (per-year errors, blind test, bootstrap). Run the notebook before building the deck.
const ROB = JSON.parse(fs.readFileSync(path.join(SALIDAS, 'robustez.json'), 'utf8'));
const OUT = path.join(__dirname, 'Grupo36_Entrega2_Propuesta_metodologica.pptx');

const THEME = {
  name: 'Kiwi Hayward',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '1F2A1C', lt1: 'FFFFFF', dk2: '2F4A1E', lt2: 'EEF3E6',
    accent1: '6E9A1F', accent2: '8B5E34', accent3: 'B9CC6A',
    accent4: 'B23A2E', accent5: '6B7280', accent6: 'D9A21B',
    hlink: '2F4A1E', folHlink: '8B5E34',
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = 'Entrega 2 - Propuesta metodológica - Grupo 36';
pres.author = 'Grupo 36 - Diplomado en Data Science 2026';
const C = pres.SchemeColor;

const W = 13.333;
const M = 0.6;

pres.defineSlideMaster({
  title: 'PORTADA',
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 1.7, w: 9.2, h: 1.9, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: 'bottom', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: M, y: 3.8, w: 9.2, h: 0.9, fontSize: 20, color: C.accent3, valign: 'top', margin: 0 }, text: '' } },
  ],
});

pres.defineSlideMaster({
  title: 'CONTENIDO',
  background: { color: C.background1 },
  margin: [0.5, M, 0.6, M],
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: M, y: 0.35, w: W - 2 * M, h: 0.9, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text2, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
    { text: { text: 'Grupo 36 · Entrega 2 · Pronóstico de rendimiento en kiwi Hayward', options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: HEX.accent5, align: 'right' },
});

// ---------- helpers ----------
function txt(slide, text, opts) {
  slide.addText(text, { isTextBox: true, margin: 0, fontSize: 15, color: C.text1, valign: 'top', ...opts });
}

function card(slide, { x, y, w, h, fill = C.background2, name }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
}

function badge(slide, x, y, label, fill = C.accent1, d = 0.5) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: `badge ${label}` });
  slide.addText(label, { isTextBox: true, x, y, w: d, h: d, align: 'center', valign: 'middle', margin: 0, fontSize: 16, bold: true, color: C.background1 });
}

function stat(slide, { x, y, w, value, label, color = C.accent1 }) {
  txt(slide, value, { x, y, w, h: 0.9, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color, valign: 'bottom' });
  txt(slide, label, { x, y: y + 0.95, w, h: 0.8, fontSize: 14, color: C.text1 });
}

function bullets(items, extra = {}) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({
      text: r.text,
      options: { bold: r.bold, color: r.color, bullet: j === 0 ? { indent: 16 } : undefined, breakLine: j === runs.length - 1 && i < items.length - 1, paraSpaceAfter: 6, ...extra },
    }));
  }).flat();
}

const AXIS = { catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.dk1, catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', catAxisLabelFontSize: 12, valAxisLabelFontSize: 12, valGridLine: { color: 'DDE3D5', size: 0.75 }, catGridLine: { style: 'none' }, dataLabelFontFace: '+mn-lt', dataLabelFontSize: 12, dataLabelColor: HEX.dk1, titleFontFace: '+mn-lt', titleFontSize: 14, titleColor: HEX.dk2, legendFontFace: '+mn-lt', legendFontSize: 12 };

// Speaker notes double as the video script: build also writes guion_video.md from them.
const GUION = [];
function notas(slide, titulo, texto) {
  slide.addNotes(texto);
  GUION.push({ titulo, texto });
}

function escribirGuion() {
  const PPM = 140; // natural speaking rate, words per minute
  const palabras = (t) => t.split(/\s+/).filter(Boolean).length;
  const total = GUION.reduce((a, g) => a + palabras(g.texto), 0);
  const lineas = [
    '# Guion del video - Entrega 2 (Grupo 36)', '',
    `Máximo 5 minutos. Ritmo natural de unas ${PPM} palabras por minuto.`, '',
    `**Total: ${total} palabras, unos ${(total / PPM).toFixed(1).replace('.', ',')} minutos.** Ensayar con cronómetro: la meta es terminar antes de 4:45.`, '',
  ];
  GUION.forEach((g, i) => {
    const n = palabras(g.texto);
    lineas.push(`## Slide ${i + 1} - ${g.titulo}`, `_${n} palabras, unos ${Math.round((n / PPM) * 60)} s_`, '', g.texto, '');
  });
  fs.writeFileSync(path.join(__dirname, 'guion_video.md'), lineas.join('\n'), 'utf8');
  return total;
}

// ---------- 1. Portada ----------
pres.addSection({ title: 'Portada' });
let s = pres.addSlide({ masterName: 'PORTADA', sectionTitle: 'Portada' });
s.addText('Pronóstico de rendimiento por cuartel en kiwi Hayward', { placeholder: 'title' });
s.addText('Entrega N.° 2 · Propuesta metodológica y diseño de la solución analítica', { placeholder: 'body' });
txt(s, 'Jayson León · Kelly Martínez · Matías Reyes · Nadia Lell',{ x: M, y: 5.6, w: 9, h: 0.4, fontSize: 15, color: C.background1 });
txt(s, 'Grupo 36 · Diplomado en Data Science 2026 · Pontificia Universidad Católica de Chile · Noviembre 2026', { x: M, y: 6.05, w: 11, h: 0.4, fontSize: 13, color: C.accent3 });
// Motivo: corte transversal de kiwi estilizado
s.addShape(pres.shapes.OVAL, { x: 9.9, y: 1.4, w: 2.9, h: 2.9, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: 'kiwi piel' });
s.addShape(pres.shapes.OVAL, { x: 10.05, y: 1.55, w: 2.6, h: 2.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: 'kiwi pulpa' });
s.addShape(pres.shapes.OVAL, { x: 10.95, y: 2.45, w: 0.8, h: 0.8, fill: { color: 'F4F7EC' }, line: { color: 'F4F7EC' }, objectName: 'kiwi centro' });
notas(s, 'Pronóstico de rendimiento por cuartel en kiwi Hayward', `Somos el Grupo 36. En la Entrega 1 caracterizamos un huerto de kiwi Hayward en Chimbarongo. Hoy proponemos cómo pronosticar su rendimiento por cuartel, cómo sabremos si funciona y cómo se usará.`);

// ---------- 2. Problema y objetivo ----------
pres.addSection({ title: 'Problema y datos' });
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Problema y datos' });
s.addText('Hoy el huerto planifica con un promedio que falla por 17 %', { placeholder: 'title' });
stat(s, { x: M, y: 1.45, w: 3.6, value: '17,4 %', label: 'error del total del huerto al usar el promedio histórico (2022–2026)', color: C.accent4 });
stat(s, { x: M, y: 3.35, w: 3.6, value: '+59 %', label: 'salto de rendimiento entre 2023 (26,8 t/ha) y 2024 (42,6 t/ha)', color: C.accent2 });
stat(s, { x: M, y: 5.05, w: 3.6, value: '×2', label: 'brecha de rendimiento medio entre cuarteles del mismo huerto', color: C.accent2 });
card(s, { x: 4.9, y: 1.5, w: 7.83, h: 5.2, name: 'objetivo' });
txt(s, 'Objetivo analítico', { x: 5.25, y: 1.75, w: 7.1, h: 0.45, fontFace: THEME.headFontFace, fontSize: 22, bold: true, color: C.text2 });
txt(s, 'Estimar al 31 de enero el rendimiento (t/ha) de cada cuartel y del huerto, con un error menor que el promedio histórico, para planificar la cosecha y anticipar si cada cuartel cubrirá su punto de equilibrio.', { x: 5.25, y: 2.3, w: 7.1, h: 1.5, fontSize: 17 });
txt(s, 'Preguntas que responde', { x: 5.25, y: 3.95, w: 7.1, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
txt(s, bullets([
  [{ text: 'P6 · ', bold: true }, { text: '¿Cuántas t/ha rendirá cada cuartel esta temporada?' }],
  [{ text: 'P7 · ', bold: true }, { text: '¿La temporada será baja, normal o alta según el histórico?' }],
  [{ text: 'P8 · ', bold: true }, { text: '¿Qué cuarteles quedarían bajo el equilibrio de 19,1 t/ha?' }],
]), { x: 5.25, y: 4.45, w: 7.1, h: 2.0, fontSize: 16 });
notas(s, 'Hoy el huerto planifica con un promedio que falla por 17 %', `Breve recordatorio: el huerto planifica con el promedio histórico de cada cuartel, y ese método falla un 17 % en el total del huerto. Nuestro objetivo es estimar al 31 de enero el rendimiento de cada cuartel con menos error que ese promedio.`);

// ---------- 3. Datos y hallazgos ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Problema y datos' });
s.addText('Tres hallazgos del EDA definen la metodología', { placeholder: 'title' });
const sc = JSON.parse(fs.readFileSync(path.join(SALIDAS, 'scatter_ene.json'), 'utf8'));
s.addChart(pres.charts.SCATTER, [{ name: 'X', values: sc.x }, { name: 'Rendimiento (t/ha)', values: sc.y }], {
  x: M, y: 1.4, w: 6.2, h: 4.6, ...AXIS,
  chartColors: [HEX.accent1], lineSize: 0, lineDataSymbol: 'circle', lineDataSymbolSize: 7,
  showTitle: true, title: 'Canopia vs. rendimiento (r = 0,66; n = 77)',
  showValAxisTitle: true, valAxisTitle: 'Rendimiento (t/ha)', showCatAxisTitle: true, catAxisTitle: 'Canopia media 1 may–31 ene (señal 0–1)',
  valAxisTitleFontSize: 12, catAxisTitleFontSize: 12, valAxisTitleColor: HEX.dk1, catAxisTitleColor: HEX.dk1,
  valAxisTitleFontFace: '+mn-lt', catAxisTitleFontFace: '+mn-lt',
  // Range covers every observation (0,449–0,584); v1 cut the axis at 0,54 and hid 22 points.
  catAxisMinVal: 0.44, catAxisMaxVal: 0.60, catAxisMajorUnit: 0.04, valAxisMinVal: 10, valAxisMaxVal: 70, catAxisLabelFormatCode: '0.00', showLegend: false,
});
txt(s, 'En la Entrega 1 la canopia de toda la temporada daba r = 0,67, pero incluía los meses de cosecha. Cortando al 31 de enero se conserva casi toda la señal (0,66) y ya sirve para pronosticar.', { x: M, y: 6.05, w: 6.2, h: 0.85, fontSize: 13, italic: true, color: C.text2 });
const hall = [
  ['1', 'Panel pequeño', '77 cuartel-temporadas con canopia (11 cuarteles × 7 años) → pocos predictores y modelos regularizados.'],
  ['2', 'Efecto cuartel persistente', 'Cada cuartel mantiene su nivel entre años → el modelo debe incluir un efecto propio por cuartel.'],
  ['3', 'Clima aún no estimable', 'Solo 4 temporadas con invierno completo y variables colineales (|r| > 0,8) → se usa como análisis de sensibilidad.'],
];
hall.forEach(([n, t, d], i) => {
  const y = 1.45 + i * 1.75;
  badge(s, 7.3, y, n, i === 2 ? C.accent2 : C.accent1);
  txt(s, t, { x: 8.0, y: y - 0.02, w: 4.7, h: 0.4, fontSize: 18, bold: true, color: C.text2 });
  txt(s, d, { x: 8.0, y: y + 0.42, w: 4.7, h: 1.1, fontSize: 15 });
});
notas(s, 'Tres hallazgos del EDA definen la metodología', `Tres hallazgos del EDA definen la metodología. La canopia hasta enero se correlaciona 0,66 con el rendimiento, casi igual que la temporada completa de la Entrega 1: podemos pronosticar antes de cosechar. El panel es pequeño, 77 casos; cada cuartel mantiene su nivel; y el clima, con solo cuatro temporadas, queda como sensibilidad.`);

// ---------- 4. Enfoque ----------
pres.addSection({ title: 'Metodología' });
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Metodología' });
s.addText('Regresión supervisada sobre un panel cuartel-temporada', { placeholder: 'title' });
const cols4 = [
  { t: 'Entradas', c: C.accent2, items: ['Canopia satelital media del 1 de mayo al 31 de enero', 'Identidad del cuartel (efecto propio)', 'Superficie del cuartel', 'Clima: solo sensibilidad'] },
  { t: 'Modelo', c: C.accent1, items: ['Variable objetivo: rendimiento en t/ha (asimetría 0,41, sin transformar)', 'Un pronóstico por cuartel y temporada', 'Reentrenado cada año con la cosecha real'] },
  { t: 'Salidas', c: C.text2, items: ['t/ha por cuartel con intervalo de predicción', 'Kilos y bins totales del huerto', 'Rango bajo, normal o alto (terciles)', 'Cuarteles en riesgo bajo equilibrio'] },
];
cols4.forEach((col, i) => {
  const x = M + i * 4.1;
  card(s, { x, y: 1.5, w: 3.75, h: 3.9, name: `enfoque ${col.t}` });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.3, y: 1.8, w: 3.15, h: 0.6, rectRadius: 0.1, fill: { color: col.c }, line: { color: col.c }, objectName: `cabecera ${col.t}` });
  txt(s, col.t, { x: x + 0.3, y: 1.8, w: 3.15, h: 0.6, align: 'center', valign: 'middle', fontSize: 18, bold: true, color: C.background1 });
  txt(s, bullets(col.items), { x: x + 0.3, y: 2.65, w: 3.15, h: 2.6, fontSize: 15 });
  if (i < 2) s.addShape(pres.shapes.CHEVRON, { x: x + 3.83, y: 3.25, w: 0.22, h: 0.4, fill: { color: C.accent5 }, line: { color: C.accent5 }, objectName: `flecha ${i}` });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.65, w: W - 2 * M, h: 1.05, rectRadius: 0.12, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: 'por que regresion' });
txt(s, [{ text: '¿Por qué regresión y no clasificación? ', options: { bold: true, color: HEX.accent3 } }, { text: 'Las decisiones (bins, cuadrillas, equilibrio) necesitan toneladas; el rango bajo, normal o alto se deriva del pronóstico sin perder información.', options: { color: HEX.lt1 } }], { x: M + 0.3, y: 5.65, w: W - 2 * M - 0.6, h: 1.05, valign: 'middle', fontSize: 15 });
notas(s, 'Regresión supervisada sobre un panel cuartel-temporada', `Es un problema de regresión sobre un panel cuartel-temporada. Entran la canopia, el cuartel y su superficie; sale el rendimiento con intervalo, los kilos del huerto, el rango de la temporada y los cuarteles en riesgo. Elegimos regresión porque las decisiones necesitan toneladas.`);

// ---------- 5. Alternativas ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Metodología' });
s.addText('Comparamos cuatro alternativas con los criterios del caso', { placeholder: 'title' });
const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 14, valign: 'middle' } });
const crit5 = (t, d) => ({ text: [{ text: t, options: { bold: true, breakLine: true } }, { text: d, options: { fontSize: 11, color: HEX.accent5 } }], options: { fontSize: 13, color: HEX.dk1, valign: 'middle', fill: { color: HEX.lt2 } } });
// Verdict cell: mark (si / parcial / no) in colour, then a short reason.
const MARCA = { si: ['✓ ', '4E7A12'], parcial: ['~ ', 'B07D0B'], no: ['✗ ', HEX.accent4] };
const v = (m, t, o = {}) => ({ text: [{ text: MARCA[m][0], options: { bold: true, color: MARCA[m][1] } }, { text: t }], options: { fontSize: 13, color: HEX.dk1, valign: 'middle', ...o } });
const plain = (t, o = {}) => ({ text: t, options: { fontSize: 13, color: HEX.dk1, valign: 'middle', ...o } });
const SEL = { fill: { color: 'E3EDCF' } };
s.addTable([
  [hdr('Criterio (por qué importa aquí)'), hdr('Modelo mixto'), hdr('Ridge + efecto cuartel'), hdr('Random Forest'), hdr('Gradient Boosting')],
  [crit5('Supuestos', 'qué exige cada modelo'), plain('Lineal; errores normales; cuarteles de un mismo huerto', SEL), plain('Lineal; escalado con datos de entrenamiento'), plain('Sin forma fija; pide más datos'), plain('Secuencial; muchos hiperparámetros')],
  [crit5('Panel pequeño', 'EDA: 77 casos'), v('si', 'Pocos parámetros', SEL), v('si', 'Regularización'), v('parcial', 'Inestable'), v('no', 'Sobreajusta')],
  [crit5('Efecto cuartel', 'EDA: nivel persistente'), v('si', 'Intercepto aleatorio', SEL), v('si', 'Efecto fijo'), v('parcial', 'Solo vía promedio'), v('parcial', 'Solo vía promedio')],
  [crit5('Años récord', 'EDA: 2024 fuera de rango'), v('si', 'Extrapola', SEL), v('si', 'Extrapola'), v('no', 'No extrapola'), v('no', 'No extrapola')],
  [crit5('Costo del error', 'decidir cerca de 19,1 t/ha'), v('si', 'Intervalo directo', SEL), v('parcial', 'Solo con bootstrap'), v('parcial', 'Bosque de cuantiles'), v('parcial', 'Pérdida cuantílica')],
  [crit5('Interpretabilidad', 'la usa el agrónomo'), v('si', 't/ha por unidad de canopia', SEL), v('si', 'Coeficientes'), v('no', 'Caja negra'), v('no', 'Caja negra')],
  [crit5('Factibilidad', 'reentrenar 1 vez al año'), v('si', 'Segundos', SEL), v('si', 'Segundos'), v('si', 'Segundos'), v('parcial', 'Calibración costosa')],
], { x: M, y: 1.35, w: W - 2 * M, colW: [2.85, 2.32, 2.32, 2.32, 2.32], border: { type: 'solid', color: 'D5DCCB', pt: 0.75 }, rowH: [0.45, 0.8, 0.56, 0.56, 0.56, 0.56, 0.56, 0.56], margin: [0.04, 0.1, 0.04, 0.1] });
txt(s, 'Elegimos el modelo mixto porque cumple todos los criterios; Ridge queda como respaldo y los árboles como contraste para detectar no linealidad.', { x: M, y: 6.3, w: W - 2 * M, h: 0.55, fontSize: 14, italic: true, color: C.text2 });
notas(s, 'Comparamos cuatro alternativas con los criterios del caso', `Comparamos cuatro alternativas con criterios que salen del caso: panel pequeño, nivel propio por cuartel, años récord, intervalos para decidir cerca del equilibrio e interpretabilidad. El modelo mixto cumple todos. Ridge es casi igual, pero sin intervalos directos. Los árboles no extrapolan a un año récord y con 77 casos sobreajustan.`);

// ---------- 6. Evidencia preliminar ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Metodología' });
s.addText('El modelo mixto reduce casi a la mitad el error del huerto', { placeholder: 'title' });
const res = { labels: ['Promedio histórico', 'Temporada anterior', 'Ridge', 'Modelo mixto', 'Random Forest', 'Gradient Boosting'] };
s.addChart(pres.charts.BAR, [
  { name: 'Error del total del huerto (%)', labels: res.labels, values: [17.4, 19.1, 10.5, 9.6, 15.4, 11.8] },
  { name: 'MAPE por cuartel (%)', labels: res.labels, values: [22.8, 24.6, 19.9, 19.3, 21.9, 21.5] },
], {
  x: M, y: 1.4, w: 7.6, h: 5.0, barDir: 'bar', barGrouping: 'clustered', barGapWidthPct: 60, ...AXIS,
  chartColors: [HEX.accent1, HEX.accent3], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0',
  showLegend: true, legendPos: 'b', valAxisHidden: true, valGridLine: { style: 'none' },
  showTitle: true, title: 'Validación temporal 2022–2026 · canopia al 31 de enero (menor es mejor)', catAxisOrientation: 'maxMin',
});
card(s, { x: 8.55, y: 1.45, w: 4.18, h: 4.95, name: 'conclusion evidencia' });
txt(s, 'Elección', { x: 8.85, y: 1.7, w: 3.6, h: 0.45, fontFace: THEME.headFontFace, fontSize: 22, bold: true, color: C.text2 });
txt(s, bullets([
  [{ text: 'Modelo mixto ', bold: true }, { text: 'como principal: menor error y entrega intervalos.' }],
  [{ text: 'Ridge ', bold: true }, { text: 'como respaldo: resultado casi idéntico, confirma que la señal es lineal.' }],
  [{ text: 'Árboles descartados ', bold: true }, { text: 'con el tamaño actual: más error y menos interpretables.' }],
]), { x: 8.85, y: 2.3, w: 3.6, h: 2.3, fontSize: 15 });
txt(s, '−45 %', { x: 8.85, y: 4.6, w: 3.6, h: 0.8, fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.accent1, valign: 'bottom' });
txt(s, 'error del total del huerto frente a la práctica actual', { x: 8.85, y: 5.45, w: 3.6, h: 0.7, fontSize: 14 });
txt(s, 'Exploración preliminar, no es el modelo final: 55 predicciones fuera de muestra.', { x: M, y: 6.45, w: 7.6, h: 0.4, fontSize: 12, italic: true, color: C.accent5 });
notas(s, 'El modelo mixto reduce casi a la mitad el error del huerto', `Probamos todo fuera de muestra, prediciendo cada año con los anteriores. El modelo mixto baja el error del huerto de 17,4 a 9,6 %. Ridge queda casi igual y los árboles rinden peor. Elegimos el mixto, con Ridge como respaldo.`);

// ---------- 7. Métricas ----------
pres.addSection({ title: 'Evaluación' });
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Evaluación' });
s.addText('Métricas elegidas según el costo de equivocarse', { placeholder: 'title' });
const met = [
  ['MAE (t/ha)', 'Métrica principal: error en las unidades que usa el agrónomo y robusta a temporadas extremas.'],
  ['Error total del huerto (%)', 'Lo que importa para contratar bins, cuadrillas y ventas: errores por cuartel que se compensan.'],
  ['RMSE (t/ha)', 'Penaliza errores grandes, que son los que descuadran la cosecha en temporadas atípicas.'],
  ['MAPE (%)', 'Compara cuarteles de distinto nivel; se interpreta con cuidado en cuarteles de bajo rendimiento.'],
];
met.forEach(([m, d], i) => {
  const y = 1.5 + i * 1.28;
  card(s, { x: M, y, w: 6.9, h: 1.1, name: `metrica ${m}` });
  txt(s, m, { x: M + 0.3, y: y + 0.12, w: 6.3, h: 0.35, fontSize: 17, bold: true, color: C.text2 });
  txt(s, d, { x: M + 0.3, y: y + 0.5, w: 6.3, h: 0.55, fontSize: 14 });
});
txt(s, 'Costo del error es asimétrico', { x: 7.95, y: 1.5, w: 4.8, h: 0.45, fontFace: THEME.headFontFace, fontSize: 22, bold: true, color: C.text2 });
txt(s, bullets([
  [{ text: 'Subestimar: ', bold: true, color: HEX.accent4 }, { text: 'faltan bins y cuadrillas en plena cosecha; fruta que madura sin cosechar.' }],
  [{ text: 'Sobreestimar: ', bold: true, color: HEX.accent2 }, { text: 'costos y contratos comprometidos sin fruta que los respalde.' }],
  [{ text: 'Cerca de 19,1 t/ha: ', bold: true, color: HEX.dk2 }, { text: 'el error cambia la decisión económica; por eso se reporta el intervalo y no solo el valor puntual.' }],
]), { x: 7.95, y: 2.1, w: 4.8, h: 2.5, fontSize: 15 });
card(s, { x: 7.95, y: 4.95, w: 4.78, h: 1.6, fill: 'FBEDEB', name: 'ejemplo costo' });
txt(s, 'Ejemplo', { x: 8.25, y: 5.1, w: 4.2, h: 0.35, fontSize: 15, bold: true, color: C.accent4 });
txt(s, 'Con un MAE de 6 t/ha, un cuartel pronosticado en 24 t/ha podría rendir entre 18 y 30: a ambos lados del equilibrio.', { x: 8.25, y: 5.5, w: 4.2, h: 0.95, fontSize: 14 });
notas(s, 'Métricas elegidas según el costo de equivocarse', `La métrica principal es el MAE, porque se lee en toneladas por hectárea. Para la operación importa el error del total del huerto. El RMSE castiga los errores grandes y el MAPE compara cuarteles. Cerca del equilibrio un error cambia la decisión: por eso reportaremos intervalos.`);

// ---------- 8. Criterio de éxito ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Evaluación' });
s.addText('Criterio de éxito, fijado antes de implementar', { placeholder: 'title' });
const bs = ROB.bootstrap;
const dec = (n) => n.toFixed(1).replace('.', ',').replace('-', '−'); // typographic minus
const crit = [
  ['≤ 10 %', 'error del total del huerto en validación temporal', 'Preliminar: 9,6 % ✓', true],
  ['≥ 10 %', 'menos MAE por cuartel que el promedio histórico', `Preliminar: ${dec(bs['reduccion_%'])} % ✓\n(MAE ${dec(bs.mae_baseline)} → ${dec(bs.mae_modelo)} t/ha)`, true],
  ['4 de 5', 'temporadas de prueba en que supera al baseline', 'Preliminar: 4 de 5 ✓\n(solo falla en 2025)', true],
  ['IC > 0', 'el intervalo de 95 % de la mejora en MAE excluye el cero (bootstrap por cuartel)', `Preliminar: ${dec(bs.ic95[0])} a ${dec(bs.ic95[1])} t/ha\npendiente`, false],
];
const cw4 = 2.85, cg4 = (W - 2 * M - 4 * cw4) / 3;
crit.forEach(([val, l, p, ok], i) => {
  const x = M + i * (cw4 + cg4);
  card(s, { x, y: 1.5, w: cw4, h: 3.3, fill: ok ? C.background2 : 'FBF3DD', name: `criterio ${i + 1}` });
  txt(s, val, { x: x + 0.25, y: 1.7, w: cw4 - 0.5, h: 0.95, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: ok ? C.accent1 : C.accent6, valign: 'bottom' });
  txt(s, l, { x: x + 0.25, y: 2.75, w: cw4 - 0.5, h: 1.15, fontSize: 15 });
  txt(s, p, { x: x + 0.25, y: 3.95, w: cw4 - 0.5, h: 0.75, fontSize: 13, italic: true, color: C.text2 });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.1, w: W - 2 * M, h: 1.6, rectRadius: 0.12, fill: { color: 'FBEDEB' }, line: { color: 'FBEDEB' }, objectName: 'ajuste meta' });
txt(s, 'Ajuste respecto de la Entrega 1', { x: M + 0.3, y: 5.25, w: 11.5, h: 0.4, fontSize: 16, bold: true, color: C.accent4 });
txt(s, 'La meta de MAPE < 10 % por cuartel no es alcanzable con la variabilidad observada: el mejor modelo preliminar llega a 19 %. Se reemplaza por criterios que reflejan la decisión real (total del huerto), la mejora frente a la práctica actual y que esa mejora no sea ruido.', { x: M + 0.3, y: 5.7, w: 11.5, h: 0.9, fontSize: 15 });
notas(s, 'Criterio de éxito, fijado antes de implementar', `El éxito tiene cuatro condiciones, fijadas antes de implementar: error del huerto de 10 % o menos, al menos 10 % menos MAE que el promedio, ganarle en cuatro de cinco años, y que la mejora no sea ruido. Hoy se cumplen tres; la cuarta, el intervalo bootstrap, todavía roza el cero. Además corregimos la Entrega 1: un MAPE bajo 10 % por cuartel no es realista.`);

// ---------- 9. Validación ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Evaluación' });
s.addText('Validación temporal: el pasado predice el año siguiente', { placeholder: 'title' });
const anios = [2020, 2021, 2022, 2023, 2024, 2025, 2026];
const gx = M + 1.0, gy = 1.95, cw = 0.78, ch = 0.5, gap = 0.08;
anios.forEach((a, j) => txt(s, String(a), { x: gx + j * (cw + gap), y: 1.5, w: cw, h: 0.35, fontSize: 12, bold: true, align: 'center', color: C.text2 }));
[2022, 2023, 2024, 2025, 2026].forEach((t, i) => {
  const y = gy + i * (ch + 0.14);
  txt(s, `Prueba ${i + 1}`, { x: M, y, w: 0.95, h: ch, fontSize: 12, valign: 'middle', color: C.text1 });
  anios.forEach((a, j) => {
    if (a > t) return;
    // 2022-2025 test years are used to choose model and canopy variable; 2026 is scored once, blind.
    const color = a !== t ? C.accent1 : (t === 2026 ? C.accent4 : C.accent2);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + j * (cw + gap), y, w: cw, h: ch, rectRadius: 0.06, fill: { color }, line: { color }, objectName: `fold ${t} ${a}` });
  });
});
badge(s, M, 5.55, '', C.accent1, 0.3); txt(s, 'Entrenamiento', { x: M + 0.4, y: 5.55, w: 1.6, h: 0.3, fontSize: 13, valign: 'middle' });
badge(s, M + 2.0, 5.55, '', C.accent2, 0.3); txt(s, 'Validación (elige modelo)', { x: M + 2.4, y: 5.55, w: 2.4, h: 0.3, fontSize: 13, valign: 'middle' });
badge(s, M + 4.85, 5.55, '', C.accent4, 0.3); txt(s, 'Prueba ciega', { x: M + 5.25, y: 5.55, w: 1.6, h: 0.3, fontSize: 13, valign: 'middle' });
txt(s, 'Ventana expansiva: nunca se usa el futuro para predecir el pasado. Modelo y variable de canopia se eligen con 2022–2025; 2026 se evalúa una sola vez.', { x: M, y: 6.0, w: 6.9, h: 0.75, fontSize: 13, italic: true, color: C.text2 });
const riesgos = [
  ['Sobreajuste', '77 casos: un predictor principal, regularización y efecto aleatorio; la elección no mira 2026.'],
  ['Fuga de información', 'Solo datos previos al corte; promedios por cuartel y escalado calculados con años de entrenamiento.'],
  ['Desbalance', 'Temporadas extremas son escasas (2024 récord): se reporta el error por año y el RMSE.'],
  ['Representatividad', 'Un huerto; 11 de 14 cuarteles con canopia. Los 3 restantes usan el baseline hasta tener señal.'],
];
riesgos.forEach(([r, d], i) => {
  const y = 1.5 + i * 1.3;
  txt(s, r, { x: 7.85, y, w: 4.9, h: 0.35, fontSize: 16, bold: true, color: C.accent4 });
  txt(s, d, { x: 7.85, y: y + 0.38, w: 4.9, h: 0.85, fontSize: 14 });
});
notas(s, 'Validación temporal: el pasado predice el año siguiente', `Validamos con ventana expansiva: entrenamos con el pasado y probamos el año siguiente. Elegimos modelo y variable con 2022 a 2025, y dejamos 2026 como prueba ciega. Controlamos el sobreajuste con pocos predictores, la fuga con datos previos al corte, el desbalance con el error por año, y la representatividad tratando aparte tres cuarteles.`);

// ---------- 10. Resultado por temporada y prueba ciega ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Evaluación' });
s.addText('El modelo aporta justo cuando el promedio falla', { placeholder: 'title' });
const filasAnio = ROB.por_anio;
const B0K = 'Error_huerto_%|B0 Promedio historico del cuartel';
const M2K = 'Error_huerto_%|M2 Modelo mixto (cuartel aleatorio)';
const etiquetasAnio = filasAnio.map((r) => (r.anio_grupo === 2026 ? '2026 (ciega)' : String(r.anio_grupo)));
s.addChart(pres.charts.BAR, [
  { name: 'Promedio histórico (baseline)', labels: etiquetasAnio, values: filasAnio.map((r) => +r[B0K].toFixed(1)) },
  { name: 'Modelo mixto', labels: etiquetasAnio, values: filasAnio.map((r) => +r[M2K].toFixed(1)) },
], {
  x: M, y: 1.4, w: 7.2, h: 5.0, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 60, ...AXIS,
  chartColors: [HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0',
  valAxisMinVal: 0, valAxisMaxVal: 30, valAxisHidden: true, valGridLine: { style: 'none' }, showLegend: true, legendPos: 'b',
  showTitle: true, title: 'Error del total del huerto (%) por temporada de prueba',
});
const ciego = Object.fromEntries(ROB.ciego_2026.map((r) => [r.modelo.slice(0, 2), r]));
card(s, { x: 8.15, y: 1.45, w: 4.58, h: 2.75, name: 'prueba ciega' });
txt(s, 'Prueba ciega 2026', { x: 8.45, y: 1.65, w: 4.0, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
txt(s, [
  { text: `${dec(ciego.B0['Error_huerto_%'])} %`, options: { color: HEX.accent5 } },
  { text: ' → ', options: { color: HEX.accent5 } },
  { text: `${dec(ciego.M2['Error_huerto_%'])} %`, options: { color: HEX.accent1 } },
], { x: 8.45, y: 2.1, w: 4.0, h: 0.8, fontFace: THEME.headFontFace, fontSize: 36, bold: true, valign: 'middle' });
txt(s, `Error del huerto con el modelo elegido solo con 2022–2025. El MAE bajó de ${dec(ciego.B0.MAE)} a ${dec(ciego.M2.MAE)} t/ha.`, { x: 8.45, y: 3.0, w: 4.0, h: 1.05, fontSize: 14 });
card(s, { x: 8.15, y: 4.4, w: 4.58, h: 2.0, name: 'lectura por anio' });
txt(s, 'Lectura', { x: 8.45, y: 4.6, w: 4.0, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
txt(s, 'Gana 4 de 5 años: los años en que el promedio erraba entre 16 y 25 %. Solo pierde en 2025, cuando el promedio ya acertaba (5,7 %).', { x: 8.45, y: 5.05, w: 4.0, h: 1.25, fontSize: 14 });
notas(s, 'El modelo aporta justo cuando el promedio falla', `Por temporada, el modelo gana en cuatro de cinco años: justo cuando el promedio erraba entre 16 y 25 %. Solo pierde en 2025, cuando el promedio ya acertaba. Y en la prueba ciega de 2026, elegido sin mirar ese año, bajó el error del huerto de ${dec(ciego.B0['Error_huerto_%'])} a ${dec(ciego.M2['Error_huerto_%'])} %.`);

// ---------- 11. Baseline y fecha de corte ----------
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Evaluación' });
s.addText('Baseline y momento del pronóstico', { placeholder: 'title' });
s.addChart(pres.charts.BAR, [
  { name: 'Promedio histórico (baseline)', labels: ['31 dic', '31 ene', '28 feb'], values: [17.4, 17.4, 17.4] },
  { name: 'Modelo mixto', labels: ['31 dic', '31 ene', '28 feb'], values: [10.3, 9.6, 9.7] },
], {
  x: M, y: 1.4, w: 6.9, h: 5.0, barDir: 'col', barGrouping: 'clustered', barGapWidthPct: 70, ...AXIS,
  chartColors: [HEX.accent5, HEX.accent1], showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: '0.0',
  valAxisMinVal: 0, valAxisMaxVal: 20, valAxisHidden: true, valGridLine: { style: 'none' }, showLegend: true, legendPos: 'b',
  showTitle: true, title: 'Error del total del huerto (%) según fecha de corte',
});
const bJusto = ROB.baseline_justo.find((r) => r.modelo.startsWith('B0b'));
card(s, { x: 7.85, y: 1.45, w: 4.88, h: 2.75, name: 'baseline' });
txt(s, 'Baseline: promedio histórico del cuartel', { x: 8.15, y: 1.65, w: 4.3, h: 0.4, valign: 'top', fontSize: 17, bold: true, color: C.text2 });
txt(s, `Es la regla que hoy usa el huerto: si el modelo no le gana, no aporta. Con toda la historia desde 2019 el baseline da ${dec(bJusto['Error_huerto_%'])} %, así que la comparación no favorece al modelo.`, { x: 8.15, y: 2.15, w: 4.3, h: 1.95, fontSize: 14 });
card(s, { x: 7.85, y: 4.4, w: 4.88, h: 2.0, name: 'corte' });
txt(s, 'Corte propuesto: 31 de enero', { x: 8.15, y: 4.6, w: 4.3, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
txt(s, 'En diciembre el modelo ya supera al baseline; en enero logra su mejor error, dos a tres meses antes de la cosecha.', { x: 8.15, y: 5.05, w: 4.3, h: 1.25, fontSize: 14 });
notas(s, 'Baseline y momento del pronóstico', `El baseline es la regla actual, el promedio del cuartel, y es justo: con toda la historia desde 2019 da ${dec(bJusto['Error_huerto_%'])} %. En diciembre el modelo ya baja el error a 10,3 % y en enero a 9,6; febrero no mejora. Por eso el corte es el 31 de enero.`);

// ---------- 12. Implementación ----------
pres.addSection({ title: 'Implementación' });
s = pres.addSlide({ masterName: 'CONTENIDO', sectionTitle: 'Implementación' });
s.addText('Del satélite a la decisión de cosecha', { placeholder: 'title' });
const flujo = [
  ['Datos', 'Canopia satelital diaria, cosecha por cuartel, clima INIA'],
  ['Preparación', 'Suavizado 7 días, promedio al corte, panel por cuartel'],
  ['Modelo', 'Modelo mixto reentrenado cada año tras la cosecha'],
  ['Resultado', 't/ha ± intervalo, kilos, bins y rango de temporada'],
  ['Usuario', 'Administrador y jefe técnico, vía tablero Streamlit'],
  ['Decisión', 'Bins, cuadrillas, ventas y cuarteles bajo equilibrio'],
];
const fw = 1.85, fg = 0.18;
flujo.forEach(([t, d], i) => {
  const x = M + i * (fw + fg);
  const fill = i === 5 ? C.accent2 : C.text2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: fw, h: 0.6, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: `paso ${t}` });
  txt(s, t, { x, y: 1.55, w: fw, h: 0.6, align: 'center', valign: 'middle', fontSize: 16, bold: true, color: C.background1 });
  card(s, { x, y: 2.3, w: fw, h: 1.3, name: `detalle ${t}` });
  txt(s, d, { x: x + 0.15, y: 2.42, w: fw - 0.3, h: 1.1, fontSize: 13 });
});
const frec = [
  ['Diciembre', 'Pronóstico preliminar para reservar bins y cuadrillas'],
  ['31 de enero', 'Pronóstico oficial: contratos de venta y presupuesto de cosecha'],
  ['Mayo', 'Se compara con la cosecha real, se mide el error y se reentrena'],
];
txt(s, 'Frecuencia de uso', { x: M, y: 3.9, w: 6, h: 0.4, fontFace: THEME.headFontFace, fontSize: 20, bold: true, color: C.text2 });
frec.forEach(([f, d], i) => {
  const x = M + i * 4.1;
  badge(s, x, 4.45, String(i + 1), C.accent1);
  txt(s, f, { x: x + 0.65, y: 4.45, w: 3.2, h: 0.4, fontSize: 16, bold: true, color: C.text2 });
  txt(s, d, { x: x + 0.65, y: 4.9, w: 3.2, h: 0.8, fontSize: 14 });
});
card(s, { x: M, y: 5.9, w: W - 2 * M, h: 0.8, name: 'monitoreo' });
txt(s, [{ text: 'Monitoreo: ', options: { bold: true, color: HEX.dk2 } }, { text: 'si el error del huerto de una temporada supera 15 %, se revisan las variables y se recalibra el modelo antes del siguiente pronóstico.' }], { x: M + 0.3, y: 5.9, w: W - 2 * M - 0.6, h: 0.8, valign: 'middle', fontSize: 15 });
notas(s, 'Del satélite a la decisión de cosecha', `La solución recibe canopia, cosecha y clima, y produce el pronóstico por cuartel. Lo usan el administrador y el jefe técnico en un tablero. En diciembre se reservan bins y cuadrillas, en enero se cierran ventas y presupuesto, y en mayo se mide el error y se reentrena.`);

// ---------- 13. Próximos pasos ----------
s = pres.addSlide({ masterName: 'PORTADA', sectionTitle: 'Implementación' });
s.addText('Próximos pasos', { placeholder: 'title' });
s.addText('Lo que falta para pasar de propuesta a solución en la entrega final', { placeholder: 'body' });
const pasos = [
  'Confirmar el mapeo de polígonos satelitales e incorporar los cuarteles sin canopia',
  'Ajustar el modelo final: variables de canopia invernal y de crecimiento, análisis de residuos',
  'Construir intervalos de predicción y medir su cobertura con bootstrap',
  'Confirmar que la mejora no es ruido (criterio 4) y probar el pronóstico del 31 ene 2027',
  'Integrar el pronóstico con el punto de equilibrio y escenarios de precio',
  'Prototipo del tablero en Streamlit para el equipo del huerto',
];
pasos.forEach((p, i) => {
  const y = 4.55 + i * 0.42;
  txt(s, `${i + 1}`, { x: M, y, w: 0.4, h: 0.4, fontSize: 16, bold: true, color: C.accent3, valign: 'middle' });
  txt(s, p, { x: M + 0.45, y, w: 11.5, h: 0.4, fontSize: 16, color: C.background1, valign: 'middle' });
});
s.addShape(pres.shapes.OVAL, { x: 10.6, y: 1.5, w: 2.1, h: 2.1, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: 'kiwi piel cierre' });
s.addShape(pres.shapes.OVAL, { x: 10.72, y: 1.62, w: 1.86, h: 1.86, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: 'kiwi pulpa cierre' });
s.addShape(pres.shapes.OVAL, { x: 11.37, y: 2.27, w: 0.56, h: 0.56, fill: { color: 'F4F7EC' }, line: { color: 'F4F7EC' }, objectName: 'kiwi centro cierre' });
notas(s, 'Próximos pasos', `Para la entrega final queda confirmar los polígonos, ajustar el modelo, construir intervalos, confirmar que la mejora no es ruido, integrar el punto de equilibrio y prototipar el tablero. En síntesis: un modelo mixto con canopia satelital, validado a ciegas contra la práctica actual. Gracias.`);

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  const palabras = escribirGuion();
  console.log('OK', OUT, `| guion: ${palabras} palabras`);
})();
