"""S38: evaluación y selección en regresión y clasificación (Python).
Un único archivo completo; las secciones 00-12 coinciden con la versión R.
VS Code: pulsar Run Cell en cada tramo # %%; ver la salida y luego continuar.
Elegir el intérprete Python con los paquetes instalados; conservar el mismo kernel.
Guardar este script, el archivo R y los dos CSV en una misma carpeta. El PDF no es necesario para ejecutar.
Usar el editor o entorno de preferencia y ejecutar los tramos en orden desde 00.
Instalación previa: python -m pip install numpy pandas matplotlib scikit-learn ipykernel
Datos locales y particiones fijas. No ejecuta R ni importa otro script de la clase.
"""
# %% 00. Entorno y funciones comunes
# Preparar una vez. Este tramo define funciones; no ajusta modelos.
# Dejar vacío para detectar el material. Si hace falta, cambiar SOLO esta línea:
# RUTA_MATERIAL = r"C:/mis clases/S38"  # carpeta de los scripts y CSV
RUTA_MATERIAL = ""

from pathlib import Path
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (log_loss, brier_score_loss, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve)

# Prioridad: ruta indicada, ubicación del script, directorio actual y workspace.
# No cambia el directorio de trabajo; busca datos junto a la presentación.
def carpeta_clase():
    relativa = Path('sesiones/38_2026-09-24_evaluacion_seleccion/clase')

    def buscar(inicio, subir=True):
        if not inicio: return None
        base = Path(inicio).expanduser().resolve()
        if not base.exists(): return None
        if base.is_file(): base = base.parent
        for padre in ([base, *base.parents] if subir else [base]):
            for candidata in [padre, padre/'clase', padre/relativa]:
                for datos in [candidata, candidata/'datos']:
                    archivos = [datos/f for f in ['hormigon_s38.csv', 'spam_s38.csv']]
                    if all(f.is_file() for f in archivos):
                        return candidata.resolve(), datos.resolve()
                    if any(f.is_file() for f in archivos):
                        faltan = ', '.join(f.name for f in archivos if not f.is_file())
                        raise FileNotFoundError(f'Material incompleto en {datos}: faltan {faltan}. Guardar ambos CSV juntos.')
                if (candidata/'DipDS_2026_S38.pdf').is_file():
                    raise FileNotFoundError(f'Material incompleto en {candidata}: faltan hormigon_s38.csv y spam_s38.csv.')
        return None

    manual = RUTA_MATERIAL.strip()
    if manual:
        encontrada = buscar(manual, subir=False)
        if encontrada is not None: return encontrada
        raise FileNotFoundError(
            f'RUTA_MATERIAL no contiene el material completo: {manual}\n'
            'Indicar la carpeta con hormigon_s38.csv y spam_s38.csv.')
    for inicio in [globals().get('__file__'), Path.cwd(), os.environ.get('DIPDS_WORKSPACE')]:
        encontrada = buscar(inicio)
        if encontrada is not None: return encontrada
    raise FileNotFoundError(
        f'No se encontró el material desde {Path.cwd()}.\n'
        'Trabajar desde la carpeta de los scripts y CSV, o completar RUTA_MATERIAL en el tramo 00.\n'
        'Conservar hormigon_s38.csv y spam_s38.csv junto a los scripts.')

SESION, DATOS = carpeta_clase()
RAIZ_RESULTADOS = Path(os.environ.get('DIPDS_RESULTADOS', SESION/'resultados'))/'python'
RAIZ_RESULTADOS.mkdir(parents=True, exist_ok=True)
print('Material:', SESION)
print('Datos:', DATOS)
print('Resultados Python:', RAIZ_RESULTADOS.resolve())
MOSTRAR = os.environ.get('DIPDS_MOSTRAR_FIGURAS', '1' if hasattr(sys,'ps1') or 'ipykernel' in sys.modules else '0') == '1'
AZUL,NARANJA,VERDE,TINTA='#4067A0','#C66A25','#247D72','#104D64'
plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','figure.facecolor':'none',
    'axes.facecolor':'none','savefig.transparent':True,'axes.spines.top':False,
    'axes.spines.right':False,'text.color':TINTA,'axes.labelcolor':TINTA})

# Claves internas de los CSV; las salidas para leer usan nombres de modelos.
def nombre_modelo(clave):
    nombres={'MEDIA':'Media de entrenamiento','LINEAL':'Regresión lineal',
        'FRECUENCIA':'Frecuencia de spam','LOGISTICA_C1':'Logística (C=1)',
        'BOSQUE_8':'Bosque (profundidad 8)','SIEMPRE_NO_SPAM':'Siempre no spam'}
    return 'Cuadrática: alfa='+clave[3:] if clave.startswith('Q_A') else nombres.get(clave,clave)

def mostrar(tabla):
    # Copia para leer; los identificadores guardados en CSV permanecen estables.
    t=tabla.copy()
    if 'modelo' in t: t['modelo']=t['modelo'].map(nombre_modelo)
    print(t.round(3).to_string(index=False))

def figura(fig,nombre):
    fig.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(salida/f'{nombre}.{ext}',transparent=True,bbox_inches='tight',dpi=160)
    if MOSTRAR:
        if 'inline' in plt.get_backend().lower():
            plt.show()
            plt.close(fig)
        else:
            plt.show(block=False)
            plt.pause(.05)
    else: plt.close(fig)

def metricas(real,predicha):
    real=np.asarray(real); predicha=np.asarray(predicha)
    e=real-predicha
    return {'RMSE':float(np.sqrt(np.mean(e**2))), 'MAE':float(np.mean(abs(e))),
            'R2':float(1-np.sum(e**2)/np.sum((real-real.mean())**2))}

def probabilisticas(y,p):
    return {'log_loss':log_loss(y,p,labels=[0,1]),'Brier':brier_score_loss(y,p),
            'ROC_AUC':roc_auc_score(y,p),'AP':average_precision_score(y,p)}

def decision(y,p,t):
    vn,fp,fn,vp=confusion_matrix(y,np.asarray(p)>=t,labels=[0,1]).ravel()
    n=vn+fp+fn+vp
    precision=vp/(vp+fp) if vp+fp else 0.
    recall=vp/(vp+fn)
    return {'umbral':float(t),'VN':int(vn),'FP':int(fp),'FN':int(fn),'VP':int(vp),
        'accuracy':(vn+vp)/n,'precision':precision,'recall':recall,
        'F1':2*vp/(2*vp+fp+fn),'costo':(5*fp+fn)/n}

def matriz(ax,row,title):
    values=np.array([[row['VN'],row['FP']],[row['FN'],row['VP']]],dtype=int)
    # Contornos sin fondo opaco; filas reales y columnas predichas, orden 0,1.
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor=NARANJA if i!=j else AZUL,alpha=.22))
            ax.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=20,color=TINTA)
    ax.set(xticks=[0,1],xticklabels=['No spam','Spam'],yticks=[0,1],yticklabels=['No spam','Spam'],
        xlim=(-.5,1.5),ylim=(1.5,-.5),xlabel='Predicha',ylabel='Real',title=title)
    ax.set_aspect('equal')

# %% 01. Regresión: La pregunta determina la partición
# Observar: tamaños y composiciones separadas. La prueba queda reservada.
salida = RAIZ_RESULTADOS/'hormigon'
salida.mkdir(parents=True,exist_ok=True)
datos = pd.read_csv(DATOS/'hormigon_s38.csv')
variables=['cemento','escoria','ceniza','agua','superplastificante',
           'agregado_grueso','agregado_fino','edad']
entrenamiento=datos.loc[datos.conjunto=='entrenamiento'].copy().reset_index(drop=True)
prueba=datos.loc[datos.conjunto=='prueba'].copy().reset_index(drop=True)
assert len(entrenamiento)==806 and len(prueba)==224
assert set(entrenamiento.grupo).isdisjoint(prueba.grupo)
assert entrenamiento.groupby('grupo').fold.nunique().eq(1).all()
assert set(entrenamiento.fold)==set(range(1,6)) and prueba.fold.eq(0).all()
print('Entrenamiento:',len(entrenamiento),'filas;',entrenamiento.grupo.nunique(),'composiciones')
print('Prueba:',len(prueba),'filas;',prueba.grupo.nunique(),'composiciones')
# id, grupo, conjunto y fold nunca se incluyen como predictores.
# grupo reúne mediciones con los mismos siete ingredientes, aunque cambie edad.
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].scatter(entrenamiento.edad,entrenamiento.resistencia,s=14,alpha=.5,c=AZUL)
axs[0].set(xscale='log',xlabel='Edad (días, escala log)',ylabel='Resistencia (MPa)',title='Solo entrenamiento')
axs[1].hist(entrenamiento.resistencia,bins=20,color=AZUL,alpha=.85)
axs[1].set(xlabel='Resistencia (MPa)',ylabel='Número de mediciones',title='La respuesta es continua')
figura(fig,'10_hormigon_datos')

# %% 02. Regresión: Pérdida, referencia y candidatos
# Observar: qué cambia entre las siete alternativas.
def candidatos():
    modelos={'MEDIA':DummyRegressor(strategy='mean'),
             'LINEAL':make_pipeline(StandardScaler(),LinearRegression())}
    for a in [.1,1,10,100,1000]:
        modelos[f'Q_A{a:g}']=make_pipeline(
            StandardScaler(), PolynomialFeatures(2,include_bias=False),
            StandardScaler(), Ridge(alpha=a,solver='svd'))
    return modelos

# Regla fijada: menor RMSE de todas las predicciones OOF.
# Empate numérico exacto: orden MEDIA, LINEAL y Q por alpha creciente.
# MAE y R2 se reportan, pero no cambian la regla después de ver los resultados.
modelos=candidatos()
print('Candidatos:',[nombre_modelo(m) for m in modelos])
print('RMSE y MAE usan MPa; R2 no tiene unidades.')

# %% 03. Regresión: Una predicción fuera de ajuste para cada observación
# Observar: una predicción OOF por fila y candidato; todavía sin prueba.
def seleccionar(datos_train,modelos):
    x=datos_train[variables]; y=datos_train.resistencia.to_numpy()
    folds=datos_train.fold.to_numpy()
    resumen=[]; por_fold=[]; predicciones=[]
    for orden,(nombre,modelo) in enumerate(modelos.items()):
        oof=np.full(len(y),np.nan); cobertura=np.zeros(len(y),dtype=int)
        sse_ajuste=0.; n_ajuste=0
        for k in range(1,6):
            valida=folds==k
            assert set(datos_train.loc[valida,'grupo']).isdisjoint(datos_train.loc[~valida,'grupo'])
            ajuste=clone(modelo).fit(x.loc[~valida],y[~valida])
            oof[valida]=ajuste.predict(x.loc[valida]); cobertura[valida]+=1
            pred_train=ajuste.predict(x.loc[~valida])
            sse_ajuste+=np.sum((y[~valida]-pred_train)**2); n_ajuste+=(~valida).sum()
            por_fold.append({'modelo':nombre,'fold':k,'n':int(valida.sum()),
                             **metricas(y[valida],oof[valida])})
            if nombre!='MEDIA':
                assert np.allclose(ajuste.steps[0][1].mean_,x.loc[~valida].mean())
        assert np.all(cobertura==1) and np.isfinite(oof).all()
        resumen.append({'modelo':nombre,'orden':orden,**metricas(y,oof),
                        'RMSE_ajuste':float(np.sqrt(sse_ajuste/n_ajuste))})
        predicciones.extend({'modelo':nombre,'id':int(datos_train.id.iloc[i]),
            'grupo':int(datos_train.grupo.iloc[i]),'fold':int(folds[i]),
            'real':float(y[i]),'predicha':float(oof[i])} for i in range(len(y)))
    ranking=pd.DataFrame(resumen).sort_values(['RMSE','orden']).reset_index(drop=True)
    return ranking,pd.DataFrame(por_fold),pd.DataFrame(predicciones)

ranking,por_fold,oof=seleccionar(entrenamiento,modelos)
ganador=ranking.modelo.iloc[0]
mostrar(ranking[['modelo','RMSE','MAE','R2','RMSE_ajuste']])
print('Elegido por CV:',nombre_modelo(ganador))

# %% 04a. Regresión: leer el ranking de validación
# Leer el ranking antes de avanzar a la curva.
fig,ax=plt.subplots(figsize=(7.5,4))
ax.scatter(ranking.RMSE,ranking.modelo.map(nombre_modelo),c=AZUL,s=65)
for _,r in ranking.iterrows(): ax.annotate(f'{r.RMSE:.2f}',(r.RMSE,nombre_modelo(r.modelo)),xytext=(7,0),textcoords='offset points',va='center')
ax.invert_yaxis(); ax.margins(x=.17); ax.grid(axis='x',alpha=.2)
ax.set(xlabel='RMSE OOF (MPa; menor es mejor)',title='Siete candidatos, los mismos cinco pliegues')
figura(fig,'11_ranking_hormigon')

# %% 04b. Regresión: comparar ajuste y validación
# Comparar los dos errores antes de avanzar al siguiente gráfico.
q=ranking[ranking.modelo.str.startswith('Q_')].copy()
q['alpha']=q.modelo.str.replace('Q_A','').astype(float); q=q.sort_values('alpha')
fig,ax=plt.subplots(figsize=(7.5,4))
ax.plot(q.alpha,q.RMSE_ajuste,'o-',c=VERDE,label='Ajuste: filas usadas para estimar')
ax.plot(q.alpha,q.RMSE,'o-',c=NARANJA,label='Validación: predicciones OOF')
ax.set(xscale='log',xlabel='alpha (mayor penalización hacia la derecha)',ylabel='RMSE (MPa)',title='Error de ajuste y error de validación')
ax.legend(frameon=False,fontsize=9)
figura(fig,'12_curva_validacion')

# %% 04c. Regresión: examinar diferencias entre pliegues
# Leer el signo de las diferencias; no son réplicas independientes.
mejores=ranking.modelo.head(2).tolist()
pares=por_fold.pivot(index='fold',columns='modelo',values='RMSE')
delta=pares[mejores[0]]-pares[mejores[1]]
fig,ax=plt.subplots(figsize=(7.5,3.8))
ax.bar(delta.index,delta,color=np.where(delta<0,AZUL,NARANJA)); ax.axhline(0,c=TINTA,lw=1)
ax.set(xticks=range(1,6),xlabel='Pliegue de composiciones excluidas',ylabel='Diferencia de RMSE (MPa)',title=f'{nombre_modelo(mejores[0])} menos {nombre_modelo(mejores[1])}')
figura(fig,'13_diferencias_hormigon')
# No tratamos estos cinco valores como observaciones independientes ni como un IC.
# También se muestra el ranking por MAE, sin reemplazar el criterio elegido.
print('Orden descriptivo por MAE:',ranking.sort_values('MAE').modelo.map(nombre_modelo).to_list())
print('Diferencia RMSE entre los dos primeros:',ranking.RMSE.iloc[1]-ranking.RMSE.iloc[0])

# %% 05. Regresión: Reajustar lo elegido y evaluar una vez
# Aplicar la elección cerrada; observar el error en datos reservados.
modelo_final=clone(modelos[ganador]).fit(entrenamiento[variables],entrenamiento.resistencia)
pred_prueba=modelo_final.predict(prueba[variables])
evaluacion={'modelo':ganador,**metricas(prueba.resistencia,pred_prueba)}
# Referencia predefinida: la media de entrenamiento, no la media de prueba.
base=DummyRegressor(strategy='mean').fit(entrenamiento[variables],entrenamiento.resistencia)
pred_base=base.predict(prueba[variables])
referencia={'modelo':'MEDIA',**metricas(prueba.resistencia,pred_base)}
print('Evaluación final:'); mostrar(pd.DataFrame([evaluacion]))
print('Referencia predefinida:'); mostrar(pd.DataFrame([referencia]))
print('No se evalúan los otros candidatos para volver a elegir con prueba.')

fig,axs=plt.subplots(1,2,figsize=(10,4))
axs[0].scatter(prueba.resistencia,pred_prueba,c=AZUL,s=20,alpha=.65)
lim=[min(prueba.resistencia.min(),pred_prueba.min())-2,max(prueba.resistencia.max(),pred_prueba.max())+2]
axs[0].plot(lim,lim,'--',c=NARANJA); axs[0].set(xlabel='Real (MPa)',ylabel='Predicha (MPa)',title='Predicciones en prueba',xlim=lim,ylim=lim)
axs[1].bar(['Media de\nentrenamiento',nombre_modelo(ganador).replace(': ', '\n')],[referencia['RMSE'],evaluacion['RMSE']],color=[NARANJA,AZUL])
axs[1].set(ylabel='RMSE (MPa)',title='Comparación con referencia fijada')
figura(fig,'14_prueba_hormigon')

# %% 06. Regresión: Diagnosticar y comunicar los límites
# Leer residuos y guardar el caso de regresión.
residuos=prueba.resistencia.to_numpy()-pred_prueba
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].scatter(pred_prueba,residuos,c=AZUL,s=20,alpha=.6); axs[0].axhline(0,c=NARANJA,ls='--')
axs[0].set(xlabel='Predicha (MPa)',ylabel='Real − predicha (MPa)',title='¿El promedio oculta un patrón?')
axs[1].scatter(prueba.edad,abs(residuos),c=AZUL,s=20,alpha=.6)
axs[1].set(xscale='log',xlabel='Edad (días, escala log)',ylabel='Error absoluto (MPa)',title='Errores según edad')
figura(fig,'15_diagnostico_hormigon')
# Estos gráficos describen prueba. Usarlos para rediseñar exige nueva evaluación.
# Una composición excluida no equivale a un laboratorio o lote independiente.
for nombre,d in [('comparacion_cv',ranking),('metricas_por_fold',por_fold),('predicciones_oof',oof),
                 ('evaluacion_final',pd.DataFrame([evaluacion])),('referencia',pd.DataFrame([referencia])),
                 ('predicciones_test',pd.DataFrame({'id':prueba.id,'grupo':prueba.grupo,
                     'real':prueba.resistencia,'predicha':pred_prueba,'referencia':pred_base,'residuo':residuos}))]:
    d.to_csv(salida/f'{nombre}.csv',index=False)
(salida/'versiones.json').write_text(json.dumps({'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},indent=2),encoding='utf-8')
print('Resultados guardados en:',salida)

# %% 07. Clasificación: Datos, referencia y protocolo fijado
# Observar: clases, candidatos y costos; todavía sin evaluar prueba.
salida = RAIZ_RESULTADOS/'spam'
salida.mkdir(parents=True,exist_ok=True)
datos=pd.read_csv(DATOS/'spam_s38.csv')
variables=[c for c in datos if c.startswith(('word_freq_','char_freq_','capital_run_'))]
assert len(variables)==57 and len(datos)==4601
tr=datos[datos.conjunto=='entrenamiento'].reset_index(drop=True)
te=datos[datos.conjunto=='prueba'].reset_index(drop=True)
assert len(tr)==3680 and len(te)==921
assert set(tr.grupo).isdisjoint(te.grupo)
assert tr.groupby('grupo').fold.nunique().eq(1).all()
assert set(tr.fold)==set(range(1,6)) and te.fold.eq(0).all()
x=tr[variables]; y=tr.spam.to_numpy()
modelos={'FRECUENCIA':DummyClassifier(strategy='prior'),
    'LOGISTICA_C1':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=3000,solver='lbfgs')),
    'BOSQUE_8':RandomForestClassifier(n_estimators=150,max_depth=8,min_samples_leaf=5,
                                      max_features='sqrt',random_state=20260924,n_jobs=1)}
umbrales=np.round(np.arange(.05,1.,.05),2)
print('Entrenamiento:',len(tr),'Prueba reservada:',len(te),'Spam en entrenamiento:',y.mean())
print('Primero: menor log loss OOF. Después: menor costo OOF del ganador.')
print('Costos hipotéticos: FP=5, FN=1. Empates: orden de modelos; mayor umbral.')
fig,ax=plt.subplots(figsize=(7,3.5))
ax.bar(['No spam (0)','Spam (1)'],np.bincount(y),color=[AZUL,NARANJA])
for i,n in enumerate(np.bincount(y)): ax.text(i,n+30,f'{n} ({n/len(y):.1%})',ha='center')
ax.set(ylabel='Correos',ylim=(0,2700),title='La referencia empieza por la frecuencia: entrenamiento')
figura(fig,'30_spam_frecuencia')

# %% 08. Clasificación: Comparar probabilidades fuera de pliegue
# Leer log loss OOF y el modelo elegido.
rows=[]; predictions=[]; folds=[]
for order,(name,recipe) in enumerate(modelos.items()):
    p=np.full(len(tr),np.nan); coverage=np.zeros(len(tr),dtype=int)
    for k in range(1,6):
        val=tr.fold.to_numpy()==k
        assert set(tr.loc[val,'grupo']).isdisjoint(tr.loc[~val,'grupo'])
        fitted=clone(recipe).fit(x.loc[~val],y[~val])
        p[val]=fitted.predict_proba(x.loc[val])[:,1]; coverage[val]+=1
        folds.append({'modelo':name,'fold':k,'n':int(val.sum()),**probabilisticas(y[val],p[val])})
    assert np.all(coverage==1) and np.isfinite(p).all()
    rows.append({'modelo':name,'orden':order,**probabilisticas(y,p)})
    predictions.append(pd.DataFrame({'modelo':name,'id':tr.id,'grupo':tr.grupo,
        'fold':tr.fold,'real':y,'probabilidad':p}))
ranking=pd.DataFrame(rows).sort_values(['log_loss','orden']).reset_index(drop=True)
oof=pd.concat(predictions,ignore_index=True)
ganador=ranking.modelo.iloc[0]
p=oof[oof.modelo==ganador].set_index('id').loc[tr.id,'probabilidad'].to_numpy()
mostrar(ranking); print('Modelo elegido:',nombre_modelo(ganador))
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
for ax,metric in zip(axs,['log_loss','Brier']):
    ax.barh(ranking.modelo.map(nombre_modelo),ranking[metric],color=[VERDE,AZUL,AZUL]); ax.invert_yaxis()
    ax.set(xlabel=metric+' (menor es mejor)')
    for j,v in enumerate(ranking[metric]): ax.text(v+.004,j,f'{v:.3f}',va='center',fontsize=11)
    ax.set_xlim(0,ranking[metric].max()*1.18)
figura(fig,'31_spam_modelos')

# %% 09a. Clasificación: leer ROC y precision-recall
# Leer los ejes y el ordenamiento antes de mirar calibración.
fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
metrics=probabilisticas(y,p)
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].plot(fpr,tpr,c=AZUL,lw=2); axs[0].plot([0,1],[0,1],'--',c='gray')
axs[0].set(xlabel='Tasa de falsos positivos',ylabel='Recall',title=f'ROC: AUC = {metrics["ROC_AUC"]:.3f}')
axs[1].plot(recall,precision,c=VERDE,lw=2); axs[1].axhline(y.mean(),c=NARANJA,ls='--',label='Frecuencia de spam')
axs[1].set(xlabel='Recall',ylabel='Precision',title=f'Precision-recall: AP = {metrics["AP"]:.3f}')
axs[1].legend(frameon=False,fontsize=10)
for ax in axs: ax.set(xlim=(0,1.02),ylim=(0,1.02))
figura(fig,'32_spam_roc_pr')
# %% 09b. Clasificación: comparar probabilidades y frecuencias
# Comparar probabilidad media y frecuencia observada.
bins=pd.qcut(p,q=5,duplicates='drop')
cal=pd.DataFrame({'p':p,'y':y,'bin':bins}).groupby('bin',observed=True).agg(
    media_probabilidad=('p','mean'),fraccion_spam=('y','mean'),n=('y','size')).reset_index(drop=True)
fig,axs=plt.subplots(1,2,figsize=(10,3.8))
axs[0].plot([0,1],[0,1],'--',c='gray',label='Calibración ideal')
axs[0].plot(cal.media_probabilidad,cal.fraccion_spam,'o-',c=AZUL,label='Cinco grupos por cuantiles')
axs[0].set(xlabel='Probabilidad media predicha',ylabel='Fracción observada de spam',xlim=(0,1),ylim=(0,1))
axs[0].legend(frameon=False,fontsize=9)
axs[1].hist(p,bins=np.linspace(0,1,11),color=AZUL,alpha=.8)
axs[1].set(xlabel='Probabilidad OOF de spam',ylabel='Número de correos',title='Dónde se concentran las predicciones')
figura(fig,'33_spam_calibracion')
print('Calibración descriptiva OOF:'); print(cal.to_string(index=False))

# %% 10a. Clasificación: elegir el umbral por costo
# Leer el umbral elegido antes de examinar sus errores.
costos=pd.DataFrame([decision(y,p,t) for t in umbrales])
eleccion=costos.sort_values(['costo','umbral'],ascending=[True,False]).iloc[0]
umbral=float(eleccion.umbral)
print('Umbral elegido:',umbral,'Costo OOF:',eleccion.costo)
fig,ax=plt.subplots(figsize=(8,3.8))
ax.plot(costos.umbral,1000*costos.costo,'o-',c=AZUL)
ax.axvline(.5,c='gray',ls='--',label='Umbral de referencia 0.50')
ax.axvline(umbral,c=NARANJA,ls='--',label=f'Elegido: {umbral:.2f}')
ax.set(xlabel='Umbral para marcar spam',ylabel='Costo por 1000 correos',title='Costo hipotético: 5 por FP y 1 por FN')
ax.legend(frameon=False,fontsize=10)
figura(fig,'34_spam_umbral')
# %% 10b. Clasificación: leer el intercambio entre FP y FN
# Leer FP y FN; la prueba se evalúa recién en el tramo 11.
fig,axs=plt.subplots(1,2,figsize=(9,3.8))
for ax,t in zip(axs,[.5,umbral]):
    row=decision(y,p,t)
    matriz(ax,row,f'OOF: t={t:.2f}\nCosto/1000={1000*row["costo"]:.1f}')
figura(fig,'35_spam_decisiones')
# El mínimo OOF sirve para elegir; no se presenta como evaluación independiente.
# Esta elección por etapas no busca el menor costo entre todas las parejas modelo/umbral.

# %% 11. Clasificación: Reajustar y evaluar la regla cerrada
# Evaluar una vez el modelo y umbral ya elegidos.
final_model=clone(modelos[ganador]).fit(x,y)
pt=final_model.predict_proba(te[variables])[:,1]
final={'modelo':ganador,**probabilisticas(te.spam,pt),**decision(te.spam,pt,umbral)}
reference={'modelo':'SIEMPRE_NO_SPAM',**decision(te.spam,np.zeros(len(te)),.5)}
print('Evaluación final:'); mostrar(pd.DataFrame([final]))
print('Referencia de decisión:'); mostrar(pd.DataFrame([reference]))
fig,axs=plt.subplots(1,2,figsize=(10,4))
matriz(axs[0],final,f'Prueba: bosque, t={umbral:.2f}')
axs[1].bar(['Siempre\nno spam','Regla\nelegida'],1000*np.array([reference['costo'],final['costo']]),color=[NARANJA,AZUL])
axs[1].set(ylabel='Costo por 1000 correos',title='Mismos costos fijados antes de comparar')
figura(fig,'36_spam_prueba')
protocol={'modelo':ganador,'umbral':umbral,'costo_FP':5,'costo_FN':1,
          'orden':'log loss OOF; costo OOF; reajuste; prueba',
          'limite':'Resultados del corpus histórico, no validación de un filtro actual.'}
for name,frame in [('comparacion_cv',ranking),('metricas_por_fold',pd.DataFrame(folds)),
    ('predicciones_oof',oof),('calibracion',cal),('costos_umbrales',costos),
    ('evaluacion_final',pd.DataFrame([final])),('referencia',pd.DataFrame([reference])),
    ('predicciones_test',pd.DataFrame({'id':te.id,'grupo':te.grupo,'real':te.spam,
        'probabilidad':pt,'predicha':(pt>=umbral).astype(int)}))]:
    frame.to_csv(salida/f'{name}.csv',index=False)
(salida/'regla_elegida.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('El resultado evalúa esta regla y esta separación. Cambiar costos o candidatos requiere otra selección.')

# %% 12. Resumen y versiones de esta ejecución
# Consultar dónde quedaron los archivos y qué versiones se usaron.
versiones = {'lenguaje':'Python','version':sys.version.split()[0],
    'ejecutable':sys.executable,'sklearn':sklearn.__version__,
    'numpy':np.__version__,'pandas':pd.__version__}
(RAIZ_RESULTADOS/'versiones.json').write_text(json.dumps(versiones,indent=2),encoding='utf-8')
print('\nPYTHON: ambos casos completos. Resultados en',RAIZ_RESULTADOS)
print('R usa el mismo protocolo con una implementación nativa; los bosques no son idénticos entre bibliotecas.')
