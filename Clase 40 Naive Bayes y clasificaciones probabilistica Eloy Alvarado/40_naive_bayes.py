"""S40. Naive Bayes y clasificación probabilística.
Ejecutar los tramos 00-10 en orden, en una misma sesión.
Guardar este archivo, el R y spam_s40.csv juntos, o conservar la carpeta clase.
Dependencias: numpy, pandas, matplotlib, scikit-learn. No descarga datos.
"""
# %% 00. Entorno y funciones
RUTA_MATERIAL = ""  # Opcional: carpeta que contiene spam_s40.csv.
from pathlib import Path
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import roc_auc_score

def buscar_datos():
    relativa = Path('sesiones/40_2026-10-01_naive_bayes/clase')
    inicios = [Path(RUTA_MATERIAL)] if RUTA_MATERIAL else [Path(globals().get('__file__', '.')).resolve(), Path.cwd()]
    for inicio in inicios:
        base = inicio.parent if inicio.is_file() else inicio
        for padre in ([base] if RUTA_MATERIAL else [base, *base.parents]):
            for carpeta in [padre, padre/'datos', padre/'clase'/'datos', padre/relativa/'datos']:
                if (carpeta/'spam_s40.csv').is_file(): return carpeta.resolve()
    raise FileNotFoundError('Guardar spam_s40.csv junto al código o completar RUTA_MATERIAL.')

DATOS = buscar_datos()
SALIDA = Path(os.environ.get('DIPDS_RESULTADOS', str(DATOS.parent/'resultados' if DATOS.name == 'datos' else DATOS/'resultados')))/'python'
SALIDA.mkdir(parents=True, exist_ok=True)
MOSTRAR = os.environ.get('DIPDS_MOSTRAR_FIGURAS', '1' if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules else '0') == '1'
AZUL, NARANJA, VERDE, TINTA = '#4067A0', '#C66A25', '#247D72', '#104D64'
plt.rcParams.update({'font.size':11, 'figure.facecolor':'none', 'axes.facecolor':'none', 'axes.spines.top':False, 'axes.spines.right':False, 'text.color':TINTA, 'axes.labelcolor':TINTA})

def tabla(nombre, x):
    x.to_csv(SALIDA/f'{nombre}.csv', index=False, float_format='%.15g')
    print('\n'+nombre); print(x.head(12).round(5).to_string(index=False))
    if len(x)>12: print(f'Mostradas 12 de {len(x)} filas. El CSV contiene la tabla completa.')

def figura(nombre, fig):
    fig.tight_layout()
    fig.savefig(SALIDA/f'{nombre}.pdf', bbox_inches='tight', transparent=True)
    if MOSTRAR: plt.show()
    plt.close(fig)

def posterior(log_conjunta):
    z = log_conjunta - np.max(log_conjunta, axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

def medir(y, p, t=.5, logp=None):
    # Logaritmos estables evitan restar de uno probabilidades casi unitarias.
    q = np.clip(p, 1e-15, 1-1e-15)
    if logp is None: logp=np.column_stack([np.log1p(-q),np.log(q)])
    perdida=-np.maximum(logp[np.arange(len(y)),y],np.log(1e-15)).mean()
    pred = p >= t
    vn, fp = np.sum((y==0)&~pred), np.sum((y==0)&pred)
    fn, vp = np.sum((y==1)&~pred), np.sum((y==1)&pred)
    return dict(log_loss=perdida,
                brier=np.mean((y-p)**2), auc=roc_auc_score(y,logp[:,1]-logp[:,0]),
                accuracy=np.mean(pred==y), VN=vn, FP=fp, FN=fn, VP=vp,
                costo_1000=1000*(5*fp+fn)/len(y))

# %% 01. Bayes con una señal y cambio de prevalencia
pi = np.array([.20, .02])
oferta_spam, oferta_legitimo = 16/20, 8/80
p = pi*oferta_spam/(pi*oferta_spam+(1-pi)*oferta_legitimo)
tabla('01_bayes', pd.DataFrame({'prior_spam':pi, 'posterior_oferta':p}))
fig, ax = plt.subplots(figsize=(7,3))
g = np.linspace(.001,.999,300)
ax.plot(g, g*oferta_spam/(g*oferta_spam+(1-g)*oferta_legitimo), color=AZUL)
ax.scatter(pi,p,color=NARANJA,zorder=3)
ax.set(xlabel='Probabilidad previa de spam', ylabel='P(spam | oferta)', ylim=(0,1), xlim=(0,1))
figura('01_prior',fig)

# %% 02. Bernoulli: presencia, ausencia y suavizado
# Ejemplo construido: 80 correos legítimos y 20 spam; dos indicadores.
# Con alfa=1: oferta (8+1)/82 y (16+1)/22; ausencia de enlace
# ((80-16)+1)/82 y ((20-14)+1)/22. Primero el cálculo de la diapositiva.
s_legitimo = (80/100)*(9/82)*(65/82)
s_spam = (20/100)*(17/22)*(7/22)
p_spam = s_spam/(s_legitimo+s_spam)
print("Spam para oferta sin enlace, con suavizado:", p_spam)
# La tabla siguiente automatiza el mismo cálculo, con y sin suavizado.
# theta: probabilidad de presencia; fila 0 legítimo y fila 1 spam.
n = np.array([80.,20.])
conteos = np.array([[8.,16.],[16.,14.]])  # columnas: oferta, enlace
x_nuevo = np.array([[1.,0.]])
filas = []
for alfa in [0.,1.]:
    theta = (conteos+alfa)/(n[:,None]+2*alfa)
    logc = np.log(n/n.sum()) + (x_nuevo*np.log(theta)+(1-x_nuevo)*np.log1p(-theta)).sum(axis=1)
    prob = posterior(logc[None,:])[0,1]
    filas.append(dict(alfa=alfa, theta_oferta_0=theta[0,0],theta_enlace_0=theta[0,1],theta_oferta_1=theta[1,0],theta_enlace_1=theta[1,1],posterior=prob))
tabla('02_bernoulli', pd.DataFrame(filas))

# %% 03. Duplicación de evidencia
# La variable duplicada es idéntica: observarla otra vez no añade información.
copias = np.array([1,2,4])
# Otro ejemplo: 100 correos por clase; señal en 80 spam y 20 legítimos.
prior_copia = 100/200
senal_spam, senal_legitimo = 80/100, 20/100
s_spam_copia = prior_copia*senal_spam**copias
s_legitimo_copia = (1-prior_copia)*senal_legitimo**copias
prob_copias = s_spam_copia/(s_spam_copia+s_legitimo_copia)
tabla('03_duplicacion',pd.DataFrame({'copias':copias,'posterior_nb':prob_copias,'posterior_correcto':.8}))
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(copias,prob_copias,'o-',color=NARANJA,label='NB: trata las copias como independientes')
ax.axhline(.8,color=AZUL,label='Información disponible: la misma señal')
ax.set(xlabel='Número de copias de la señal',ylabel='Probabilidad de clase 1',ylim=(.65,1.02),xticks=copias)
ax.legend(loc='lower right',fontsize=9); figura('03_duplicacion',fig)

# %% 04. GaussianNB y MultinomialNB: cálculos ilustrativos
z=np.linspace(-3,5,400)
f0=np.exp(-z*z/2)/np.sqrt(2*np.pi); f1=np.exp(-(z-2)**2/2)/np.sqrt(2*np.pi)
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(z,.8*f0,color=AZUL,label='Clase 0: 0.8 f0(x)')
ax.plot(z,.2*f1,color=NARANJA,label='Clase 1: 0.2 f1(x)')
ax.axvline(1.5,color=TINTA,linestyle=':'); ax.set(xlabel='x',ylabel='Densidad ponderada'); ax.legend()
figura('04_gaussiana',fig)
# Medición ficticia x=1.5; medias 0 y 2, desviaciones estándar 1.
densidad_0 = np.exp(-1.5**2/2)/np.sqrt(2*np.pi)
densidad_1 = np.exp(-(1.5-2)**2/2)/np.sqrt(2*np.pi)
s0_gauss, s1_gauss = .8*densidad_0, .2*densidad_1
tabla('04_gaussiana',pd.DataFrame({'x':[1.5],'densidad_0':[densidad_0],
    'densidad_1':[densidad_1],'producto_0':[s0_gauss],'producto_1':[s1_gauss],
    'posterior':[s1_gauss/(s0_gauss+s1_gauss)]}))
cuentas=np.array([[0.,4.,2.],[4.,0.,2.]]) # oferta, reunion, hoy
theta_m=(cuentas+1)/(cuentas.sum(axis=1,keepdims=True)+3)
mensajes=np.array([[1.,0.,1.],[2.,0.,1.]])
pm=posterior(np.log(.5)+mensajes@np.log(theta_m).T)[:,1]
tabla('04_multinomial',pd.DataFrame({'oferta':mensajes[:,0],'reunion':mensajes[:,1],'hoy':mensajes[:,2],'posterior':pm}))

# %% 05. Spambase y protocolo
datos=pd.read_csv(DATOS/'spam_s40.csv')
variables=[c for c in datos if c.startswith('word_freq_') or c.startswith('char_freq_')]
assert len(variables)==54
assert not datos.isna().any().any()
ent=datos[datos.conjunto=='entrenamiento'].copy()
prueba=datos[datos.conjunto=='prueba'].copy()
assert set(ent.grupo).isdisjoint(prueba.grupo)
assert ent.groupby('grupo').fold.nunique().max()==1
X=ent[variables].to_numpy(); y=ent.spam.to_numpy(); folds=ent.fold.to_numpy()
# Las mismas 54 frecuencias se representan por presencia o log(1+x).
# Ningún identificador, etiqueta, fold o medida de mayúsculas entra al modelo.
RECETAS=[('FRECUENCIA','prior',0.),('BERN_05','bern',.5),('BERN_1','bern',1.),('BERN_2','bern',2.),('GAUSS_LOG','gauss',1e-9)]
NOMBRES={'FRECUENCIA':'Frecuencia de spam','BERN_05':'Bernoulli: alfa 0.5','BERN_1':'Bernoulli: alfa 1','BERN_2':'Bernoulli: alfa 2','GAUSS_LOG':'Gaussiana: log(1+x)'}

def ajustar_predecir(xa,ya,xv,tipo,valor):
    if tipo=='prior':
        p=np.full(len(xv),ya.mean())
        return p,None,np.column_stack([np.log1p(-p),np.log(p)])
    if tipo=='bern':
        modelo=BernoulliNB(alpha=valor,binarize=0.0).fit(xa,ya)
        lp=modelo.predict_log_proba(xv)
        return np.exp(lp[:,1]),modelo,lp
    modelo=GaussianNB(var_smoothing=valor).fit(np.log1p(xa),ya)
    lp=modelo.predict_log_proba(np.log1p(xv))
    return np.exp(lp[:,1]),modelo,lp

tabla('05_particion', datos.groupby(['conjunto','spam']).size().reset_index(name='n'))

# %% 06. Predicciones fuera de pliegue y comparación
oof={}; logs_oof={}; comparacion=[]
for clave,tipo,valor in RECETAS:
    pred=np.empty(len(y)); lp=np.empty((len(y),2))
    for k in sorted(np.unique(folds)):
        val=folds==k
        pred[val],_,lp[val]=ajustar_predecir(X[~val],y[~val],X[val],tipo,valor)
    oof[clave]=pred
    logs_oof[clave]=lp
    comparacion.append(dict(modelo=clave,**medir(y,pred,logp=lp)))
comparacion=pd.DataFrame(comparacion)
tabla('06_comparacion_oof',comparacion)
tabla('06_predicciones_oof',pd.DataFrame({'id':ent.id.to_numpy(),'y':y,'fold':folds,**oof}))
ganador=comparacion.loc[comparacion.log_loss.idxmin(),'modelo']
print('Selección por log loss OOF:',NOMBRES[ganador])
fig,ax=plt.subplots(figsize=(8,3.5))
ax.barh([NOMBRES[c] for c in comparacion.modelo],comparacion.log_loss,color=[VERDE if c==ganador else AZUL for c in comparacion.modelo])
ax.invert_yaxis(); ax.set(xlabel='Log loss OOF (menor es mejor)'); figura('06_comparacion',fig)

# %% 07. Calibración descriptiva de probabilidades OOF
cal=[]
for clave in ['BERN_1','GAUSS_LOG']:
    pred=oof[clave]; bins=np.minimum((pred*10).astype(int),9)
    for b in sorted(np.unique(bins)):
        m=bins==b
        cal.append(dict(modelo=clave,bin=b,n=int(m.sum()),prob_media=pred[m].mean(),fraccion_spam=y[m].mean()))
cal=pd.DataFrame(cal); tabla('07_calibracion',cal)
fig,axes=plt.subplots(1,2,figsize=(9,3.7),sharex=True,sharey=True)
for ax,clave in zip(axes,['BERN_1','GAUSS_LOG']):
    d=cal[cal.modelo==clave]
    ax.plot([0,1],[0,1],':',color=TINTA)
    ax.scatter(d.prob_media,d.fraccion_spam,s=25+100*d.n/d.n.max(),color=AZUL)
    ax.set(title=NOMBRES[clave],xlabel='Probabilidad media',xlim=(-.03,1.03),ylim=(-.03,1.03))
axes[0].set_ylabel('Fracción observada de spam'); figura('07_calibracion',fig)

# %% 08. Probabilidades por clase para las señales de un correo
# Caso elegido por id entre entrenamiento, sin consultar su etiqueta.
modelo_bern=BernoulliNB(alpha=1,binarize=0.).fit(X,y)
indice=int(np.argmin(ent.id.to_numpy()))
b=(X[indice]>0).astype(float)
th=np.exp(modelo_bern.feature_log_prob_)
aportes=b*np.log(th[1]/th[0])+(1-b)*np.log((1-th[1])/(1-th[0]))
prior_log=np.log(y.mean()/(1-y.mean()))
# Contar el estado observado: presencia si b=1, ausencia si b=0.
# Alfa=1 agrega una unidad al conteo y dos al denominador por clase.
coincide = (X>0)==b.astype(bool)
n0, n1 = int(np.sum(y==0)), int(np.sum(y==1))
cuenta0, cuenta1 = coincide[y==0].sum(axis=0), coincide[y==1].sum(axis=0)
prob0, prob1 = (cuenta0+1)/(n0+2), (cuenta1+1)/(n1+2)
explicacion=pd.DataFrame({'variable':variables,'presente':b.astype(int),
    'conteo_legitimo':cuenta0,'n_legitimo':n0,'conteo_spam':cuenta1,'n_spam':n1,
    'prob_legitimo':prob0,'prob_spam':prob1,'aporte_log_odds':aportes})
tabla('08_aportes',explicacion)
tabla('08_correo',pd.DataFrame({'id':[ent.id.iloc[indice]],'log_odds_prior':[prior_log],'suma_aportes':[aportes.sum()],'posterior':[modelo_bern.predict_proba(X[indice:indice+1])[0,1]]}))
# El detalle logarítmico permite reconstruir la probabilidad completa.
# Para leer el gráfico basta comparar las probabilidades dentro de cada clase.
top=explicacion.iloc[np.argsort(-np.abs(aportes),kind='stable')[:8]].sort_values('aporte_log_odds')
fig,ax=plt.subplots(figsize=(8,3.8))
labels=[c.replace('word_freq_','').replace('char_freq_','carácter ') + (' (presente)' if p else ' (ausente)') for c,p in zip(top.variable,top.presente)]
pos=np.arange(len(top))
ax.barh(pos-.18,top.prob_legitimo,height=.34,color=AZUL,label='Dentro de legítimos')
ax.barh(pos+.18,top.prob_spam,height=.34,color=NARANJA,label='Dentro de spam')
ax.set(yticks=pos,yticklabels=labels,xlabel='P(señal observada | clase)',xlim=(0,1))
ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=10,frameon=False)
figura('08_aportes',fig)

# %% 09. Regla de decisión y comprobación histórica
# Selección secuencial: primero log loss; después costo 5 FP + FN.
# Desempate de modelos: orden de RECETAS; de umbrales: el mayor.
umbrales=np.arange(1,20)/20
costos=pd.DataFrame([dict(umbral=t,**medir(y,oof[ganador],t,logs_oof[ganador])) for t in umbrales])
elegido=costos.sort_values(['costo_1000','umbral'],ascending=[True,False],kind='stable').iloc[0]
t=float(elegido.umbral)
tabla('09_umbrales',costos)
fig,ax=plt.subplots(figsize=(7,3))
ax.plot(costos.umbral,costos.costo_1000,'o-',color=AZUL)
ax.axvline(t,color=NARANJA,linestyle='--',label=f'Umbral elegido: {t:.2f}')
ax.set(xlabel='Umbral',ylabel='Costo OOF por 1000 correos'); ax.legend(); figura('09_umbrales',fig)
_,tipo,valor=next(r for r in RECETAS if r[0]==ganador)
pp,modelo,logpp=ajustar_predecir(X,y,prueba[variables].to_numpy(),tipo,valor)
yp=prueba.spam.to_numpy()
resultado=pd.DataFrame([dict(modelo=ganador,umbral=t,**medir(yp,pp,t,logpp)),dict(modelo='SIEMPRE_NO_SPAM',umbral=1.,**medir(yp,np.full(len(yp),y.mean()),1.))])
tabla('09_comprobacion_historica',resultado)
tabla('09_predicciones_historicas',pd.DataFrame({'id':prueba.id.to_numpy(),'y':yp,'p':pp,'pred':(pp>=t).astype(int)}))
print('Los 921 correos ya se utilizaron al comparar clasificadores: esta es una comprobación retrospectiva, no una nueva prueba independiente.')

# %% 10. Síntesis y versiones
print('Prior de clase, verosimilitud y normalización producen la probabilidad.')
print('La representación determina la variante de Naive Bayes.')
print('Suavizar evita ceros; la independencia y la calibración requieren otra evaluación.')
print('Python',sys.version.split()[0])
import sklearn
print('numpy',np.__version__,'pandas',pd.__version__,'scikit-learn',sklearn.__version__)
