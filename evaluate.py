"""Walk-forward evaluation: true 1-week-ahead forecast (no same-week weather).
Run:  python evaluate.py   (needs dengu_weather_dataset.csv in the same folder)
Outputs: outputs/evaluation_table.csv, outputs/predictions.csv
"""
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import numpy as np, pandas as pd, tensorflow as tf, warnings; warnings.filterwarnings('ignore')
from tensorflow.keras import layers, Model
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('dengu_weather_dataset.csv',parse_dates=['start_date']).sort_values('start_date').reset_index(drop=True)
W=['temp_mean','rainfall_total','humidity_mean']; SEQ=8
df['lc']=np.log1p(df.dengue_cases); doy=df.start_date.dt.dayofyear
df['sin_w']=np.sin(2*np.pi*doy/365.25); df['cos_w']=np.cos(2*np.pi*doy/365.25)
# tabular features known at t-1 (true forecast: no concurrent weather)
F={}
for l in range(1,9): F[f'lc_l{l}']=df.lc.shift(l)
F['d1']=df.lc.shift(1)-df.lc.shift(2); F['d2']=df.lc.shift(2)-df.lc.shift(3); F['d4']=df.lc.shift(1)-df.lc.shift(5)
for c in W:
    for l in range(1,5): F[f'{c}_l{l}']=df[c].shift(l)
    F[f'{c}_r4']=df[c].shift(1).rolling(4).mean()
F['sin_w']=df.sin_w; F['cos_w']=df.cos_w
X=pd.DataFrame(F); ytrue=df.dengue_cases.values; base=df.lc.shift(1)
y=(df.lc-base)   # log growth target
seqcols=['lc']+W+['sin_w','cos_w']
S=df[seqcols].values
def seqs(idx): return np.stack([S[i-SEQ:i] for i in idx])
valid=np.where(X.notna().all(axis=1)&y.notna()&(np.arange(len(df))>=SEQ))[0]
yr=df.start_date.dt.year.values
folds=[('2023',pd.Timestamp('2023-01-01'),pd.Timestamp('2024-01-01')),('2024',pd.Timestamp('2024-01-01'),pd.Timestamp('2025-01-01')),('2025',pd.Timestamp('2025-01-01'),pd.Timestamp('2026-01-01')),('2026',pd.Timestamp('2026-01-01'),pd.Timestamp('2027-01-01'))]
def lstm_fit(Str,ytr,seed,emb=16):
    tf.keras.utils.set_random_seed(seed)
    i=layers.Input(shape=(SEQ,Str.shape[2])); h=layers.LSTM(emb,name='emb')(i); o=layers.Dense(1)(layers.Dropout(0.2)(h)); m=Model(i,o)
    m.compile(optimizer=tf.keras.optimizers.Adam(0.003),loss='mse')
    m.fit(Str,ytr,epochs=60,batch_size=16,verbose=0,validation_split=0.15,callbacks=[tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)])
    return m, Model(m.input,m.get_layer('emb').output)
def xgb(seed): return XGBRegressor(n_estimators=250,learning_rate=0.03,max_depth=3,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,random_state=seed)
res={}
def add(name,fold,pred_g,idx):
    pc=np.expm1(base.values[idx]+pred_g); res.setdefault(name,[]).append(pd.DataFrame({'fold':fold,'actual':ytrue[idx],'pred':pc,'persist':np.expm1(base.values[idx])}))
SEEDS=[0,1,2]
for fn,a,b in folds:
    tr=valid[df.start_date.values[valid]<np.datetime64(a)]; te=valid[(df.start_date.values[valid]>=np.datetime64(a))&(df.start_date.values[valid]<np.datetime64(b))]
    if len(te)==0 or len(tr)<60: continue
    sc=StandardScaler().fit(X.iloc[tr]); Xtr,Xte=sc.transform(X.iloc[tr]),sc.transform(X.iloc[te])
    ss=StandardScaler().fit(S[:tr.max()+1]); 
    Str=ss.transform(S.reshape(-1,S.shape[1])).reshape(S.shape); 
    def sq(idx): return np.stack([Str[i-SEQ:i] for i in idx])
    A,B=sq(tr),sq(te)
    add('Persistence',fn,np.zeros(len(te)),te)
    add('Ridge',fn,Ridge(alpha=10).fit(Xtr,y.values[tr]).predict(Xte),te)
    add('XGBoost-only',fn,xgb(0).fit(Xtr,y.values[tr]).predict(Xte),te)
    pl,ph,pxo=[],[],[]
    for s in SEEDS:
        m,e=lstm_fit(A,y.values[tr],s); pl.append(m.predict(B,verbose=0).ravel())
        Etr,Ete=e.predict(A,verbose=0),e.predict(B,verbose=0)
        ph.append(xgb(s).fit(np.hstack((Xtr,Etr)),y.values[tr]).predict(np.hstack((Xte,Ete))))
    add('LSTM-only',fn,np.mean(pl,0),te); add('Hybrid LSTM-XGBoost',fn,np.mean(ph,0),te)
    print('fold',fn,'done',len(tr),len(te),flush=True)
def met(d):
    e=d.pred-d.actual; mae=e.abs().mean(); mase=mae/(d.persist-d.actual).abs().mean()
    return dict(RMSE=np.sqrt((e**2).mean()),MAE=mae,MASE=mase,MAPE=(e.abs()/d.actual.clip(lower=50)).mean()*100)
rows=[]
for n,l in res.items():
    d=pd.concat(l); r=met(d); r['model']=n
    for f,g in d.groupby('fold'): r[f'MASE_{f}']=met(g)['MASE']
    rows.append(r)
tab=pd.DataFrame(rows).set_index('model').round(3); print(tab.to_string())
os.makedirs('outputs',exist_ok=True); tab.to_csv('outputs/evaluation_table.csv')
pd.concat([l.assign(model=n) for n,L in res.items() for l in L]).to_csv('outputs/predictions.csv',index=False)
