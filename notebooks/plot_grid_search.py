from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import VASPsol as vs
files = [i for i in next(os.walk('.'))[1] if 'POSCAR' in next(os.walk(i))[2]]
f = [os.path.abspath(i) for i in files]
#f = ['i178']
d = [vs.data(i) for i in f]
dfs = [i.ml_df for i in d if 'ml_df' in dir(i)]
df = pd.concat(dfs)
df = df.dropna()
df = df.reset_index(drop=True)
df = df[df['NC_K'] > 1e-3]
df = df[df['SIGMA_K'] > 0.2]



data = df.groupby(['NC_K','SIGMA_K']).mean()['error_frac']
data2 = df.groupby(['NC_K','SIGMA_K']).mean()['DeltaGsolv']



x = np.array([i[0] for i in data.index.values]).reshape(-1,1)
y = np.array([i[1] for i in data.index.values]).reshape(-1,1)
z =  np.abs(data.to_numpy())
z = z.reshape(-1,1)
df2 = pd.DataFrame()
df2['NC_K'] = x.reshape(-1)
df2['SIGMA_K'] = y.reshape(-1)
df2['error_ev'] = z.reshape(-1)
x1 = np.linspace(df2['NC_K'].min(), df2['NC_K'].max(), len(df2['NC_K'].unique()))
y1 = np.linspace(df2['SIGMA_K'].min(), df2['SIGMA_K'].max(), len(df2['SIGMA_K'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata(np.hstack([x,y]), z, (x2, y2))
z2 = z2.reshape(x2.shape)
fig = plt.subplot()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1)
plt.savefig('Grid_Search.png')
plt.show()


plt.contourf(x2, y2, z2)
plt.colorbar()

