## 
import torch
import pandas as pd
import os
import numpy as np
import COSMO_TL as ctl
from dask.distributed import Client, LocalCluster, progress
import dask
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn
from scipy.interpolate import LinearNDInterpolator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import  mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import curve_fit, minimize, differential_evolution
# get particle swarm optimizer
from pyswarm import pso
import time
import os
import pickle
import datetime
import kmodels as kmk
import shutil
import dask.dataframe as dd


def get_X_solute(df):
    X = df[['volume_solute', 'area_solute', 'NC_K', 'SIGMA_K','TAU', 'default_error']]
    sig_cols = [col for col in df.columns if 'sigma_solute' in col]
    sigs = df[sig_cols].to_numpy()
    X = X.to_numpy().reshape(len(df), -1)
    X = np.column_stack((X, sigs))
    return X

def get_X_solvent(df):
    X = df[['volume_solvent', 'area_solvent','NC_K','SIGMA_K','TAU', 'default_error']]
    sig_cols = [col for col in df.columns if 'sigma_solvent' in col]
    sigs = df[sig_cols].to_numpy()
    X = X.to_numpy().reshape(len(df), -1)
    X = np.column_stack((X, sigs))
    return X

def get_X(df):
    X_solute = get_X_solute(df)
    X_solvent = get_X_solvent(df)
    # solvent prop cols = eps,n,alpha,beta,gamma,phi**2,psi**2,beta**2
    solvent_props_names = ['eps', 'n', 'alpha', 'beta', 'gamma', 'phi**2', 'psi**2', 'beta**2']
    solvent_props = df[solvent_props_names].to_numpy()
    X = np.column_stack((X_solute, X_solvent, solvent_props))
    return X
# given a dataframe return a dataframe with the mean value for all columns with error in the name
# grouped by SoluteName
def get_mean_df(df):
    df2 = df.copy()
    original_columns = list(df2.columns)
    print(original_columns)
    cols = [col for col in df2.columns if 'error' in col]
    original_columns = [col for col in original_columns if col not in cols]
    df3 = df2.groupby(['SoluteName', 'NC_K','SIGMA_K','TAU'])[cols].mean()
    df3 = df3.reset_index()
    # return a dataframe with the unique values in SoluteName and the mean values for all columns with error in the name
    # get all the other colums  from the original dataframe
    df4 = df2[original_columns]
    df4 = df4.reset_index(drop=True)
    df5 = pd.merge(df4, df3, on=['SoluteName', 'NC_K','SIGMA_K','TAU'])
    df5 = df5.drop_duplicates()
    return df5

<<<<<<< Updated upstream
csv_path = '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/data/vaspsol_data_3_2_2023_balanced'

#df = pd.read_csv(csv_path)
df = dd.read_parquet(csv_path)
df = df.compute()
=======
csv_path = '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/data/vaspsol_data_3_2_2023_balanced.csv'
csv_path = os.path.abspath('../data/vaspsol_data_3_2_2023_balanced.csv')
df = pd.read_csv(csv_path)
>>>>>>> Stashed changes
print(df)
df['error'] = df['error'].abs()
df = df[df['error'] < 10]
df = df[df['Solvent'] == 'water']
df = df[df['Charge'] == 0]
NC_K_default = 0.0025
SIGMA_K_default = 0.6
TAU_default = 0.000525
# default_df = df[(df['NC_K'] == NC_K_default) & (df['SIGMA_K'] == SIGMA_K_default) & (df['TAU'] == TAU_default)]
# print(default_df)



# df_to_append = default_df[['SoluteName','error']]
# # rename error to default_error
# df_to_append = df_to_append.rename(columns={'error': 'default_error'})
# df_to_append


# # match up the default error back to the original dataframe
# df = pd.merge(df, df_to_append, on=['SoluteName'])
# # this expanded the number of rows in the dataframe. This is not what we want
#df = df.drop_duplicates('Unnamed: 0')

groups = df[df['Solvent']=='water'].groupby(['NC_K', 'SIGMA_K', 'TAU'])
# print(df)

#df_test = pd.read_csv(csv_path)
df_test = dd.read_parquet(csv_path)
df_test = df_test.compute()
# we want the NC_K, SIGMA_K and TAU combinations that are not in 
# the training set
df_test = df_test[~df_test[['NC_K', 'SIGMA_K', 'TAU']].isin(df[['NC_K', 'SIGMA_K', 'TAU']]).all(axis=1)]
default_df = df_test[(df_test['NC_K'] == NC_K_default) & (df_test['SIGMA_K'] == SIGMA_K_default) & (df_test['TAU'] == TAU_default)]
print(default_df)

# get the number of unique groups
# using the groups split of the dataframe so that unique combos of NC_K, SIGMA_K, and TAU are in each group
split = 0.99
## get the unique groups
#groups = df.groupby(['NC_K', 'SIGMA_K', 'TAU'])
# get the indicies of the groups
indicies = [np.array(i) for i in groups.indices.values()]
# get the number of groups
num_groups = len(indicies)
print('Number of groups: ', num_groups)

# get the number of groups to use for training
num_train_groups = int(num_groups*split)
print('Number of groups to use for training: ', num_train_groups)
print('Number of groups to use for testing: ', num_groups - num_train_groups)
# get the indicies of the groups to use for training
print(len(indicies))

idx_temp = np.arange(len(indicies))
train_indicies = [indicies[i] for i in np.random.choice(idx_temp, size=num_train_groups, replace=False)]
train_indicies = np.concatenate(train_indicies)
# get the indicies of the groups to use for testing
test_indicies = np.array([i for i in np.concatenate(indicies) if i not in train_indicies])
train_df = df.iloc[train_indicies]
test_df = df.iloc[test_indicies]

X_train = get_X_solute(train_df)
X_test = get_X_solute(test_df)
y_train = train_df['error'].to_numpy()
y_test = test_df['error'].to_numpy()
X_test = get_X_solute(test_df)

# print out the shape of the training data and the training labels. Nice retro looking print statment
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
n_observations_train = X_train.shape[0]
n_features_train = X_train.shape[1]
n_observations_test = X_test.shape[0]
n_features_test = X_test.shape[1]

print('TRAINING SET DETAILS')
print(f'Number of observations: {n_observations_train}')
print(f'Number of features: {n_features_train}')

print('TESTING SET DETAILS')
print(f'Number of observations: {n_observations_test}')
print(f'Number of features: {n_features_test}')



scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
# send the training data to gpu
X_train = torch.from_numpy(X_train).float().cuda().reshape(-1, n_features_train)
y_train = torch.from_numpy(y_train).float().cuda().reshape(-1, 1)
X_test = torch.from_numpy(X_test).float().cuda().reshape(-1, n_features_train)


model = kmk.NN(n_inputs=n_features_train, 
                     n_outputs=1, 
                     layer_size=60, 
                     layers=2)


def get_n_parmas(model):
    n_params = 0
    for param in model.parameters():
        n_params += param.numel()
    return n_params
n_params = get_n_parmas(model)
print('Model architecture:')
print(model)
print(f'The number of parameters in the model is {n_params}')
n_epochs = 1001
batch_size = 32
lr = 0.0001
losses = kmk.run_Pytorch(model, X_train, y_train, 
                               n_epochs=n_epochs, 
                               batch_size=batch_size, 
                               learning_rate=lr,
                               optimizer=torch.optim.Adam(model.parameters(), 
                                                          lr=lr,
                                                          weight_decay=0.001))

import matplotlib.pyplot as plt
# create a professional learning curve plot function
def plot_learning_curve(losses, title=None, save=False, filename=None):
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if title:
        plt.title(title)
    if save:
        # check if the figures directory exists
        if not os.path.exists('figures'):
            os.mkdir('figures')
        plt.savefig('figures/'+filename)
# use today's date and time to create a unique filename


now = datetime.datetime.now()
# get the model attributes and create a file string. we only want attributes like layers, layer_size, n_inputs, n_outputs
model_attributes = [attr for attr in dir(model) if not callable(getattr(model, attr)) and not attr.startswith("__")]
model_string = ''
date_string = now.strftime("%d-%m-%Y_%H-%M-%S")

learning_curve_fname = f'learning_curve_{date_string}.png'
plot_learning_curve(losses, title='NN Learning Curve', save=True, filename=learning_curve_fname)


# save the model parameters attributes to the name of the file and save the state dict 
# also include the time in dd-mm-yy_hh-mm-ss format


for attr in model_attributes:
    if attr in ['layers', 'layer_size', 'n_inputs', 'n_outputs']:
        model_string += f'{attr}_{getattr(model, attr)}_'
print(model_string)
filename = f'{model_string}{date_string}.pkl'
# save the model parameters
with open(filename, 'wb') as f:
    model = model.to('cpu')
    pickle.dump(model.state_dict(), f)
# move the model to ./models/
shutil.move(filename, './models/')

# we also need to save the scaler
scaler_filename = f'scaler_{model_string}{date_string}.pkl'
with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)
# move the scaler to ./models/
shutil.move(scaler_filename, './models/')

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
model = model.to('cuda')
pred = model(X_test).cpu().detach().numpy()
plt.scatter(pred, y_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Predicted')
plt.ylabel('Actual')
# put the rmse, mse and r2 in the bottom left corner of the plot
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(np.mean((pred - y_test)**2))
r2 = r2_score(y_test, pred)
plt.text(0.7, 0.1, f'MAE: {mae:.2f}', transform=plt.gca().transAxes)
plt.text(0.7, 0.2, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)
plt.text(0.7, 0.3, f'R2: {r2:.2f}', transform=plt.gca().transAxes)


# save parity plot with the date too
parity_plot_fname = f'parity_plot_{date_string}.png'
plt.savefig('figures/'+parity_plot_fname)