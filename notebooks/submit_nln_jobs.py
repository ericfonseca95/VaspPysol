import VASPsol as vs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask
import os
ef = vs.ef
os.getcwd()
client, cluster = ef.dask_workers(32, 6, 10, burst=False)
client
# get all dirs with a "VAC" folder in them
root_dir = '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/'
dirs = [os.path.abspath(root_dir+i) for i in next(os.walk(root_dir))[1] if 'VAC' in os.listdir(root_dir+i)]
data_jobs = [client.submit(vs.data, i) for i in dirs]
len(data_jobs)
# get all folders in root_dir
dirs = [os.path.abspath(i) for i in next(os.walk('.'))[1]]
dirs
from dask.distributed import progress
progress(data_jobs)
ml_df = pd.concat([i.result().ml_df for i in data_jobs])
ml_df['error'] = ml_df['error'].abs()
ml_df = ml_df[ml_df['error'] < 10]
ml_df
grouped_df = ml_df.groupby(['NC_K', 'SIGMA_K','TAU'])
# get only groups with at least 10 data points
grouped_df = grouped_df.filter(lambda x: len(x) > 10)
grouped_df = grouped_df.groupby(['NC_K', 'SIGMA_K','TAU'])
grouped_df = grouped_df.mean()
# include the count of data points in the group
grouped_df['count'] = ml_df.groupby(['NC_K', 'SIGMA_K','TAU']).count()['error']
grouped_df = grouped_df.reset_index()
grouped_df
# compute the % increase in performance for each group
grouped_df['% increase'] = (grouped_df['error'] - grouped_df['error'].min()) / grouped_df['error'].min() * 100
grouped_df
# parameters from optimizer
#NC_K_opt = 0.002750
#SIGMA_K_opt = 0.660000
#TAU_opt = 0.000472
# NC_K_opt = 3.66e-3
# SIGMA_K_opt = 0.5813
# TAU_opt =9.363e-4
    
NC_K_opt = 2.997e-03
SIGMA_K_opt = 6.482e-01
TAU_opt = 8.541e-04
# new NC_K_opt, etc = 0.00308742, 0.59155393, 0.00079023
# NC_K_opt = 0.00308742
# SIGMA_K_opt = 0.59155393
# TAU_opt = 0.00079023
NC_K_default = 0.0025
SIGMA_K_default = 0.6
TAU_default = 0.000525
# make opt = default
NC_K_opt = NC_K_default
SIGMA_K_opt = SIGMA_K_default
TAU_opt = TAU_default

default_df = ml_df[(ml_df['NC_K'] == NC_K_default) & (ml_df['SIGMA_K'] == SIGMA_K_default) & (ml_df['TAU'] == TAU_default)]
default_df['error'] = default_df['error'].abs()
default_df['error'].mean(), len(default_df)
# get existing data that was ran with the above parameters
ran_df = ml_df[(ml_df['NC_K'] == NC_K_opt) & (ml_df['SIGMA_K'] == SIGMA_K_opt) & (ml_df['TAU'] == TAU_opt)]
ran_df = ran_df[ran_df['FileHandle'].isin(default_df['FileHandle'])]
ran_df = ran_df.dropna()
ran_df = ran_df.reset_index(drop=True)
ran_df['error'] = ran_df['error'].abs()
ran_df = ran_df[ran_df['error'] < 10]
ran_df
# get the mean absolute error for ran_df 
error = ran_df['error'].mean()
print(error)
# sort by the error
sorted_df = ran_df.sort_values(by=['error'], ascending=False)
sorted_df
def solv_calc(file, nc, sig, tau):
    # This is where you can change the solvent that is being run~!!!
    return vs.calculate_solvent(file, NSW=0, solvent='water', calc_type='non-linear', dict_of_additional_tags={'NC_K':nc, 'SIGMA_K':sig, 'TAU':tau})

ran_files = [os.path.abspath(i) for i in sorted_df['FileHandle']]
ran_files
files_to_run = [i for i in dirs if i not in ran_files]
# we are going to use rejection sampling to get 20 random files to run
# we will use the error as the probability of being selected
def rejection_sampling(vector, samples=100):
    indicies = np.arange(len(vector))
    # fit a gaussian kernel to the data
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(vector.reshape(-1, 1))
    # create a vector of each points probability
    logprob = kde.score_samples(vector.reshape(-1, 1))
    p = np.exp(logprob)
    # normalize the probabilities
    p = p / p.sum()

    output = []
    for sample in range(samples):
        # use the probability to sample the data
        sample = np.random.choice(indicies, p=p)
        # add the sample to the vector
        output.append(sample)
    return output
# use the df to get the error and find the corresponding run files

# UNCOMMENT LATER
# error = sorted_df['error'].values
# # get the indicies of the files to run
# indicies = rejection_sampling(error, samples=20)
# # get the files to run
# fnames_to_run = sorted_df['FileHandle'].iloc[indicies].values
# #print(len(fnames_to_run), fnames_to_run)
# # get the full path to the files
# files_to_run = [os.path.abspath(i) for i in fnames_to_run]

files_to_run = ['/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0036tol', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0221tri', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0046eth', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0029hex', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0176pdi', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0033pen', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0176pdi', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0234ENmb', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0571dim', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0401amia', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0071proa', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0063die', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0050met', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0078pen', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0093met', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0110but', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0068ani', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0076but', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0034hex', 
                '/blue/hennig/ericfonseca/NASA/VASPsol/Truhlar_Benchmarks/VaspPysol/calculations/0438pho']
fnames_to_run = [os.path.basename(i) for i in files_to_run]
print(len(files_to_run), files_to_run, fnames_to_run)
ml_df[(ml_df['FileHandle'].isin(fnames_to_run))].groupby(['NC_K','SIGMA_K','TAU']).mean()

jobs = [client.submit(solv_calc, os.path.abspath('../calculations/'+i), NC_K_opt, SIGMA_K_opt, TAU_opt) for i in ml_df['FileHandle'].unique()]
jobs = client.gather(jobs)
exit()