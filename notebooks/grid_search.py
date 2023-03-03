import VASPsol as vs
from dask.distributed import Client, LocalCluster, progress
import os
import shutil as sh
import numpy as np
import time

def status(jobs):
    st = ['pending','finished','error']
    status = []
    finished_jobs = []
    for s in st:
        count = 0
        for job in jobs:
           # count = 0
            if job.status == s:
                count += 1
                if s == 'finished':
                    finished_jobs.append(job)
        status.append(count)
    return status, finished_jobs


n_jobs = 4
client, cluster = vs.ef.dask_workers(16, 4, 1, burst=False)
cluster.scale(n_jobs)
client.wait_for_workers(n_jobs)
#cluster = LocalCluster(n_workers=1)
#client = Client(cluster)
print('ALL WORKERS PRESENT')

directories = [os.path.abspath(i) for i in next(os.walk('.'))[1] if 'POSCAR' in next(os.walk(i))[2]]

### FIRST RUN VACUUM CALCUALTIONS

def vac_calc(file):
    return vs.calculate_vacuum(file, NSW=0)

#vac_jobs = client.map(vac_calc, directories)

### THEN RUN 

nc_k = np.linspace(1e-3, 4e-3, 4)
sigma = np.linspace(0.2, 1.2, 4)
tau = np.linspace(1e-5, 1e-3, 4)

def solv_calc(file, nc, sig, tau):
    # This is where you can change the solvent that is being run~!!!
    return vs.calculate_solvent(file, NSW=0, solvent='water', dict_of_additional_tags={'NC_K':nc, 'SIGMA_K':sig, 'TAU':tau})

jobs = []
for nc in nc_k:
    for sig in sigma:
        for t in tau:
            for file in directories:
                jobs.append(client.submit(solv_calc, file, nc, sig, t))

counts, finished_jobs = status(jobs)
complete = counts[1]+counts[2]
while complete < len(jobs):
    counts, finished_jobs = status(jobs)
    complete = counts[1] + counts[2]
    print('PENDING : ',counts[0], ' | FINISHED : ', counts[1], ' | ERROR : ', counts[2],' ***************')
    time.sleep(10)

jobs = client.gather(jobs)

