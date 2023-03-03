import os
import shutil

cwd = os.getcwd()

x = [i[0] for i in os.walk(cwd)]
delete = ['CHGCAR','CHG','WAVECAR']
print(x)
space = 0


for i in x:
    os.chdir(i)
    for j in delete:
        if os.path.exists(j):
            space = space+float(os.path.getsize(j))
            os.remove(j)
space = space/1e-9
print(space,' GB have been removed')
