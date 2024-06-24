import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import time
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from config import *
def modelchar(x):
    if x >= 0 and x <= 9:
        return str(x)
    elif x >= 10:
        return chr(65 + x - 10)
def cmdfull(x,y,t,e,s,g):
    return f"python ./trainer.py {x} {y} {t} {e} {g} {s}\n"
def cmdimp(x,y,t,e,s,g):
    return f"python ./trainerimpute.py {x} {y} {t} {e} {g} {s}\n"
def cmdpri(x,y,t,e,s,g):
    return f"python ./trainerpriority.py {x} {y} {t} {e} {g} {s}\n"
def cmdsurvpri(x,y,t,e,s,g):
    return f"python ./trainersurvpriority.py {x} {y} {t} {e} {g} {s}\n"

def cmdsc(x,y,t,e,s,g):
    return f"python ./trainersc.py {x} {y} {t} {e} {g} {s}\n"

def cmdsurv(x,y,t,e,s,g):
    return f"python ./trainersurv.py {x} {y} {t} {e} {g} {s}\n"


def cmdsurvsc(x,y,t,e,s,g):
    return f"python ./trainersurvsc.py {x} {y} {t} {e} {g} {s}\n"

def cmdspecial(x,y,t,e,s,g):
    return f"python ./trainerspecialdata.py {x} {y} {t} {e} {g} {s}\n"
def cmdsurvspecial(x,y,t,e,s,g):
    return f"python ./trainersurvspecial.py {x} {y} {t} {e} {g} {s}\n"

commands=[]
gpu=0


for cat in [1,2,3,4,5,6]:
    for model in [0,1,2,3]:
        for imageimputetype in range(15):
            for xtype in [0]:
                c=cmdfull(cat,model,imageimputetype,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1
for cat in [1,2,3,4,5,6]:
    for ycat in [1]:
        for hyperp in [0]:
            for xtype in [0]:
                for count in range(6):
                    c=cmdspecial(cat,ycat,hyperp,xtype,f'{folder}_{count}',ava_gpus(gpu))
                    commands.append(c)
                    gpu+=1
cats=[1,2,3,4,5,6]
for cat in cats:
    for model in [1]:
        for hyperp in [1,2]:
            for xtype in [0]:
                c=cmdsurvpri(cat,model,hyperp,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1
for cat in cats:
    for model in [1]:
        for hyperp in [1,2]:
            for xtype in [0]:
                c=cmdpri(cat,model,hyperp,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1

for cat in cats:
    for model in [1]:
        for imageimputetype in [1,2,3,4,5,6]:
            for xtype in [0]:
                c=cmdimp(cat,model,imageimputetype,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1
cats=[1,2,3,4,5,6]
for cat in cats:
    for model in [0,1,2,3]:
        for imageimputetype in [0]:
            for xtype in [0]:
                c=cmdsurv(cat,model,imageimputetype,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1

cats=[6]
for cat in cats:
    for ycat in range(30):
        for hyperp in [1]:
            for xtype in [0]:
                c=cmdsurvsc(cat,ycat,hyperp,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1
for cat in cats:
    for ycat in range(30):
        for hyperp in [1]:
            for xtype in [0]:
                c=cmdsc(cat,ycat,hyperp,xtype,folder,ava_gpus(gpu))
                commands.append(c)
                gpu+=1


print(commands)
max_concurrent =15
def execute_command(command):
    print(f"Executing command: {command},total {len(commands)}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Command finished: {command}")

with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
    futures = [executor.submit(execute_command, command) for command in commands]
    for future in futures:
        future.result()
print("All commands executed.")
