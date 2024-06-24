import subprocess
from concurrent.futures import ThreadPoolExecutor
from config import *
commands = []
gpu=0
for i in range(0,1560,30):
    gpu+=1
    commands.append(
        f'python ./varimpsurv.py {i} {i+30} {ava_gpus(gpu)} {folder}')
max_concurrent = 10

def execute_command(command):
    print(f"Executing command: {command},total {len(commands)}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    print(f"Command finished: {command}")

with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
    futures = [executor.submit(execute_command, command)
                for command in commands]        
    for future in futures:
        future.result()
print("All commands executed.")

commands = []
gpu=0
for i in range(0,1560,30):
    gpu+=1
    commands.append(
        f'python ./varimp.py {i} {i+30} {ava_gpus(gpu)} {folder}')
max_concurrent = 10


with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
    futures = [executor.submit(execute_command, command)
                for command in commands]        
    for future in futures:
        future.result()
print("All commands executed.")