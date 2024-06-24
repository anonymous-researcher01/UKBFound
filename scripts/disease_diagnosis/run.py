import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import time
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from config import *
commands=['python ./xgb.py','python ./lgb.py',
          'python ./rf.py']#Change this line to your rapids python path, find it with 'which python'
gpu=0
print(commands)
max_concurrent =20
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