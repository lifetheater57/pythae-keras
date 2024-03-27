import subprocess
import sys

command = "python utils/test_keras.py torch"

if len(sys.argv) > 1:
    command += f" {sys.argv[1]}"
    
subprocess.run(command, shell=True)