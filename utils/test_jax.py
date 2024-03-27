import subprocess
import sys

command = "python utils/test_keras.py jax"

if len(sys.argv) > 1:
    command += f" {sys.argv[1]}"
    
subprocess.run(command, shell=True)