import subprocess
import sys

for backend in ["tensorflow", "torch"]:#, "jax"]:
    command = f"python utils/test_{backend}.py"
    # command = "echo $KERAS_BACKEND"

    if len(sys.argv) > 1:
        command += f" {sys.argv[1]}"
        
    subprocess.run(command, shell=True)