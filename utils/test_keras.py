import os
import subprocess
import sys

# Validate arguments
if len(sys.argv) < 2 and sys.argv[1] in ["tensorflow", "torch", "jax"]:
    raise Exception("The first argument must be the Keras backend to use. \n\
                       Options are \"tensorflow\", \"torch\", \"jax\".")

# Set keras backend
os.environ["KERAS_BACKEND"] = sys.argv[1]
print(f"Running Keras on {sys.argv[1]} backend.")

# Run tests
if len(sys.argv) > 2:
    command = f"pytest {sys.argv[2]}"
else:
    command = "pytest tests/"

subprocess.run(command, shell=True)