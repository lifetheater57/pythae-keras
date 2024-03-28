import os
import subprocess
import sys

"""Arguments
    arg 1 : the Keras backend to use among tensorflow, torch and jax
    [optional] arg 2 : the target of the test (a test file or folder)
"""

# Validate arguments
if len(sys.argv) < 2 or sys.argv[1] not in ["tensorflow", "torch", "jax"]:
    raise Exception("The first argument must be the Keras backend to use. \n\
                       Options are \"tensorflow\", \"torch\", \"jax\".")

if sys.argv[1] == "jax":
    raise Exception("jax backend is not supported yet.") 

# Set keras backend
os.environ["KERAS_BACKEND"] = sys.argv[1]
print(f"Running Keras on {sys.argv[1]} backend.")

# Run tests
if len(sys.argv) > 2:
    if not os.path.exists(sys.argv[2]):
        raise FileNotFoundError(f"The test file {sys.argv[2]} does not exist.")
    command = f"pytest {sys.argv[2]}"
else:
    dirname = os.path.dirname(__file__)
    command = f"pytest {os.path.join(dirname, '../tests/')}"

subprocess.run(command, shell=True, check=True)