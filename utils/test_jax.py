import os
import subprocess
import sys

"""Arguments
    [optional] arg 1 : the target of the test (a test file or folder)
"""

dirname = os.path.dirname(__file__)
command = f"python {os.path.join(dirname, 'python ../utils/test_keras.py jax')}"

if len(sys.argv) > 1:
    command += f" {sys.argv[1]}"
    
subprocess.run(command, shell=True)