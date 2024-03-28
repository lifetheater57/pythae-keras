import os
import pytest
import subprocess


PATH = os.path.dirname(os.path.abspath(__file__))
PATH_UTILS = os.path.join(PATH, "../utils")
FILE_TEST = os.path.join(PATH, "test_baseAE.py")


@pytest.fixture(params=["tensorflow", "torch", "jax"])
def backend(request):

    return request.param

@pytest.fixture(params=[os.path.join(PATH, "..")])
def root_path(request):
    return request.param


@pytest.fixture(
    params=[
        os.path.join(PATH, "../utils"), 
        os.path.join(PATH, "../tests"), 
        os.path.join(PATH, "../examples"),
    ]
)
def non_root_path(request):
    return request.param


class Test_Library_Testing:
    def test_run_single_backend_test(self, backend):
        script = os.path.join(PATH_UTILS, f"test_{backend}.py")
        subprocess.run(f"python {script} {FILE_TEST}", shell=True, check=True)

    def test_run_all_backends_test(self):
        script = os.path.join(PATH_UTILS, f"test_all.py")
        subprocess.run(f"python {script} {FILE_TEST}", shell=True, check=True)
        
        # file_do_not_exists
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(f"python {script} dummy.file", shell=True, check=True)

    def test_run_root_directory(self, root_path):
        script = os.path.join(PATH_UTILS, f"test_all.py")
        subprocess.run(f"python {script} {FILE_TEST}", shell=True, check=True, cwd=root_path)

    def test_run_from_non_root_directory(self, non_root_path):
        script = os.path.join(PATH_UTILS, f"test_all.py")
        subprocess.run(f"python {script} {FILE_TEST}", shell=True, check=True, cwd=non_root_path)

    def test_raises_wrong_backend(self):
        script = os.path.join(PATH_UTILS, f"test_keras.py")
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(f"python {script} 'dummy' {FILE_TEST}", shell=True, check=True)
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(f"python {script} 'jax' {FILE_TEST}", shell=True, check=True)
        
