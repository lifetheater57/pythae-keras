import os

if os.environ.get("KERAS_BACKEND", None) is None:
    os.environ["KERAS_BACKEND"] = "torch"