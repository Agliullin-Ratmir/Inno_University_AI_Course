import os
from contextlib import contextmanager

@contextmanager
def envvars():

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] ="false"
        yield
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"


if __name__ == "__main__":

    print("Before:", os.environ.get("CUDA_VISIBLE_DEVICES"), ", ", os.environ.get("TOKENIZERS_PARALLELISM"))
    with envvars():
        print("Inside:", os.environ.get("CUDA_VISIBLE_DEVICES"), ", ", os.environ.get("TOKENIZERS_PARALLELISM"))

    print("after:", os.environ.get("CUDA_VISIBLE_DEVICES"), ", ", os.environ.get("TOKENIZERS_PARALLELISM"))