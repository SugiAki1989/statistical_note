import os
import sys

import torch


def main():
    cwd = os.getcwd()
    print("Virtualenv: ", os.path.relpath(sys.executable, cwd))
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")

    x = torch.rand(5, 3)
    return print(x)


if __name__ == "__main__":
    main()
