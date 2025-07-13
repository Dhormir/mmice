from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path
import os


# Set your desired cache directory
os.environ["HF_HOME"] = str(Path.cwd() / ".cache")

# Local imports
from mmice.stage_two import run_edit_test
from mmice.utils import get_args


def main():
    args = get_args("stage2")
    with logging_redirect_tqdm():
        run_edit_test(args)


if __name__ == "__main__":
    main()
