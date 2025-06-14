from tqdm.contrib.logging import logging_redirect_tqdm
import os

# Set your desired cache directory
os.environ['HF_HOME'] = 'D:\Repositories\multilingual_mice\.cache'

# Local imports
from mmice.stage_two import run_edit_test
from mmice.utils import get_args


if __name__ == '__main__':

    args = get_args("stage2")
    with logging_redirect_tqdm():
        run_edit_test(args)
