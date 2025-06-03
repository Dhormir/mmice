import sys
import io
import torch
import os
import csv
from tqdm.auto import tqdm
import re
import numpy as np
import time
import logging
import pandas as pd

# Local imports
from .utils import load_base_editor, get_device, load_predictor, write_args, get_dataset_reader, wrap_text
from .edit_finder import EditFinder, EditEvaluator
from .editor import Editor, RaceEditor
from .maskers.gradient_masker import GradientMasker

# Re-wrap stdout with UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s",
)
logger = logging.getLogger(__name__)

# Random Number Generator
RNG = np.random.default_rng(seed=42)



def get_grad_sign_direction(grad_type, grad_pred):
    """ Helper function to get sign direction. When grad_type is signed, 
    determines whether to get most negative or positive gradient values. 
    This should depend on grad_pred, i.e. what label is being used to 
    compute gradients for masking. 
    
    During Stage Two, we want to mask tokens that push *away* from the contrast 
    label or *towards* the original label. 
    
    Sign direction plays no role if only gradient *magnitudes* are used 
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.)
    """
    assert grad_pred in ["contrast", "original"]
    assert grad_type in ["integrated_l1", "integrated_signed", "normal_l1", 
            "normal_signed", "normal_l2", "integrated_l2"]
    if "signed" in grad_type and "contrast" in grad_pred:
        sign_direction = -1
    elif "signed" in grad_type and "original" in grad_pred:
        sign_direction = 1
    else:
        sign_direction = None
    return sign_direction


def check_editor_path(editor_path):
    """ Checks Editor weights from editor_path """
    if os.path.isdir(editor_path):
        editor_path = os.path.join(editor_path, "best.pth")
        if not os.path.exists(editor_path):
            raise NotImplementedError("If directory given for editor_path, " +
                    "it must contain a 'best.pth' file but found none in given " +
                    "dir. Please give direct path to file containing weights.")
    else:
        raise NotImplementedError("The given directory path for editor_path, " +
                    "has not been recognized by the OS as a valid path or " +
                    "none path was given.")
    logger.info(f"Loading Editor weights from: {editor_path}")
    return editor_path


def load_models(args):
    """ Loads Predictor and Editor by task and other args """

    logger.info("Loading models...")
    predictor = load_predictor(args.meta.task)
    editor_path = check_editor_path(args.meta.editor_path)
    editor_tokenizer, editor_model = load_base_editor(model_name=args.model.model_name,
                                                      max_length=args.model.model_max_length,
                                                      editor_path=editor_path,
                                                      lora=args.model.lora)
    device = get_device()
    editor_model = editor_model.to(device)
    
    sign_direction = get_grad_sign_direction(args.mask.grad_type, args.misc.grad_pred) 
    
    masker = GradientMasker(args.search.max_mask_frac, 
            editor_tokenizer, predictor, 
            args.model.model_max_length,
            grad_type=args.mask.grad_type, 
            sign_direction=sign_direction)

    if "race" in args.meta.task:
        editor = RaceEditor(editor_tokenizer, editor_model, masker,
                            num_gens=args.generation.num_generations,
                            num_beams=args.generation.generation_num_beams,
                            grad_pred=args.misc.grad_pred,
                            generate_type=args.generation.generate_type,
                            length_penalty=args.generation.length_penalty,
                            no_repeat_ngram_size=args.generation.no_repeat_ngram_size,
                            top_p=args.generation.top_p,
                            top_k=args.generation.top_k,
                            verbose=False,
                            editable_key="article")
    else:
        editor = Editor(editor_tokenizer, editor_model, masker,
                        num_gens=args.generation.num_generations,
                        num_beams=args.generation.generation_num_beams,
                        grad_pred=args.misc.grad_pred,
                        generate_type=args.generation.generate_type,
                        no_repeat_ngram_size=args.generation.no_repeat_ngram_size,
                        top_p=args.generation.top_p,
                        top_k=args.generation.top_k,
                        length_penalty=args.generation.length_penalty,
                        verbose=False,
                        lang=args.meta.lang
                        )
    logger.info("Done loading models.")
    return editor, predictor


def edit_indices(out_file, inputs):
    """ Helper Function to resume previous work from edits file
        returns non used indices"""
    input_indices = np.arange(len(inputs))
    if os.path.isfile(out_file):
        logger.info("Previously saved file found")
        logger.info(f"Resuming work from: {out_file}")
        edits = pd.read_csv(out_file, sep="\t", lineterminator="\n",
                            on_bad_lines="warn", encoding="utf-8").dropna()
        previous_indices = edits['data_idx'].unique().astype(np.int64)
        input_indices = np.array(list(set(input_indices) - set(previous_indices)))
    else:
        logger.info("No previously saved file found")
    RNG.shuffle(input_indices)
    return input_indices


def run_edit_test(args):
    """ Runs Stage 2 on test inputs by task. """
    task_dir = os.path.join(args.meta.results_dir, args.meta.task)
    stage_two_dir = os.path.join(task_dir, f"edits/{args.meta.stage2_exp}")

    if not os.path.exists(stage_two_dir):
        os.makedirs(stage_two_dir)

    # setting log file and output file    
    meta_log_file = os.path.join(stage_two_dir, "meta_log.txt")
    out_file = os.path.join(stage_two_dir, "edits.csv")
    # add output to log file
    logger.addHandler(logging.FileHandler(meta_log_file, encoding='utf-8'))
   

    logger.info(f"Task dir: {task_dir}")
    logger.info(f"Stage two dir: {stage_two_dir}")

    # Save args
    args_path = os.path.join(stage_two_dir, "stage_two_args.json")
    write_args(args_path, args)

    # Load models and Edit objects 
    editor, predictor = load_models(args)
    dr = get_dataset_reader(args.meta.task, split='test')
    dr = dr.shuffle(seed=42).select(range(args.misc.n_samples)) if args.misc.n_samples != 0 else dr.shuffle(seed=42)
    edit_evaluator = EditEvaluator()
    edit_finder = EditFinder(predictor, editor,
                             beam_width=args.search.beam_width, 
                             max_mask_frac=args.search.max_mask_frac,
                             search_method=args.search.search_method,
                             max_search_levels=args.search.max_search_levels,
                             min_metric=args.search.min_metric,
                             )
    
    inputs = dr['text']
    if "race" not in args.meta.task:
        # we check whether the input is empty??
        inputs = [x for x in inputs if len(x) > 0 and re.search('[a-zA-Z]', x)]

    input_indices = edit_indices(out_file, inputs)
    logger.info(f'inputs: {len(inputs)}, input_indices: {len(input_indices)}')
    # Find edits and write to file
    with open(out_file, "a", encoding="utf-8") as csv_file:
        fieldnames = ["data_idx", "sorted_idx", "orig_pred", "new_pred",
                      "contrast_pred", "orig_contrast_prob_pred",
                      "new_contrast_prob_pred", "orig_input", "edited_input",
                      "orig_editable_seg", "edited_editable_seg",
                      "minimality", "num_edit_rounds", "mask_frac",
                      "duration", "error"]
        writer = csv.writer(csv_file, delimiter="\t")
        if len(inputs) == len(input_indices):
            writer.writerow(fieldnames)
        for _, i in tqdm(enumerate(input_indices), total=len(input_indices)):
            inp = inputs[i]
            logger.info(wrap_text(f"ORIGINAL INSTANCE ({i}): {inp}"))
            start_time = time.time()
            error = False
            edited_list = None
            try:
                edited_list = edit_finder.minimally_edit(inp,
                                                         max_edit_rounds=args.search.max_edit_rounds,
                                                         edit_evaluator=edit_evaluator)
                logger.info("successful round")
                torch.cuda.empty_cache()
                sorted_list = edited_list.get_sorted_edits() 

            except Exception as e:
                logger.exception(f"ERROR: Finding edits:\n{e}")
                error = True
                sorted_list = []

            end_time = time.time()
            duration = end_time - start_time

            for s_idx, s in enumerate(sorted_list):
                    writer.writerow([i, s_idx, edited_list.orig_label,
                                     s['edited_label'], edited_list.contrast_label, 
                                     edited_list.orig_contrast_prob, s['edited_contrast_prob'], 
                                     edited_list.orig_input, s['edited_input'], 
                                     edited_list.orig_editable_seg, 
                                     s['edited_editable_seg'], s['minimality'], 
                                     s['num_edit_rounds'], s['mask_frac'], duration, error])
                    csv_file.flush()

            if sorted_list == []:
                writer.writerow([i, 0, edited_list.orig_label,
                                 None, edited_list.contrast_label,
                                 edited_list.orig_contrast_prob, None, 
                                 edited_list.orig_input, None,
                                 edited_list.orig_editable_seg,
                                 None, None, None, None, duration, error])
                csv_file.flush()
    csv_file.close()
    logger.info("Successful Execution!!")
