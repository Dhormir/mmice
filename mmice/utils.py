from transformers import T5ForConditionalGeneration, T5Config, \
    T5TokenizerFast, MT5ForConditionalGeneration, MT5Config, \
    UMT5ForConditionalGeneration, UMT5Config, MyT5Tokenizer,\
    BertForMaskedLM, BertTokenizerFast, BertConfig
from transformers import pipeline
from peft import PeftModel, PeftConfig, TaskType
from datasets import load_dataset
from munch import Munch
import numpy as np
import textwrap
import argparse
import difflib
import logging
import torch
import json
import sys
import os, re

from typing import List, Optional, Any

# Local imports
from .task_loader import load_chilean_hate, load_42k_hcuch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Static list of Available models
AVAILABLE_MODELS = [
                    "t5-small",
                    "t5-base",
                    "t5-large",
                    "t5-3b",
                    "t5-11b",
                    "google/mt5-small",
                    "google/mt5-base",
                    "google/mt5-large",
                    "google/mt5-xl",
                    "google/mt5-xxl",
                    "google/umt5-small",
                    "google/umt5-base",
                    "google/umt5-xl",
                    "google/umt5-xxl",
                    "Tomlim/myt5-small",
                    "bert-base-uncased",
                    "dccuchile/bert-base-spanish-wwm-uncased",
                    ]

####################################################################
######################## Arg Parsing Utils #########################
####################################################################


def get_shared_parsers():
    """ Helper function to get parsers.
    Gets parsers that are shared across stage one and stage two. """
    meta_parser = argparse.ArgumentParser()
    meta_parser.add_argument("-task", required=True, 
            help='Name of task. Currently, only RACE, IMDB, \
                Newsgroups and ChileanHate are supported.', 
            choices=['race', 'imdb', 'newsgroups', 'chileanhate', '42k_hcuch'])
    meta_parser.add_argument("-results_dir", default="results", 
            help='Results dir. Where to store results.')
    meta_parser.add_argument("-lang", default="en",
            help='Task language. Indicates the language of the task.',
            choices=['en', 'es'])

    mask_parser = argparse.ArgumentParser()
    mask_parser.add_argument("-mask_type", default="random",
            choices=["grad", "random"])
    mask_parser.add_argument("-grad_type", default="normal_l1", 
            choices=["integrated_l1", "integrated_signed", "normal_l1", \
                    "normal_signed", "normal_l2", "integrated_l2"],
            help="Which gradient method to use for grad-based masking. \
                    l1/signed/l2 determine how to aggregate over the emb dim.")

    model_parser = argparse.ArgumentParser()
    model_parser.add_argument("-model_name", default="t5-small", 
            help='Name of editor model. Currently, T5, MT5, UMT5 and BERT are supported.')
    model_parser.add_argument("-model_max_length", default=700, type=int,
            help="Maximum number of tokens that Editor model can take.")
    model_parser.add_argument("-lora", default=False, type=bool,
            help="Whether to use Low Rank Adaptation for the editor model.")
    return {"meta": meta_parser, "mask": mask_parser, "model": model_parser}


def get_stage_one_parsers():
    """ Helper function to get parsers for Stage 1. """
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("-train_batch_size", default=1, type=int)
    train_parser.add_argument("-val_batch_size", default=1, type=int)
    train_parser.add_argument("-num_epochs", default=5, type=int)
    train_parser.add_argument("-lr", default=1e-4, type=float)
    train_parser.add_argument("-seed", default=42, type=int)
    train_parser.add_argument("-data_split_ratio", default=0.75, type=float)
    # LoRA Arguments
    train_parser.add_argument("-r", default=8, type=int)
    train_parser.add_argument("-lora_alpha", default=32, type=int)
    train_parser.add_argument("-lora_dropout", default=.1, type=float)
    train_parser.add_argument("-bias", default="none", type=str, choices=['none', 'all', 'lora_only'])
    train_parser.add_argument("-task_type", default=TaskType.SEQ_2_SEQ_LM, type=TaskType)

    misc_parser = argparse.ArgumentParser()
    misc_parser.add_argument("-target_label", default="gold", 
            choices=["gold", "pred"], 
            help="Which label to use as the target during Editor training")
    return {"train": train_parser, "misc": misc_parser} 


def get_stage_two_parsers():
    """ Helper function to get parsers for Stage 2. """

    generation_parser = argparse.ArgumentParser()
    generation_parser.add_argument("-generate_type", default="sample", 
            choices=['beam', 'sample'])
    generation_parser.add_argument("-top_k", default=30, type=int)
    generation_parser.add_argument("-top_p", default=0.95, type=float)
    generation_parser.add_argument("-length_penalty", default=1.0, type=float)
    generation_parser.add_argument("-generation_num_beams", default=15, type=int)
    generation_parser.add_argument("-num_generations", default=15, type=int)
    generation_parser.add_argument("-no_repeat_ngram_size", default=2, type=int)
    
    search_parser = argparse.ArgumentParser()
    search_parser.add_argument("-max_mask_frac", default=0.55, type=float,
            help="Maximum mask fraction")
    search_parser.add_argument("-max_edit_rounds", default=3, type=int,
            help="Maximum number of edit rounds")
    search_parser.add_argument("-max_search_levels", default=4, type=int,
            help="Maximum number of search levels")
    search_parser.add_argument("-beam_width", default=3, type=int,
            help="Beam width for beam search over edits.")
    search_parser.add_argument("-search_method", default="binary", 
            choices=["binary", "linear"], 
            help="Which kind of search method to use: binary or linear.")
    search_parser.add_argument("-min_metric", default="levenshtein", 
            choices=["levenshtein", "cosine", "mauve"], 
            help="Which kind of metric to use for minimality: levenshtein, cosine, mauve.")

    misc_parser = argparse.ArgumentParser()
    misc_parser.add_argument("-grad_pred", default="original", 
            choices=["original", "contrast"], help="Whether to take gradient \
                    with respect to the contrast or original prediction")
    misc_parser.add_argument("-n_samples", default=0, type=int,
            help="Whether to use all the test samples or use a certain amount")

    return {"generation": generation_parser, 
            "search": search_parser, 
            "misc": misc_parser}


def get_parsers_by_stage(stage="stage1"):
    """ Gets parsers by stage. """

    if stage not in ["stage1", "stage2"]:
        raise ValueError(f"stage must be 'stage1' or 'stage2' but got {stage}")
    parsers = get_shared_parsers()
    if stage == "stage1":
        parsers.update(get_stage_one_parsers())
        parsers["meta"].add_argument("-stage1_exp", required=True, 
                help='Stage 1 exp name. Used to create subdir in results dir \
                        for trained Editor.')
    else:
        parsers.update(get_stage_two_parsers())
        parsers["meta"].add_argument("-editor_path", required=True, 
                help="Path to trained Editor checkpoint. Can be a directory \
                        containing 'best.pth' file OR a direct path to file \
                        containing weights (if training ended prematurely).") 
        parsers["meta"].add_argument("-stage2_exp", required=True, 
                help='Stage 2 experiment name. Used to create subdir within \
                        stage 1 directory for editing results.')
    return parsers


def get_args(stage):
    """ Gets args by stage. """
    if stage not in ["stage1", "stage2"]:
        raise ValueError("stage must be one of ['stage1', 'stage2'] " + \
                f"but got value {stage}")
    parsers = get_parsers_by_stage(stage)
    args = {}
    extra_args = sys.argv[1:]
    for arg_subset, parser in parsers.items():
        temp_args, extra_args = parser.parse_known_args(extra_args)
        args[arg_subset] = Munch(vars(temp_args))
    assert extra_args == [], f"Unrecognized arguments supplied: {extra_args}"
    return Munch(args)


def write_args(args_path, args):
    """ Helper function to write args
    Args:
        args: list[Dict]
        args_path: str
    """
    logger.info("Writing args to: " + args_path)
    for name, sub_args in args.items():
        logger.info(f"{name} args: {sub_args}")
    f = open(args_path, "w")
    f.write(json.dumps(args, indent=4))
    f.close()

####################################################################
####################### Task Specific Utils ########################
####################################################################

def clean_text(example, special_chars=["\n", "\t", "\x85", "\x97", "#", "<br />", "<br/>"]):
    text = example['text']
    url_pattern = r'\b(?:http[s]?://|www\.)\S+\b'
    text = re.sub(url_pattern, '', text)
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")

    # Find the first @mention
    first_mention = re.search(r'@\w+', text)
    if first_mention:
        end = first_mention.end()
        text = text[:end] + re.sub(r'@\w+', '', text[end:])
        
    # Replace repeated whitespace with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    example['text'] = text.encode().lower()
    return example

def get_dataset_reader(task_name, split='train'):
    task_options = ["imdb", "race", "newsgroups", "chileanhate", "42k_hcuch"]
    if task_name not in task_options:
        raise NotImplementedError(f"Task {task_name} not implemented; \
                must be one of {task_options}")
    elif task_name == "imdb":
        return load_dataset("imdb", split=split).map(clean_text)
    elif task_name == "race":
        return load_dataset("race")
    elif task_name == "newsgroups":
        return load_dataset("SetFit/20_newsgroups", split=split).filter(lambda example:  example['label'] in set(range(16))).map(clean_text)
    # Example for new tasks
    elif task_name == "chileanhate":
        task_data_dir = os.path.join('data', task_name)
        data_files = [os.path.join(task_data_dir, "tweets_train.csv"),
                      os.path.join(task_data_dir, "tweets_test.csv")]
        return load_chilean_hate(data_files=data_files)[split].map(clean_text)
    elif task_name == "42k_hcuch":
        task_data_dir = os.path.join('data', task_name)
        data_files = [os.path.join(task_data_dir, "labeled_data_3-label_train.csv"),
                      os.path.join(task_data_dir, "labeled_data_3-label_test.csv"),
                      os.path.join(task_data_dir, "labeled_data_3-label_validation.csv")]
        return load_42k_hcuch(data_files=data_files)[split]#.map(clean_text)

# Languages format
LANGUAGES_FORMAT = {
    "en": {"label": "label", "input": "input"},
    "es": {"label": "etiqueta", "input": "entrada"},
    }

def format_classif_input(inp, label, lang):
    return f"{LANGUAGES_FORMAT[lang]['label']}: " + str(label) + f". {LANGUAGES_FORMAT[lang]['input']}: " + str(inp)


def format_multiple_choice_input(context, question, options, answer_idx):
    formatted_str = f"question: {question} answer: choice {answer_idx}:" + \
            f"{options[answer_idx]} context: {context}"
    for option_idx, option in enumerate(options):
        formatted_str += " choice" + str(option_idx) + ": " + option
    return formatted_str


def load_predictor(task, predictor_folder="trained_predictors/"):
    task_options = ["imdb", "race", "newsgroups", "chileanhate"]
    if task not in task_options:
        raise NotImplementedError(f"Task {task} not implemented; \
                must be one of {task_options}")
    predictor_path = os.path.join(predictor_folder, task, "model")
    if not os.path.exists(predictor_path):
        raise ValueError(f"Cannot find predictor path {predictor_path}")
    logger.info(f"Loading Predictor from: {predictor_path}")
    task_name = "text-classification" if task == task_options[2] else "sentiment-analysis"
    predictor = pipeline(task_name,
                         model=predictor_path,
                         device=get_device(),
                         padding=True,
                         truncation=True,
                         top_k=None)
    logger.info("Done loading predictor.")
    return predictor

####################################################################
########################### Model Utils ############################
####################################################################

def get_token_char_span(encoding, tokenizer, token_index):
    """
    Approximates the character span (start, end) of a token at a given index in the encoded sequence.

    Args:
        text (str): The original input text.
        tokenizer: A non-fast tokenizer instance.
        token_index (int): The index of the token in the encoded sequence.

    Returns:
        tuple: A tuple (start, end) representing the character span of the token in the original text.
               Returns (None, None) if the span cannot be determined.
    """
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    text = tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)

    # Initialize variables
    current_position = 0

    for idx, token in enumerate(tokens):
        # Skip special tokens
        if token in tokenizer.all_special_tokens:
            if idx == token_index:
                return (None, None)
            continue

        # Remove '▁' prefix used in SentencePiece tokenization
        clean_token = token.lstrip("▁")

        # Find the token in the text starting from the current position
        start = text.find(clean_token, current_position)
        if start == -1:
            if idx == token_index:
                return (None, None)
            continue
        end = start + len(clean_token)
        if idx == token_index:
            return (start, end)
        current_position = end

    return (None, None)



def load_base_editor(model_name, max_length=700, editor_path=None, lora=False):
    editor_model_path = editor_path if editor_path else model_name
    if model_name not in AVAILABLE_MODELS:
        raise NotImplementedError(f"Model {model_name} not implemented; \
                must be one of {AVAILABLE_MODELS}")

    if lora and editor_path:
        logger.info("Loading LoRa Model")
    else:
        logger.info("Loading regular Model")
    if "umt5-" in model_name:
        # Load base model from the config
        if lora and editor_path:
            peft_config = PeftConfig.from_pretrained(editor_model_path)
            base_model = UMT5ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path,
                                                                    torch_dtype=torch.bfloat16)
            # Apply LoRA weights
            model = PeftModel.from_pretrained(base_model, editor_model_path)
            model.generation_config.max_new_tokens = None
            tokenizer = T5TokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                        model_max_length=max_length,
                                                        truncation=True,
                                                        padding=True)
        else:  
            model_config = UMT5Config.from_pretrained(model_name, force_download=True)
            model_config.max_new_tokens = None
            model = UMT5ForConditionalGeneration.from_pretrained(editor_model_path,
                                                                torch_dtype=torch.bfloat16,
                                                                config=model_config)
            # umt5 config fix
            model.max_new_tokens = None
            # We use legacy false to prevent warning
            # something was misplaced in version 4.36.2 because its throwing the warning anyway
            tokenizer = T5TokenizerFast.from_pretrained(model_name,
                                                        model_max_length=max_length,
                                                        truncation=True,
                                                        padding=True, force_download=True)
        
    elif "mt5-" in model_name:
         # Load base model from the config
        if lora and editor_path:
            peft_config = PeftConfig.from_pretrained(editor_model_path)
            base_model = MT5ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path,
                                                                    torch_dtype=torch.bfloat16)
            # Apply LoRA weights
            model = PeftModel.from_pretrained(base_model, editor_model_path)
            tokenizer = T5TokenizerFast.from_pretrained("Dhurmir/patched-mt5-tokenizer",
                                                        model_max_length=max_length,
                                                        extra_ids=0,
                                                        legacy=False,
                                                        truncation=True,
                                                        padding=True)
        else:
            model_config = MT5Config.from_pretrained(model_name, force_download=True)
            model_config.max_new_tokens = None
            model = MT5ForConditionalGeneration.from_pretrained(editor_model_path,
                                                                torch_dtype=torch.bfloat16,
                                                                config=model_config)
            # We use legacy false to prevent warning
            # something was misplaced in version 4.36.2 because its throwing the warning anyway
            tokenizer = T5TokenizerFast.from_pretrained("Dhurmir/patched-mt5-tokenizer",
                                                        # extra_ids=0 why? I dont know but it works!
                                                        extra_ids=0,
                                                        model_max_length=max_length,
                                                        truncation=True,
                                                        padding=True)
    elif "myt5-" in model_name:
        model_config = T5Config.from_pretrained(model_name, force_download=True)
        model = T5ForConditionalGeneration.from_pretrained(editor_model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            config=model_config)
        # We use legacy false to prevent warning
        # something was misplaced in version 4.36.2 because its throwing the warning anyway
        tokenizer = MyT5Tokenizer.from_pretrained(model_name,
                                                model_max_length=max_length,
                                                truncation=True, from_slow=True,
                                                padding=True, force_download=True)
    elif "t5-" in model_name:
        # Load base model from the config
        if lora and editor_path:
            peft_config = PeftConfig.from_pretrained(editor_model_path)
            base_model = T5ForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path,
                                                                    torch_dtype=torch.bfloat16)
            # Apply LoRA weights
            model = PeftModel.from_pretrained(base_model, editor_model_path)
            tokenizer = T5TokenizerFast.from_pretrained(peft_config.base_model_name_or_path,
                                                        model_max_length=max_length,
                                                        truncation=True,
                                                        padding=True)
        else:  
            model_config = T5Config.from_pretrained(model_name, force_download=True)
            model = T5ForConditionalGeneration.from_pretrained(editor_model_path,
                                                               torch_dtype=torch.bfloat16,
                                                               config=model_config)
            tokenizer = T5TokenizerFast.from_pretrained(model_name,
                                                        model_max_length=max_length,
                                                        truncation=True,
                                                        padding=True, force_download=True)

    elif "bert" in model_name:
        model_config = BertConfig.from_pretrained(model_name, force_download=True)
        model = BertForMaskedLM.from_pretrained(editor_model_path,
                                                config=model_config)
        tokenizer = BertTokenizerFast.from_pretrained(model_name,
                                                    model_max_length=max_length,
                                                    truncation=True,
                                                    padding=True, force_download=True)
    return tokenizer, model


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def get_prob_pred(pred, label_idx):
    """ Given a prediction, gets predicted probability of label_idx. """
    for idx, prob in enumerate(pred['probs']):
        if idx == label_idx:
            return prob


def get_labels_to_ints(predictor):
    vocab = predictor.model.config
    labels_to_ints = vocab.convert_tokens_to_ids('labels')
    return labels_to_ints


def get_ints_to_labels(predictor):
    vocab = predictor.tokenizer
    labels_to_ints = vocab.convert_ids_to_tokens('labels')
    return labels_to_ints


def get_predictor_tokenized(predictor, sequence):
    """_summary_

    Args:
        predictor (_type_): Transformers pipeline.
        string (string): String to be tokenized.

    Returns:
        transformers.tokenization_utils_base.BatchEncoding: Tokenized string.
    """
    return predictor.tokenizer(sequence,
                               truncation=True,
                               return_tensors="pt",
                               max_length=predictor.tokenizer.model_max_length)


def add_probs(pred):
    """ Computes predicted score from logits. """
    if 'score' not in pred.keys():
        if isinstance(pred['logits'], torch.Tensor):
            pred['score'] = torch.nn.functional.softmax(pred['logits'], dim=1)
        else:
            pred['score'] = np.exp(pred['logits'])/sum(np.exp(pred['logits'])) 
    return pred

def add_probs_batch(pred):
    """ Computes predicted score from logits. """
    if 'score' not in pred.keys():
        if isinstance(pred['logits'], torch.Tensor):
            pred['score'] = torch.nn.functional.softmax(pred['logits'], dim=0)
        else:
            pred['score'] = np.exp(pred['logits'])/sum(np.exp(pred['logits'])) 
    return pred

####################################################################
########################### Other Utils ############################
####################################################################

def wrap_text(text, num_indents=6, width=100):
    """ Util for pretty printing. """

    indent = "".join(['\t' for _ in range(num_indents)])
    return textwrap.fill(text, subsequent_indent=indent, width=width)


def html_highlight_diffs(orig, edited, tokenizer_wrapper):
    """ Given an orig and edited inputs, mark up differences in HTML. """
    
    orig = orig.replace("<br ", "<-br ").replace(" .", ".")
    edited = edited.replace("<br ", "<-br ").replace(" .", ".")

    orig_tok = tokenizer_wrapper(orig)
    edited_tok = tokenizer_wrapper(edited)

    orig_text_tok = [t.text for t in orig_tok]
    edited_text_tok = [t.text for t in edited_tok]

    edited_mark_indices, num_add, num_del = get_marked_indices(orig_text_tok, edited_text_tok, "+")
    orig_mark_indices, num_add_2, num_del_2 = get_marked_indices(orig_text_tok, edited_text_tok, "-")

    marked_original = orig 
    for idx in reversed(orig_mark_indices):
        token = orig_tok[idx]
        start, end = token.idx, token.idx + len(token)
        if start == None or end == None:
            logger.info(token, start, end)
        marked_original = marked_original[:start] + "<b>" + \
                marked_original[start:end] + "</b>" + marked_original[end:]
    
    marked_edited = edited.replace("<br />", "<-br />") 
    for idx in reversed(edited_mark_indices):
        token = edited_tok[idx]
        start, end = token.idx, token.idx + len(token)
        if start == None or end == None:
            logger.info(token, start, end)
        marked_edited = marked_edited[:start] + "<b>" + \
                marked_edited[start:end] + "</b>" + marked_edited[end:]
    return marked_original, marked_edited


def get_marked_indices(orig_tokinal, tokenized_contrast, symbol):
    """ Helper function for html_highlight_diffs. 
    Will only return indices of words deleted or replaced (not inserted). """

    index_offset = 0
    d = difflib.Differ()
    diff = d.compare(orig_tokinal, tokenized_contrast)
    list_diff = list(diff)
    tokens, modified_tokens, indices = [], [], []
    counter = 0
    additions, deletions = 0, 0

    for token_idx, token in enumerate(list_diff):
        marker = token[0]
        word = token[2:]
        if marker == symbol:        
            tokens.append(word)
            indices.append(counter)
            counter += 1
        elif marker == " ":
            modified_tokens.append(word)
            counter += 1

        if marker == "+":
            additions += 1
        if marker == "-":
            deletions += 1
            
    return indices, additions, deletions


####################################################################
########################### Masker Utils ############################
####################################################################


def get_token_offsets_from_text_field_inputs(text_field_inputs: List[Any],) -> Optional[torch.Tensor]:
    """
    Given a list of inputs to a TextFieldEmbedder, tries to find token offsets from those inputs, if
    there are any.  You will have token offsets if you are using a mismatched token embedder; if
    you're not, the return value from this function should be None.  This function is intended to be
    called from a `forward_hook` attached to a `TextFieldEmbedder`, so the inputs are formatted just
    as a list.

    It's possible in theory that you could have multiple offsets as inputs to a single call to a
    `TextFieldEmbedder`, but that's an extremely rare use case (I can't really imagine anyone
    wanting to do that).  In that case, we'll only return the first one.  If you need different
    behavior for your model, open an issue on github describing what you're doing.
    """
    for input_index, text_field_input in enumerate(text_field_inputs):
        if not isinstance(text_field_input, dict):
            continue
        for input_value in text_field_input.values():
            if not isinstance(input_value, dict):
                continue
            for embedder_arg_name, embedder_arg_value in input_value.items():
                if embedder_arg_name == "offsets":
                    return embedder_arg_value
    return None