import torch
from torch.utils.data import Dataset
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm
import numpy as np
import logging

# Local imports
from .maskers.mask_error import MaskError
from .utils import get_predictor_tokenized, format_classif_input, wrap_text, format_multiple_choice_input


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Random Number Generator
RNG = np.random.default_rng(seed=42)

class StageOneDataset(Dataset):
    """ Dataset for training Editor models in Stage One. Creates masked inputs 
    from task training inputs. Inherits from torch.utils.data.Dataset. """


    def __init__(self, tokenizer, max_length=512, masked_strings=None, targets=None):
        self.tokenizer = tokenizer
        self.masked_strings = masked_strings
        self.targets = targets
        self.max_length = max_length


    def __len__(self):
        return len(self.masked_strings)


    def __getitem__(self, index):
        input_text = self.masked_strings[index]
        label_text = self.targets[index]
        source = self.tokenizer.batch_encode_plus([input_text],
                                                  truncation=True,
                                                  padding='max_length',
                                                  pad_to_max_length=True,
                                                  max_length=self.max_length,
                                                  return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([label_text],
                                                  truncation=True,
                                                  padding='max_length',
                                                  pad_to_max_length=True,
                                                  max_length=self.max_length,
                                                  return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        # Not really used, can probably be removed
        # target_mask = target['attention_mask'].squeeze()
        eos_id = torch.LongTensor([self.tokenizer.encode(label_text,
                                                         truncation=True,
                                                         padding='max_length',
                                                         pad_to_max_length=True,
                                                         max_length=self.max_length,)[-1]])
        return {'eos_id': eos_id,
                'source_ids': source_ids.to(dtype=torch.long),
                'source_mask': source_mask.to(dtype=torch.long),
                'target_ids': target_ids.to(dtype=torch.long),
                'target_ids_y': target_ids.to(dtype=torch.long),
                }


    def create_inputs(self, orig_inputs, orig_labels, predictor, masker, target_label="pred",
                      mask_fracs=np.arange(0.2, 0.6, 0.05), mask_frac_probs=[0.125] * 8):
        target_label_options = ["pred", "gold"]
        if target_label not in target_label_options:
            error_msg = f"target_label must be in {target_label_options} "
            error_msg += f"but got '{target_label}'"
            raise ValueError(error_msg)
        
        masked_strings, targets = [], []
        # We get the label mapping from the pipeline
        labels_to_ints = predictor.model.config.label2id

        num_errors = 0
        iterator = enumerate(zip(orig_inputs, orig_labels))
        for i, (orig_inp, orig_label) in tqdm(iterator, total=len(orig_inputs), desc='create_inputs loop progress'):
            masker.mask_frac = RNG.choice(mask_fracs, 1, p=mask_frac_probs)[0]
            # This is more memory efficient than always using the predictor whether we are using gold or predicted labels
            label_to_use = predictor(orig_inp)[0]['label'] if target_label == "pred" else orig_label
            # If its not in mapping we assume it's because it is already encoded and therefore we do nothing
            label_idx = labels_to_ints[label_to_use] if label_to_use in labels_to_ints.keys() else label_to_use
            predictor_tokenized = get_predictor_tokenized(predictor, orig_inp)
            predictor_tok_end_idx = len(predictor_tokenized.input_ids)
            try:
                # Q: why make the masker return more that what will be used?
                # A: Maybe it's for stage two...
                _, _, masked_input, target = masker.get_masked_string(orig_inp,
                                                                      pred_idx=label_idx,
                                                                      predictor_tok_end_idx=predictor_tok_end_idx)
                masked_string = format_classif_input(masked_input, label_to_use) 
                masked_strings.append(masked_string)
                targets.append(target)
                
                verbose = True if i % 500 == 0 else False
                if verbose:
                    rounded_mask_frac = round(masker.mask_frac, 3)
                    logger.info(wrap_text(f"Original input ({i}): " + orig_inp))
                    logger.info(wrap_text(f"Mask frac: {rounded_mask_frac}"))
                    logger.info(wrap_text(f"Editor input: {masked_string}"))
                    logger.info(wrap_text(f"Editor target: {target}"))
                    logger.info(wrap_text(f"Errors: {num_errors}"))
                
            except MaskError as e:
                num_errors += 1
                print('Error')
                print(e)


        self.masked_strings = masked_strings
        self.targets = targets

class RaceStageOneDataset(StageOneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_inputs(self, dataset_reader, orig_inputs, orig_labels, predictor, masker,
                      mask_fracs=np.arange(0.2, 0.6, 0.05), mask_frac_probs=[0.125] * 8, editable_key="article", target_label="pred"):
            
        editable_keys = ["article", "question"]
        if editable_key not in editable_keys:
            raise ValueError(f"Editable key must be in {editable_keys} \
                    but got value {editable_key}")
        
        labels_to_ints = predictor.model.config.label2id
        
        num_errors = 0
        masked_strings, targets = [], []
        
        iterator = enumerate(zip(orig_inputs, orig_labels))
        for i, (orig_inp, gold_label) in tqdm(iterator, total=len(orig_inputs)):
            masker.mask_frac = np.random.choice(mask_fracs, 1, p=mask_frac_probs)[0] 

            instance, length_lst, _ = dataset_reader.text_to_instance(orig_inp["id"],
                                                                      orig_inp["article"],
                                                                      orig_inp["question"],
                                                                      orig_inp["options"])
            options = orig_inp["options"]
            pred = predictor.predict_instance(instance)
            pred_label = int(pred['best_alternative'])

            # For RACE, label is already int, not string
            label_idx = pred_label if target_label == "pred" else gold_label 

            try:
                # Mask the article
                if editable_key == "article":
                    article_tok = get_predictor_tokenized(predictor, orig_inp["article"])
                    predictor_tok_end_idx = min(len(article_tok), length_lst[label_idx])
                    _, _, masked_article, target = masker.get_masked_string(orig_inp["article"], label_idx,
                                                                                               labeled_instance=instance, predictor_tok_end_idx=predictor_tok_end_idx)
                    question = orig_inp["question"]
                    article = masked_article

                # Mask the question
                # TODO: Does this work? Have only tested article
                elif editable_key == "question":
                    question_tok = get_predictor_tokenized(predictor, orig_inp["question"]) 
                    predictor_tok_end_idx = length_lst[label_idx] + len(question_tok)
                    _, _, masked_question, target = masker.get_masked_string(orig_inp["question"], label_idx, labeled_instance=instance,
                                                                                                predictor_tok_start_idx=length_lst[label_idx],
                                                                                                predictor_tok_end_idx=predictor_tok_end_idx)
                    question = masked_question
                    article = orig_inp["article"]

                masked_string = format_multiple_choice_input(article, question, options, label_idx)
                masked_strings.append(masked_string)
                targets.append(target)
                
            except MaskError as e:
                num_errors += 1
                print('Error')
                print(e)

        self.masked_strings = masked_strings
        self.targets = targets
