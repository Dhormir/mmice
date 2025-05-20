import torch
import numpy as np
import re
import os
import more_itertools as mit
import math
import logging
from tqdm.auto import tqdm
import sys

# Local imports
from .utils import get_device, get_predictor_tokenized, wrap_text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Editor():
    def __init__(self, editor_tokenizer, editor_model, masker,
                 num_gens=15,
                 num_beams=30,
                 grad_pred="contrast",
                 generate_type="sample",
                 no_repeat_ngram_size=2,
                 top_k=30,
                 top_p=0.92,
                 length_penalty=0.5,
                 verbose=True,
                 prepend_label=True,
                 ):
        self.device = get_device()
        self.num_gens = num_gens
        self.tokenizer = editor_tokenizer
        self.editor_model = editor_model.to(self.device)
        self.max_length = editor_tokenizer.model_max_length
        self.masker = masker
        self.predictor = self.masker.predictor
        self.predictor_ints_to_labels = self.masker.predictor.model.config.id2label
        self.predictor_labels_to_ints = self.masker.predictor.model.config.label2id
        self.grad_pred = grad_pred
        self.verbose = verbose 
        self.generate_type = generate_type
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.top_k = top_k
        self.top_p = top_p
        self.length_penalty = length_penalty 
        self.num_beams = num_beams
        self.prepend_label = prepend_label

    def get_editor_input(self, targ_pred_label, masked_editable_seg):
        """ Format input for editor """
        prefix = "" if not self.prepend_label else "label: " + \
                targ_pred_label + ". input: " 
        return prefix + masked_editable_seg

    # Why do these methods exist??
    def get_editable_seg_from_input(self, inp):
        """ Map whole input -> editable seg. 
        These are the same for single-input classification. """
        return inp
    # It does Nothing why??????
    def get_input_from_editable_seg(self, inp, editable_seg):
        """ Map whole input -> editable seg. 
        These are the same for IMDB/Newsgroups. """
        return editable_seg 

    # Why manually truncate when its already available in transformers ???
    def truncate_editable_segs(self, editable_segs, **kwargs):
        """ Truncate editable segments to max length of Predictor. """
        trunc_es = [None] * len(editable_segs)
        for s_idx, s in enumerate(editable_segs):
            assert(len(s) > 0)
            predic_tokenized = get_predictor_tokenized(self.predictor, s)
            max_predic_tokens = self.predictor.tokenizer.model_max_length
            if len(predic_tokenized["input_ids"][0]) >= max_predic_tokens: 
                trunc_es[s_idx] = self.predictor.tokenizer.decode(
                    predic_tokenized["input_ids"][0],
                    skip_special_tokens=True)
            else:
                trunc_es[s_idx] = s
        return trunc_es

    def input_to_instance(self, inp, editable_seg=None, return_tuple=False):
        """ Convert input to Transformers BatchEncoding object """

        if editable_seg is None:
            batchencoding = self.predictor.tokenizer(inp)
        else:
            batchencoding = self.predictor.tokenizer(editable_seg)
        if return_tuple:
            # TODO: hacky bc for race dataset reader, we return length list
            return batchencoding, [None] 
        return batchencoding

    def get_sorted_token_indices(self, inp, grad_pred_idx):
        """ Get token indices to mask, sorted by gradient value """
        # this first call quite literally does nothing!!!
        editable_seg = inp
        editor_tokenized = self.tokenizer(
            editable_seg,
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        editable_seg = self.tokenizer.decode(editor_tokenized["input_ids"][0][:-1])
        sorted_token_indices = self.masker.get_important_editor_tokens(editable_seg, grad_pred_idx, editor_tokenized,
                                                                       num_return_toks=len(editor_tokenized.input_ids[0]))
        return sorted_token_indices 

    def get_candidates(self, targ_pred_label, inp, targ_pred_idx, orig_pred_idx,
                       sorted_token_indices=None):
        """ Gets edit candidates after infilling with Editor. 
        Returns dicts with edited inputs (i.e. whole inputs, dicts in the case
        of RACE) and edited editable segs (i.e. just the parts of inputs
        that are editable, articles in the case of RACE). """
        assert targ_pred_idx != orig_pred_idx

        if self.grad_pred == "contrast":
            grad_pred_idx = targ_pred_idx 
        elif self.grad_pred == "original":
            grad_pred_idx = orig_pred_idx 
        else:
            raise ValueError

        num_spans, _, masked_inp, orig_spans, max_length = \
                self._prepare_input_for_editor(inp, grad_pred_idx,
                                               sorted_token_indices=sorted_token_indices)

        if "t5" in self.tokenizer.name_or_path:
            edited_editable_segs = self._sample_edits(targ_pred_label, masked_inp, targ_pred_idx,
                                                    num_spans=num_spans,
                                                    orig_spans=orig_spans,
                                                    max_length=max_length)
        elif "bert" in self.tokenizer.name_or_path:
            edited_editable_segs = self._sample_edits_bert(targ_pred_label, masked_inp, targ_pred_idx,
                                                           num_spans=num_spans,
                                                           orig_spans=orig_spans,
                                                           max_length=max_length)
        else:
            raise NotImplementedError(f"Model {self.editor_tok_wrapper.name_or_path} not implemented; \
                must be bert or t5 style")
            
        assert len(edited_editable_segs) > 0, "Generated zero edited_editable_segs"

        edited_cands = [None] * len(edited_editable_segs)
        for idx, es in enumerate(edited_editable_segs):
            cand = {}
            es = self.truncate_editable_segs([es], inp=inp)[0]
            # this does nothing
            cand['edited_input'] = self.get_input_from_editable_seg(inp, es) 
            cand['edited_editable_seg'] = es 
            edited_cands[idx] = cand
        return edited_cands, masked_inp

    def _prepare_input_for_editor(self, inp, grad_pred_idx, sorted_token_indices=None):
        """ Helper function that prepares masked input for Editor. """
        tokenized_input = self.tokenizer(inp)
        # we heuristically eliminate the last token which is end of sentence
        tokens = tokenized_input.tokens()[:-1]
       
        if sorted_token_indices is not None: 
            num_return_toks = math.ceil(self.masker.mask_frac * len(tokens))
            token_ind_to_mask = sorted_token_indices[:num_return_toks]
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(inp, editor_mask_indices=token_ind_to_mask)
        else:
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(inp, grad_pred_idx)

        max_length = math.ceil((self.masker.mask_frac + 0.2) * \
                len(sorted_token_indices))
        num_spans = len(grouped_ind_to_mask)

        return num_spans, token_ind_to_mask, masked_inp, orig_spans, max_length

    def _process_gen(self, masked_inp, gen, sentinel_toks):
        """ Helper function that processes decoded gen """
        bad_gen = False
        first_bad_tok = None

        # Hacky: If sentinel tokens are consecutive, then re split won't work
        gen = gen.replace("><extra_id", "> <extra_id")

        # Remove <pad> prefix, etc.
        gen = gen[gen.find("<extra_id_0>"):]

        # Sanity check
        assert not gen.startswith(self.tokenizer.pad_token) 

        # This is for baseline T5 which does not handle masked last tokens well.
        # Baseline often predicts </s> as last token instead of sentinel tok.
        # Heuristically treating the first </s> tok as the final sentinel tok.
        # TODO: will this mess things up for non-baseline editor_models?
        if sentinel_toks[-1] not in gen:
            first_eos_token_idx = gen.find(self.tokenizer.eos_token)
            gen = gen[:first_eos_token_idx] + sentinel_toks[-1] + \
                    gen[first_eos_token_idx + len(self.tokenizer.eos_token):]

        last_sentin_idx = gen.find(sentinel_toks[-1])
        if last_sentin_idx != -1:
        # If the last token we are looking for is in generation, truncate 
            gen = gen[:last_sentin_idx + len(sentinel_toks[-1])]

        # If </s> is in generation, truncate 
        eos_token_idx = gen.find(self.tokenizer.eos_token)
        if eos_token_idx != -1:
            gen = gen[:eos_token_idx]

        # Check if every sentinel token is in the gen
        for x in sentinel_toks:
            if x not in gen:
                bad_gen = True
                first_bad_tok = self.tokenizer.encode(x)[0]
                break
        
        tokens = list(filter(None, re.split('<extra_id_.>|<extra_id_..>|<extra_id_...>', gen)))
        gen_sentinel_toks = re.findall('<extra_id_.>|<extra_id_..>|<extra_id_...>', gen)

        gen_sentinel_toks = gen_sentinel_toks[:len(tokens)]

        temp = masked_inp 
        ctr = 0
        prev_temp = temp
        tok_sentinel_iterator = zip(tokens, gen_sentinel_toks)
        for idx, (token, sentinel_tok) in enumerate(tok_sentinel_iterator):
            sentinel_idx = sentinel_tok[-2:-1] if len(sentinel_tok) == 12 \
                    else sentinel_tok[-3:-1]
            sentinel_idx = int(sentinel_idx)
            
            # Check order of generated sentinel tokens 
            if sentinel_idx != ctr:
                first_bad_tok = self.tokenizer.encode(f"<extra_id_{ctr}>")[0]
                bad_gen = True
                break

            if idx != 0:
                temp = temp.replace(prev_sentinel_tok, prev_token)
            prev_sentinel_tok = sentinel_tok
            prev_token = token
            
            # If last replacement, make sure final sentinel token was generated
            is_last = (idx == len(tokens)-1)
            if is_last and gen_sentinel_toks[-1] in sentinel_toks and not bad_gen:
                if " " + sentinel_tok in temp:
                    temp = temp.replace(" " + sentinel_tok, token)
                elif "-" + sentinel_tok in temp:
                    # If span follows "-" character, remove first white space
                    if token[0] == " ":
                        token = token[1:]
                    temp = temp.replace(sentinel_tok, token)
                else:
                    temp = temp.replace(sentinel_tok, token)
            else:
                first_bad_tok = self.tokenizer.encode("<extra_id_{ctr}>")[0]
            ctr += 1

        return bad_gen, first_bad_tok, temp, gen 


    def _get_pred_with_replacement(self, temp_gen, orig_spans):
        """ Replaces sentinel tokens in gen with orig text and returns pred. 
        Used for intermediate bad generations. """
        orig_tokens = list(filter(None, re.split('<extra_id_.>|<extra_id_..>', orig_spans)))
        orig_sentinel_toks = re.findall('<extra_id_.>|<extra_id_..>', orig_spans)

        for token, sentinel_tok in zip(orig_tokens, orig_sentinel_toks[:-1]):
            if sentinel_tok in temp_gen:
                temp_gen = temp_gen.replace(sentinel_tok, token)
        return temp_gen, self.predictor(temp_gen)[0]
        
        
    def _sample_edits(self, targ_pred_label, masked_editable_seg, targ_pred_idx,
                      num_spans=None, orig_spans=None, max_length=None):
        """ Returns self.num_gens copies of masked_editable_seg with infills.
        Called by get_candidates(). """
        
        self.editor_model.eval()       

        editor_input = self.get_editor_input(targ_pred_label, masked_editable_seg)
        
        editor_inputs = [editor_input]
        editable_segs = [masked_editable_seg]
        span_end_offsets = [num_spans]
        orig_token_ids_lst = [self.tokenizer.encode(orig_spans)[:-1]]
        orig_spans_lst = [orig_spans]
        masked_token_ids_lst = [self.tokenizer.encode(editor_input)[:-1]]

        k_intermediate = 3 

        sentinel_start = self.tokenizer.encode("<extra_id_99>")[0]
        sentinel_end = self.tokenizer.encode("<extra_id_0>")[0]

        num_sub_rounds = 0
        edited_editable_segs = [] # list of tuples with meta information

        max_sub_rounds = 3
        while editable_segs != []:
       
            # Break if past max sub rounds 
            if num_sub_rounds > max_sub_rounds:
                break
            
            new_editor_inputs = []
            new_editable_segs = []
            new_span_end_offsets = []
            new_orig_token_ids_lst = []
            new_orig_spans_lst = []
            new_masked_token_ids_lst = []

            iterator = enumerate(zip(
                editor_inputs, editable_segs, masked_token_ids_lst, 
                span_end_offsets, orig_token_ids_lst, orig_spans_lst))
            for inp_idx, (editor_input, editable_seg, masked_token_ids, \
                    span_end, orig_token_ids, orig_spans) in iterator: 

                num_inputs = len(editor_inputs)
                num_return_seqs = int(math.ceil(self.num_gens/num_inputs)) \
                        if num_sub_rounds != 0 else self.num_gens
                num_beams = self.num_beams if num_sub_rounds == 0 \
                        else num_return_seqs
                last_sentin = f"<extra_id_{span_end}>"
                end_token_id = self.tokenizer.convert_tokens_to_ids(last_sentin)
                # ToDo: this will fuck up with bert style architectures
                eos_token_id = self.tokenizer.eos_token_id
                masked_token_ids_tensor = torch.LongTensor(masked_token_ids + [eos_token_id]).unsqueeze(0).to(self.device)
                bad_tokens_ids = [[x] for x in range(sentinel_start, end_token_id)] + [[eos_token_id]]
                max_length = max(int(4/3 * max_length), 200)
                #logger.info(wrap_text(f"max lenght:{max_length}\nmax new tokens: {max_length - len(masked_token_ids_tensor.squeeze())}"))
                logger.info(wrap_text(f"Sub round: {num_sub_rounds}"))    
                logger.info(wrap_text(f"Input: {inp_idx + 1} of {num_inputs}"))
                logger.info(wrap_text(f"Last sentinel: {last_sentin}"))
                logger.info(wrap_text("INPUT TO EDITOR: " + \
                        f"{self.tokenizer.decode(masked_token_ids)}"))

                with torch.no_grad():
                    if self.generate_type == "beam":
                        output = self.editor_model.generate(
                            input_ids=masked_token_ids_tensor, 
                            num_beams=num_beams, 
                            num_return_sequences=num_return_seqs, 
                            no_repeat_ngram_size=self.no_repeat_ngram_size, 
                            eos_token_id=end_token_id, 
                            early_stopping=True if num_beams > 1 else False,
                            length_penalty=self.length_penalty, 
                            bad_words_ids=bad_tokens_ids, 
                            max_length=max_length
                            ) 

                    elif self.generate_type == "sample":
                        output = self.editor_model.generate(
                            input_ids=masked_token_ids_tensor, 
                            do_sample=True,
                            top_p=self.top_p, 
                            top_k=self.top_k, 
                            num_return_sequences=num_return_seqs, 
                            no_repeat_ngram_size=self.no_repeat_ngram_size, 
                            eos_token_id=end_token_id,
                            length_penalty=self.length_penalty,
                            bad_words_ids=bad_tokens_ids, 
                            max_length=max_length,
                            ) 
                output = output.cpu()
                del masked_token_ids_tensor 
                torch.cuda.empty_cache()
                batch_decoded = self.tokenizer.batch_decode(output)
                num_gens_with_pad = 0
                num_bad_gens = 0
                temp_edited_editable_segs = []
                logger.info(wrap_text(f"first batch: {batch_decoded[0]}"))
                for batch_idx, batch in enumerate(batch_decoded):
                    sentinel_toks = [f"<extra_id_{idx}>" for idx in \
                            range(0, span_end + 1)]
                    bad_gen, first_bad_tok, temp, stripped_batch = \
                            self._process_gen(editable_seg, batch, sentinel_toks)

                    if len(sentinel_toks) > 3: 
                        assert sentinel_toks[-2] in editor_input

                    if "<pad>" in batch[4:]:
                        num_gens_with_pad += 1
                    if bad_gen:
                        num_bad_gens += 1
                        temp_span_end_offset = first_bad_tok - end_token_id + 1

                        new_editable_token_ids = np.array(
                                self.tokenizer.encode(temp)[:-1])

                        sentinel_indices = np.nonzero(
                                (new_editable_token_ids >= sentinel_start) & \
                                (new_editable_token_ids <= sentinel_end))[0]
                        
                        new_first_token = max(new_editable_token_ids[sentinel_indices])

                        diff = sentinel_end - new_first_token
                        new_editable_token_ids[sentinel_indices] += diff
                        
                        new_span_end_offsets.append(len(sentinel_indices))

                        new_editable_seg = self.tokenizer.decode(new_editable_token_ids)
                        new_editable_segs.append(new_editable_seg)
                        
                        new_input = self.get_editor_input(targ_pred_label, new_editable_seg)

                        new_masked_token_ids = self.tokenizer.encode(new_input)[:-1]
                        new_masked_token_ids_lst.append(new_masked_token_ids)

                        # Hacky but re-decode to remove spaces b/w sentinels
                        new_editor_input = self.tokenizer.decode(new_masked_token_ids)
                        new_editor_inputs.append(new_editor_input)

                        # Get orig token ids from new first token and on
                        new_orig_token_idx = np.nonzero(orig_token_ids == new_first_token)
                        new_orig_token_ids = np.array(orig_token_ids[new_orig_token_idx[0][0]:]) 
                        sentinel_indices = np.nonzero((new_orig_token_ids >= sentinel_start) & \
                                (new_orig_token_ids <= sentinel_end))[0]
                        new_orig_token_ids[sentinel_indices] += diff
                        new_orig_token_ids_lst.append(new_orig_token_ids)
                        new_orig_spans = self.tokenizer.decode(new_orig_token_ids)
                        new_orig_spans_lst.append(new_orig_spans)

                    else:
                        # All generations must be editable in order to be useful
                        temp_edited_editable_segs.append(temp)
                        assert "<extra_id" not in temp, "We didnt generate editable segments"

                    assert "</s>" not in temp
                edited_editable_segs.extend(temp_edited_editable_segs)
            if new_editor_inputs == []:
                break

            _, unique_batch_indices = np.unique(new_editor_inputs, return_index=True)

            targ_probs = [-1] * len(new_editable_segs)
            for idx in unique_batch_indices:
                ot = new_orig_spans_lst[idx].replace("<pad>", "")
                temp, edit_preds = self._get_pred_with_replacement(new_editable_segs[idx], ot)
                # predictions are always sorted by score from higher to lower
                edit_probs = [edit_pred['score'] for edit_pred in edit_preds]
                edit_labels = [edit_pred['label'] for edit_pred in edit_preds]
                targ_pred_idx = targ_pred_idx if edit_labels[targ_pred_idx] == targ_pred_label else edit_labels.index(targ_pred_label)
                #preds = add_probs(preds)
                targ_probs[idx] = edit_probs[targ_pred_idx]
                predicted_label = edit_preds[0]["label"]
                contrast_label = edit_labels[targ_pred_idx]
                if predicted_label == contrast_label: 
                    edited_editable_segs.append(temp)

            highest_indices = np.argsort(targ_probs)[-k_intermediate:]
            filt_indices = [idx for idx in highest_indices \
                    if targ_probs[idx] != -1]
            editor_inputs = [new_editor_inputs[idx] for idx in filt_indices]
            editable_segs = [new_editable_segs[idx] for idx in filt_indices]
            span_end_offsets = [new_span_end_offsets[idx] for idx in filt_indices] 
            orig_token_ids_lst = [new_orig_token_ids_lst[idx] for idx in filt_indices] 
            orig_spans_lst = [new_orig_spans_lst[idx] for idx in filt_indices] 
            masked_token_ids_lst = [new_masked_token_ids_lst[idx] for idx in filt_indices] 

            sys.stdout.flush()
            num_sub_rounds += 1

        for idx, es in enumerate(edited_editable_segs):
            assert es.find("</s>") in [len(es)-4, -1]
            edited_editable_segs[idx] = es.replace("</s>", " ")
            assert "<extra_id_" not in es, \
                    f"Extra id token should not be in edited inp: {es}"
            assert "</s>" not in es, \
                    f"</s> should not be in edited inp: {edited_editable_segs[idx][0]}"


        return set(edited_editable_segs)


    def _sample_edits_bert(self, targ_pred_label, masked_editable_seg, targ_pred_idx,
                           num_spans=None, orig_spans=None, max_length=None):
        """ Returns self.num_gens copies of masked_editable_seg with infills.
        Called by get_candidates(). """
        
        self.editor_model.eval()       

        editor_input = self.get_editor_input(targ_pred_label, masked_editable_seg)
        
        editor_inputs = [editor_input]
        editable_segs = [masked_editable_seg]
        span_end_offsets = [num_spans]
        orig_token_ids_lst = [self.tokenizer.encode(orig_spans)[1:-1]]
        orig_spans_lst = [orig_spans]
        masked_token_ids_lst = [self.tokenizer.encode(editor_input)[1:-1]]
        k_intermediate = 3 

        mask_token = self.tokenizer.encode("[MASK]")
        logger.info(f'mask token: {mask_token}')

        num_sub_rounds = 0
        edited_editable_segs = [] # list of tuples with meta information

        max_sub_rounds = 3
        while editable_segs != []:
            # Break if past max sub rounds 
            if num_sub_rounds > max_sub_rounds:
                break
            
            new_editor_inputs = []
            new_editable_segs = []
            new_span_end_offsets = []
            new_orig_token_ids_lst = []
            new_orig_spans_lst = []
            new_masked_token_ids_lst = []

            iterator = enumerate(zip(
                editor_inputs, editable_segs, masked_token_ids_lst, 
                span_end_offsets, orig_token_ids_lst, orig_spans_lst))
            for inp_idx, (editor_input, editable_seg, masked_token_ids, \
                    span_end, orig_token_ids, orig_spans) in iterator: 

                num_inputs = len(editor_inputs)
                num_return_seqs = int(math.ceil(self.num_gens/num_inputs)) \
                        if num_sub_rounds != 0 else self.num_gens
                num_beams = self.num_beams if num_sub_rounds == 0 \
                        else num_return_seqs
                
                masked_token_ids_tensor = torch.LongTensor(masked_token_ids).unsqueeze(0).to(self.device)
                
                max_length = max(int(4/3 * max_length), 200)
                logger.info(wrap_text(f"Sub round: {num_sub_rounds}"))    
                logger.info(wrap_text(f"Input: {inp_idx + 1} of {num_inputs}"))
                logger.info(wrap_text("INPUT TO EDITOR: " + \
                        f"{self.tokenizer.decode(masked_token_ids)}"))

                batch_decoded = get_beam_prediction(sent=editable_seg, tokenizer=self.tokenizer,
                                                    model=self.editor_model, device=self.device,
                                                    k=num_return_seqs, fill_order='right')

                num_gens_with_pad = 0
                num_bad_gens = 0
                temp_edited_editable_segs = []
                logger.info(wrap_text(f"first batch: {batch_decoded[0]}"))
                for batch_idx, batch in enumerate(batch_decoded):
                    temp = batch
                    if "<pad>" in batch[4:]:
                        num_gens_with_pad += 1

                    temp_edited_editable_segs.append(temp)
                    assert "<extra_id" not in temp
                    
                    assert "</s>" not in temp

                edited_editable_segs.extend(temp_edited_editable_segs)
            if new_editor_inputs == []:
                break

            _, unique_batch_indices = np.unique(new_editor_inputs, return_index=True)

            targ_probs = [-1] * len(edited_editable_segs)
            for idx in enumerate(edited_editable_segs):
                ot = new_orig_spans_lst[idx].replace("<pad>", "")
                temp = edited_editable_segs[idx]
                edit_preds = self.predictor(temp)[0]
                # predictions are always sorted by score from higher to lower
                edit_probs = [edit_pred['score'] for edit_pred in edit_preds]
                edit_labels = [edit_pred['label'] for edit_pred in edit_preds]
                targ_pred_idx = targ_pred_idx if edit_labels[targ_pred_idx] == targ_pred_label else edit_labels.index(targ_pred_label)
                #preds = add_probs(preds)
                targ_probs[idx] = edit_probs[targ_pred_idx]
                predicted_label = edit_preds[0]["label"]
                contrast_label = edit_labels[targ_pred_idx]
                if predicted_label == contrast_label: 
                    edited_editable_segs.append(temp)
            
            highest_indices = np.argsort(targ_probs)[-k_intermediate:]
            filt_indices = [idx for idx in highest_indices \
                    if targ_probs[idx] != -1]
            editor_inputs = [new_editor_inputs[idx] for idx in filt_indices]
            editable_segs = [new_editable_segs[idx] for idx in filt_indices]
            span_end_offsets = [new_span_end_offsets[idx] for idx in filt_indices] 
            orig_token_ids_lst = [new_orig_token_ids_lst[idx] for idx in filt_indices] 
            orig_spans_lst = [new_orig_spans_lst[idx] for idx in filt_indices] 
            masked_token_ids_lst = [new_masked_token_ids_lst[idx] for idx in filt_indices] 

            sys.stdout.flush()
            num_sub_rounds += 1

        for idx, es in enumerate(edited_editable_segs):
            assert es.find("</s>") in [len(es)-4, -1]
            edited_editable_segs[idx] = es.replace("</s>", " ")
            assert "<extra_id_" not in es, \
                    f"Extra id token should not be in edited inp: {es}"
            assert "</s>" not in es, \
                    f"</s> should not be in edited inp: {edited_editable_segs[idx][0]}"


        return set(edited_editable_segs) 

# ToDO: Change this
class RaceEditor(Editor):
    def __init__(
            self, 
            tokenizer_wrapper, 
            tokenizer, 
            editor_model, 
            masker, 
            num_gens = 30, 
            num_beams = 30, 
            grad_pred = "contrast", 
            generate_type = "sample", 
            length_penalty = 1.0, 
            no_repeat_ngram_size = 2, 
            top_k = 50, 
            top_p = 0.92, 
            verbose = False, 
            editable_key = "article"
        ):
        super().__init__(
                tokenizer_wrapper, tokenizer, editor_model, masker, 
                num_gens=num_gens, num_beams=num_beams, 
                ints_to_labels=[str(idx) for idx in range(4)], 
                grad_pred=grad_pred, 
                generate_type=generate_type, 
                no_repeat_ngram_size=no_repeat_ngram_size, 
                top_k=top_k, top_p=top_p, 
                length_penalty=length_penalty, 
                verbose=verbose)
        
        self.editable_key = editable_key
        if self.editable_key not in ["question", "article"]:
            raise ValueError("Invalid value for editable_key")

    def _get_pred_with_replacement(self, temp_gen, orig_spans, inp):
        """ Replaces sentinel tokens in gen with orig text and returns pred. 
        Used for intermediate bad generations. """

        orig_tokens = list(filter(None, re.split(
            '<extra_id_.>|<extra_id_..>|<extra_id_...>', orig_spans)))
        orig_sentinel_toks = re.findall(
                '<extra_id_.>|<extra_id_..>|<extra_id_...>', orig_spans)

        for token, sentinel_tok in zip(orig_tokens, orig_sentinel_toks[:-1]):
            if sentinel_tok in temp_gen:
                temp_gen = temp_gen.replace(sentinel_tok, token)
        # temp_gen is article for RACE
        temp_instance = self.tokenizer(inp["id"],
                                       temp_gen,
                                       inp["question"],
                                       inp["options"])[0]
        return temp_gen, self.predictor.predict_instance(temp_instance)

    def get_editable_seg_from_input(self, inp):
        """ Map whole input -> editable seg. """ 
        
        return inp[self.editable_key]

    def get_input_from_editable_seg(self, inp, editable_seg):
        """ Map editable seg -> whole input. """ 

        new_inp = inp.copy()
        new_inp[self.editable_key] = editable_seg
        return new_inp

    def truncate_editable_segs(self, editable_segs, inp = None):
        """ Truncate editable segments to max length of Predictor. """ 
        
        trunc_inputs = [None] * len(editable_segs)
        instance, length_lst, max_length_lst = self.input_to_instance(inp, return_tuple=True)
        for s_idx, es in enumerate(editable_segs):
            editable_toks = get_predictor_tokenized(self.predictor, es)
            predic_tok_end_idx = len(editable_toks)
            predic_tok_end_idx = min(predic_tok_end_idx, max(max_length_lst))
            last_index = editable_toks[predic_tok_end_idx - 1].idx_end
            editable_seg = es[:last_index]
            trunc_inputs[s_idx] = editable_seg
        return trunc_inputs

    def get_editor_input(self, targ_pred_label, masked_editable_seg, inp):
        """ Format input for editor """
        
        options = inp["options"]
        if masked_editable_seg is None:
            article = inp["article"]
            question = inp["question"]
        else: # masked editable input given
            if self.editable_key == "article":
                article = masked_editable_seg
                question = inp["question"]
            elif self.editable_key == "question":
                article = inp["article"] 
                question = masked_editable_seg 

        editor_input = format_multiple_choice_input(
                article, question, options, int(targ_pred_label))
        return editor_input

    def input_to_instance(self, inp, editable_seg=None, return_tuple=False):
        """ Convert input to AllenNLP instance object """
        
        if editable_seg is None:
            article = inp["article"]
            question = inp["question"]
        else: # editable input given
            if self.editable_key == "article":
                article = editable_seg
                question = inp["question"]
            elif self.editable_key == "question":
                article = inp["article"] 
                question = editable_seg
        output = self.tokenizer(inp["id"],
                                article,
                                question,
                                inp["options"])
        if return_tuple:
            return output
        return output[0]

    def get_sorted_token_indices(self, inp, grad_pred_idx):
        """ Get token indices to mask, sorted by gradient value """

        editable_seg = self.get_editable_seg_from_input(inp)

        inst, length_lst, _ = self.input_to_instance(inp, return_tuple=True)
        editable_toks = get_predictor_tokenized(self.predictor, editable_seg)
        num_editab_toks = len(editable_toks)

        predic_tok_end_idx = len(editable_toks)
        predic_tok_end_idx = min(predic_tok_end_idx, length_lst[grad_pred_idx])
        
        if self.editable_key == "article":
            predic_tok_start_idx = 0 
        elif self.editable_key == "question":
            predic_tok_start_idx = length_lst[grad_pred_idx]
            predic_tok_end_idx = length_lst[grad_pred_idx] + num_editab_toks 
        
        editable_toks = self.tokenizer_wrapper.tokenize(editable_seg)[:-1]
        sorted_token_indices = self.masker.get_important_editor_tokens(
                editable_seg, grad_pred_idx, editable_toks,
                num_return_toks=len(editable_toks), 
                labeled_instance=inst, 
                predic_tok_end_idx=predic_tok_end_idx, 
                predic_tok_start_idx=predic_tok_start_idx)
        return sorted_token_indices 
        
    def _prepare_input_for_editor(self, inp, targ_pred_idx, grad_pred_idx,
            sorted_token_indices = None):
        """ Helper function that prepares masked input for Editor. """

        editable_seg = self.get_editable_seg_from_input(inp)

        tokens = [t.text for t in \
                self.tokenizer_wrapper.tokenize(editable_seg)[:-1]]

        instance, length_lst, _ = self.input_to_instance(
                inp, return_tuple=True)
        editable_toks = get_predictor_tokenized(self.predictor, editable_seg) 
        num_editab_toks = len(editable_toks)
        predic_tok_end_idx = len(editable_toks)
        predic_tok_end_idx = min(
                predic_tok_end_idx, length_lst[grad_pred_idx])
        
        if self.editable_key == "article":
            predic_tok_start_idx = 0 
        elif self.editable_key == "question":
            predic_tok_start_idx = length_lst[grad_pred_idx]
            predic_tok_end_idx = length_lst[grad_pred_idx] + num_editab_toks

        if sorted_token_indices is not None: 
            num_return_toks = math.ceil(
                    self.masker.mask_frac * len(tokens))
            token_ind_to_mask = sorted_token_indices[:num_return_toks]

            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(editable_seg, grad_pred_idx, 
                            editor_mask_indices=token_ind_to_mask, 
                            predic_tok_start_idx=predic_tok_start_idx, 
                            predic_tok_end_idx=predic_tok_end_idx)

        else:
            grouped_ind_to_mask, token_ind_to_mask, masked_inp, orig_spans = \
                    self.masker.get_masked_string(
                            editable_seg, grad_pred_idx, 
                            labeled_instance=instance, 
                            predic_tok_end_idx=predic_tok_end_idx, 
                            predic_tok_start_idx=predic_tok_start_idx)

        num_spans = len(grouped_ind_to_mask)
        max_length = math.ceil(
                (self.masker.mask_frac+0.2) * len(sorted_token_indices))

        masked_inp = masked_inp.replace(self.tokenizer.eos_token, " ")
        return num_spans, token_ind_to_mask, masked_inp, orig_spans, max_length


# Beam search version
def get_beam_prediction(sent, tokenizer, model, k=2):
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]

    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()
    curr_beam = []

    for index, mask_index in tqdm(enumerate(masked_pos), total=len(masked_pos)):
        mask_hidden_state = last_hidden_state[mask_index]
        rankings, idx = torch.topk(mask_hidden_state, k=k, dim=0, sorted=True)

        if len(curr_beam) == 0:
          curr_beam = list(zip(rankings, fill_mask_sentence(mask_index, idx, token_ids.squeeze())))
        else:
          prev_beam = curr_beam.copy()
          topk_list = list(zip(rankings, idx))
          while len(topk_list) != 0:
            value = topk_list.pop()
            for index, cand in enumerate(curr_beam):
              if cand == prev_beam[index] and cand[0] + value[0] > cand[0]:
                curr_beam[index] = (cand[0] + value[0], fill_mask_sentence(mask_index, value[1], cand[1]))
              elif cand != prev_beam[index] and prev_beam[index][0] + value[0] > curr_beam[index][0]:
                curr_beam[index] = (prev_beam[index][0] + value[0], fill_mask_sentence(mask_index, value[1], prev_beam[index][1]))

    # guesses sorted in descending probability order
    for cand in curr_beam:
      print(f'cand: {tokenizer.decode(cand[1])}')

    return curr_beam

def fill_mask_sentence(mask_index, fill_token_ids, token_ids):
  filled_sentences = []
  if fill_token_ids.numel() < 2:
    token_ids.squeeze()[mask_index] = fill_token_ids
    return token_ids.clone()
  for idx in fill_token_ids:
    token_ids.squeeze()[mask_index] = idx
    filled_sentences.append(token_ids.clone())
  token_ids.squeeze()[mask_index] = 0
  return filled_sentences

# We pass the masked sentence several times until we fill it
def get_prediction_II(sent, tokenizer, model, device, k=2, fill_order='left'):
    order = {'left': 1, 'right': -1}
    token_ids = tokenizer.encode(sent,
                                 truncation=True,
                                 return_tensors='pt').to(device)
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position][::order[fill_order]]
    list_of_ids = [token_ids]
    for index, mask_index in enumerate(masked_pos):
      list_of_guesses = []
      for ids in list_of_ids:
        with torch.no_grad():
          output = model(ids)
        last_hidden_state = output[0].squeeze()
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=k, dim=0)[1]
        list_of_guesses += fill_mask_sentence(mask_index, idx, ids)
      list_of_ids = list_of_guesses[:k]

    torch.cuda.empty_cache()
    return [tokenizer.decode(guess.squeeze()).replace('[CLS]', '').replace('[SEP]', '') for guess in list_of_ids]