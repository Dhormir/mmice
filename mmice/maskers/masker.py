import more_itertools as mit
import logging

from .mask_error import MaskError
from ..utils import logger

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)


class Masker():
    """
    Class used to mask inputs for Editors.
    Two subclasses: RandomMasker and GradientMasker

    mask_frac: float
        Fraction of input tokens to mask.
    editor_to_wrapper: transformers.PreTrainedTokenizer
        Wraps around Editor tokenizer.
        Has capabilities for mapping Predictor tokens to Editor tokens.
    max_tokens: int
        Maximum number of tokens a masked input should have.
    """

    def __init__(self, mask_frac, editor_tok_wrapper, max_tokens):
        self.mask_frac = mask_frac
        self.editor_tok_wrapper = editor_tok_wrapper
        self.max_tokens = max_tokens

    def _get_mask_indices(self, **kwargs):
        """ Helper function to get indices of Editor tokens to mask. """
        raise NotImplementedError("Need to implement this in subclass")

    def get_all_masked_strings(self, editable_seq):
        """
        Returns a list of masked inputs/targets where each input has one word replaced by a sentinel token.
        Used for calculating fluency.    

        Args:
            editable_seq (string): editable string sequence

        Returns:
            list: list of masked inputs/targets where each input has one word replaced by a sentinel token.
        """
        editor_tokenized_input = self.editor_tok_wrapper(editable_seq)
        inputs_targets = list()
        for idx, _ in enumerate(editor_tokenized_input.tokens()[:-1]):
            token_span = editor_tokenized_input.token_to_chars(idx)
            token_start, token_end = token_span.start, token_span.end
            masked_seq = editable_seq[:token_start] + self._get_sentinel_token(0) + editable_seq[token_end:]
            label = self._get_sentinel_token(0) + editable_seq[token_start:token_end] + self._get_sentinel_token(1)
            inputs_targets.append((masked_seq, label))

        masked_seqs, labels = zip(*inputs_targets)

        return masked_seqs, labels

    def _get_sentinel_token(self, idx):
        """
        Helper method to get sentinel token based on given idx

        Args:
            idx (integer): given token index

        Returns:
            string : sentinel token based on given idx
        """
        # Fix due to MT5 tokenizer malfunction of sentinel tokens
        # for some reason on current version needs a space to be recognized
        # its seems to be a current config issue on model page.
        # Also we are assuming you are not loading the tokenizer from a path
        #if 'umt5' not in self.editor_tok_wrapper.name_or_path and 'mt5-' in self.editor_tok_wrapper.name_or_path:
        #    return " <extra_id_" + str(idx) + ">"
        return "<extra_id_" + str(idx) + ">"

    def _get_mask_token(self):
        """
        Helper method to get mask token based

        Returns:
            string : mask token
        """
        return "[MASK]"

    def _get_grouped_mask_indices(self, editable_seq, editor_mask_indices, **kwargs):
        """
        Groups consecutive mask indices.
        Applies heuristics to enable better generation:
            - If > 27 spans, mask tokens b/w neighboring spans as well.
                (See Appendix: observed degeneration after 27th sentinel token)
            - Mask max of 100 spans (since there are 100 sentinel tokens in T5)
            possibly generate automatic adjustment according to model

        Args:
            editable_seq (string): editable string sequence
            editor_mask_indices (list): list of token indices that can be masked

        Returns:
            list: list of token indices to mask
        """
        if editor_mask_indices is None:
            editor_mask_indices = self._get_mask_indices(editable_seq=editable_seq, **kwargs)
        
        # Removes [CLS] token index
        if "bert" in self.editor_tok_wrapper.name_or_path and 0 in editor_mask_indices:
            editor_mask_indices.remove(0)

        new_editor_mask_indices = set(editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 2 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)

        grouped_editor_mask_indices = [list(group) for group in \
                mit.consecutive_groups(sorted(new_editor_mask_indices))]

        if len(grouped_editor_mask_indices) > 27:
            for t_idx in editor_mask_indices:
                if t_idx + 3 in editor_mask_indices:
                    new_editor_mask_indices.add(t_idx + 1)
                    new_editor_mask_indices.add(t_idx + 2)

        new_editor_mask_indices = list(new_editor_mask_indices)
        grouped_editor_mask_indices = [list(group) for group in mit.consecutive_groups(sorted(new_editor_mask_indices))]
        # Mask max of 100 spans
        grouped_editor_mask_indices = grouped_editor_mask_indices[:99]
        return grouped_editor_mask_indices

    def get_masked_string(self, editable_seq,
                          editor_mask_indices=None, **kwargs):
        """
        Gets masked string masking tokens w highest predictor gradients.
        Requires mapping predictor tokens to Editor tokens because edits are made on Editor tokens.

        Args:
            editable_seq (string): editable string sequence
            editor_mask_indices (_type_, optional): _description_. Defaults to None.

        Raises:
            MaskError: A masking error has ocurred

        Returns:
            grpd_editor_mask_indices: grouped editor mask indices
            editor_mask_indices: editor mask indices
            masked_seg: masked editable_seq
            label: label string
        """
        editor_tokenized = self.editor_tok_wrapper(editable_seq,
                                                   truncation=True,
                                                   max_length=self.max_tokens)
        grpd_editor_mask_indices = self._get_grouped_mask_indices(editable_seq, editor_mask_indices,
                                                                  editor_tokenized=editor_tokenized,
                                                                  editor_tokens=editor_tokenized.tokens(),
                                                                  **kwargs)
        
        span_idx = len(grpd_editor_mask_indices) - 1
        label = self._get_sentinel_token(len(grpd_editor_mask_indices))
        masked_seg = editable_seq
        # Iterate over spans in reverse order and mask tokens
        for span in grpd_editor_mask_indices[::-1]:
            # we mask whatever it is in span index
            span_char = editor_tokenized.token_to_chars(span[0])
            end_span_char = editor_tokenized.token_to_chars(span[-1])
            span_char_start = span_char.start
            span_char_end = end_span_char.end
            end_token_idx = span[-1]

            # If last span tok is last t5 tok, heuristically set char end idx
            if span_char_end is None and end_token_idx == len(editor_tokenized.input_ids) - 1:
                span_char_end = span_char_start + 1

            if span_char_end <= span_char_start:
                logger.info("Esta pasando algo raro!!")
                raise MaskError
            if "t5" in self.editor_tok_wrapper.name_or_path:
                label = self._get_sentinel_token(span_idx) + masked_seg[span_char_start:span_char_end] + label
                masked_seg = masked_seg[:span_char_start] + self._get_sentinel_token(span_idx) + masked_seg[span_char_end:]
            elif "bert" in self.editor_tok_wrapper.name_or_path:
                masked_seg = self.mask_bert_string(span, masked_seg, editor_tokenized)
            span_idx -= 1
        if "bert" in self.editor_tok_wrapper.name_or_path:
            label = editable_seq
        return grpd_editor_mask_indices, editor_mask_indices, masked_seg, label

    def mask_bert_string(self, span, editable_seg, editor_tokenized):
        masked_seg = editable_seg
        for token in span[::-1]:
            span_char = editor_tokenized.token_to_chars(token)
            masked_seg = masked_seg[:span_char.start] + self._get_mask_token() + masked_seg[span_char.end:]
        return masked_seg