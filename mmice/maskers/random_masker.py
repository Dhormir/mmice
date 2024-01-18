import random
import math

from .masker import Masker

class RandomMasker(Masker):
    """ Masks randomly chosen spans. """

    def __init__(self, mask_frac, editor_tok_wrapper, max_tokens):
        super().__init__(mask_frac, editor_tok_wrapper, max_tokens)

    def _get_mask_indices(self, **kwargs):
        """
        Helper function to get indices of Editor tokens to mask.
        
        Args:
            editable_seq (string): editable string sequence
            editor_tokens (list[string]): editable_seq tokenized by editor tokenizer 
            pred_idx (integer): index of 'label' token

        Returns:
            list: randomly chosen spans
        """
        editor_tokens = kwargs.pop('editor_tokens')
        # We heuristically exclude last token which is always end of sentence: </s>
        num_tokens = min(self.max_tokens, len(editor_tokens[:-1]))
        return random.sample(range(num_tokens), math.ceil(self.mask_frac * num_tokens))