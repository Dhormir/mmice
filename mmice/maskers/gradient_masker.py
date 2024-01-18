import math
import numpy as np
from typing import List, Dict, Any

from torch import cuda, clamp_min
from torch import Tensor
from torch import backends
from torch.utils.hooks import RemovableHandle

from .util import batched_span_select
from .masker import Masker
from .masker import MaskError
from ..utils import get_token_offsets_from_text_field_inputs


class GradientMasker(Masker):
    """ Masks spans based on gradients of Predictor wrt. given predicted label.

    mask_frac: float
        Fraction of input tokens to mask.
    Changed to transformers.tokenizers.Tokenizer object
    editor_to_wrapper: allennlp.data.tokenizers.tokenizer
        Wraps around Editor tokenizer.
        Has capabilities for mapping Predictor tokens to Editor tokens.
    max_tokens: int
        Maximum number of tokens a masked input should have.
    grad_type: str, one of ["integrated_l1", "integrated_signed",
            "normal_l1", "normal_signed", "normal_l2", "integrated_l2"]
        Specifies how gradient value should be calculated
            Integrated vs. normal:
                Integrated: https://arxiv.org/pdf/1703.01365.pdf
                Normal: 'Vanilla' gradient
            Signed vs. l1 vs. l2:
                Signed: Sum gradients over embedding dimension.
                l1: Take l1 norm over embedding dimension.
                l2: Take l2 norm over embedding dimension.
    predictor: is a transformers pipeline
    sign_direction: One of [-1, 1, None]
        When grad_type is signed, determines whether we want to get most
        negative or positive gradient values.
        This should depend on what label is being used
        (pred_idx argument to get_masked_string).
        For example, Stage One, we want to mask tokens that push *towards*
        gold label, whereas during Stage Two, we want to mask tokens that
        push *away* from the target label.
        Sign direction plays no role if only gradient *magnitudes* are used
        (i.e. if grad_type is not signed, but involves taking the l1/l2 norm.)
    num_integrated_grad_steps: int
        Hyperparameter for integrated gradients.
        Only used when grad_type is one of integrated types.
    """

    def __init__(self, mask_frac, editor_tok_wrapper, predictor, max_tokens,
                 grad_type="normal_l2", sign_direction=None,
                 num_integrated_grad_steps=10):
        super().__init__(mask_frac, editor_tok_wrapper, max_tokens)

        self.predictor = predictor
        self.grad_type = grad_type
        self.num_integrated_grad_steps = num_integrated_grad_steps
        self.sign_direction = sign_direction
        self._token_offsets: List[Tensor] = []

        if ("signed" in self.grad_type and sign_direction is None):
            error_msg = "To calculate a signed gradient value, need to specify sign direction but got None for sign_direction"
            raise ValueError(error_msg)

        if sign_direction not in [1, -1, None]:
            error_msg = f"Invalid value for sign_direction: {sign_direction}"
            raise ValueError(error_msg)

        temp_tokenizer = self.predictor.tokenizer
        # Used later to avoid skipping special tokens like <s>
        # Why did they go to the trouble of generating a wrapper that
        # reverse engineered the hugginface tokenizer when you could just use
        # the original class?
        self.predictor_special_toks = temp_tokenizer.all_special_tokens

    def _register_embedding_gradient_hooks(self, model, embedding_gradients):
        """
        Registers a backward hook on the embedding layer of the model.  Used to save the gradients
        of the embeddings for use in get_gradients()

        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.

        We additionally add a hook on the _forward_ pass of the model's `TextFieldEmbedder` to save
        token offsets, if there are any.  Having token offsets means that you're using a mismatched
        token indexer, so we need to aggregate the gradients across wordpieces in a token.  We do
        that with a simple sum.
        """

        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            if self._token_offsets:
                # If you have a mismatched indexer with multiple TextFields, it's quite possible
                # that the order we deal with the gradients is wrong.  We'll just take items from
                # the list one at a time, and try to aggregate the gradients.  If we got the order
                # wrong, we should crash, so you'll know about it.  If you get an error because of
                # that, open an issue on github, and we'll see what we can do.  The intersection of
                # multiple TextFields and mismatched indexers is pretty small (currently empty, that
                # I know of), so we'll ignore this corner case until it's needed.
                offsets = self._token_offsets.pop(0)
                span_grads, span_mask = batched_span_select(grads.contiguous(), offsets)
                span_mask = span_mask.unsqueeze(-1)
                span_grads *= span_mask  # zero out paddings

                span_grads_sum = span_grads.sum(2)
                span_grads_len = span_mask.sum(2)
                # Shape: (batch_size, num_orig_tokens, embedding_size)
                grads = span_grads_sum / clamp_min(span_grads_len, 1)

                # All the places where the span length is zero, write in zeros.
                grads[(span_grads_len == 0).expand(grads.shape)] = 0

            embedding_gradients.append(grads)

        def get_token_offsets(module, inputs, outputs):
            offsets = get_token_offsets_from_text_field_inputs(inputs)
            if offsets is not None:
                self._token_offsets.append(offsets)

        hooks = []
        text_field_embedder = model.base_model.embeddings
        hooks.append(text_field_embedder.register_forward_hook(get_token_offsets))
        embedding_layer = model.base_model.embeddings
        hooks.append(embedding_layer.register_full_backward_hook(hook_layers))
        return hooks

    def _get_gradients_by_prob(self, instance, pred_idx):
        """ Helper function to get gradient values of predicted logit
        Largely copied from Predictor class of AllenNLP """
        instances = instance
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.predictor.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            # what is this??
            # why check and then change anyways?
            # This has to be some kind of error
            param.requires_grad = True

        embedding_gradients: list[Tensor] = []
        hooks: list[RemovableHandle] = self._register_embedding_gradient_hooks(self.predictor.model, embedding_gradients)

        tokenized_instances = self.predictor.tokenizer(instances['sentence'],
                                                       truncation=True,
                                                       max_length=self.predictor.tokenizer.model_max_length,
                                                       return_tensors="pt").to(self.predictor.device)
        with backends.cudnn.flags(enabled=True):
            outputs = self.predictor.model(**tokenized_instances)
            # Differs here
            prob = outputs["logits"][0][pred_idx]
            self.predictor.model.zero_grad()
            prob.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad.detach().cpu().numpy()

        # Restore original requires_grad values of the parameters
        for param_name, param in self.predictor.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]

        cuda.empty_cache()
        return grad_dict, outputs

    def _get_word_positions(self, predic_tok_span, editor_tokenized):
        """ Helper function to map from (sub)tokens of Predictor to
        token indices of Editor tokenizer. Assumes the tokens are in order.
        Raises MaskError if tokens cannot be mapped
            This sometimes happens due to inconsistencies in way text is
            tokenized by different tokenizers. """
        return_word_idx = None
        # We determine predictor token position in the editable_sequence
        predic_tok_start = predic_tok_span.start
        predic_tok_end = predic_tok_span.end
        editor_tokens = editor_tokenized.tokens()
        
        if predic_tok_start is None or predic_tok_end is None:
           return [], [], []

        # Why use Try except when a simple *for* *break* should suffice
        class Found(Exception): pass
        try:
            for word_idx, word_token in reversed(list(enumerate(editor_tokens))):
                # this is not optimized probably change it so it doesnt need the editable sequence and tokenize in a 
                # previous step instead of perfomer it every time
                word_token_span = editor_tokenized.token_to_chars(word_idx)
                if word_token_span is None:
                    continue
                # Ensure predic_tok start >= start of last Editor tok
                if word_idx == len(editor_tokens) - 1 and predic_tok_start >= word_token_span.start:
                    return_word_idx = word_idx
                    raise Found
                # For all other Editor toks, ensure predic_tok start
                # >= Editor tok start and < next Editor tok start
                elif predic_tok_start >= word_token_span.start:
                    for cand_idx, cand_token in enumerate(editor_tokens[word_idx + 1:]):
                        cand_idx += word_idx
                        cand_token_span = editor_tokenized.token_to_chars(cand_idx)
                        if editor_tokens[cand_idx] is None:
                            continue
                        elif predic_tok_start < cand_token_span.start:
                            return_word_idx = word_idx
                            raise Found
        except Found:
            # Means it found the index
            pass

        if return_word_idx is None:
            return [], [], []

        last_idx = return_word_idx
        editor_token_span = editor_tokenized.token_to_chars(return_word_idx)
        if predic_tok_end > editor_token_span.end:
            for next_idx in range(return_word_idx, len(editor_tokens)):
                if editor_token_span.end is None:
                    continue
                if predic_tok_end <= editor_token_span.end:
                    last_idx = next_idx
                    break

            return_indices = []
            return_starts = []
            return_ends = []

            for cand_idx in range(return_word_idx, last_idx + 1):
                cand_token_span = editor_tokenized.token_to_chars(cand_idx)
                return_indices.append(cand_idx)
                return_starts.append(cand_token_span.start)
                return_ends.append(cand_token_span.end)

            if predic_tok_start < editor_token_span.start:
                print('in Here!I')
                raise MaskError

            # Sometimes BERT tokenizers add extra tokens if spaces at end
            last_editor_token_span = editor_tokenized.token_to_chars(last_idx)
            if last_idx == len(editor_tokens) - 1 and predic_tok_end > last_editor_token_span.end:
                print(f'predic_tok_end: {predic_tok_end}')
                print(f'last_editor_token_span.end: {last_editor_token_span.end}')
                print('in Here!II')
                raise MaskError

            return return_indices, return_starts, return_ends

        return_editor_token_span = editor_tokenized.token_to_chars(return_word_idx)
        return_tuple = ([return_word_idx], [return_editor_token_span.start], [return_editor_token_span.end])
        return return_tuple

    # Copied from AllenNLP integrated gradient
    def _integrated_register_forward_hook(self, alpha, embeddings_list):
        """ Helper function for integrated gradients """

        def forward_hook(module, inputs, output):
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach().cpu().numpy())

            output.mul_(alpha)

        embedding_layer = self.predictor.model.base_model.embeddings
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    # Copied from AllenNLP integrated gradient
    def _get_integrated_gradients(self, instance, pred_idx, steps):
        """ Helper function for integrated gradients """

        ig_grads: Dict[str, Any] = {}

        # List of Embedding inputs
        embeddings_list: List[np.ndarray] = []

        # Exclude the endpoint because we do a left point integral approx
        for alpha in np.linspace(0, 1.0, num=steps, endpoint=False):
            # Hook for modifying embedding value
            handle = self._integrated_register_forward_hook(
                    alpha, embeddings_list)

            grads = self._get_gradients_by_prob(instance, pred_idx)[0]
            handle.remove()

            # Running sum of gradients
            if ig_grads == {}:
                ig_grads = grads
            else:
                for key in grads.keys():
                    ig_grads[key] += grads[key]

        # Average of each gradient term
        for key in ig_grads.keys():
            ig_grads[key] /= steps

        # Gradients come back in reverse order of order sent into the network
        embeddings_list.reverse()

        # Element-wise multiply average gradient by the input
        for idx, input_embedding in enumerate(embeddings_list):
            key = "grad_input_" + str(idx + 1)
            ig_grads[key] *= input_embedding

        return ig_grads

    def _get_gradient_magnitudes(self, labeled_instance, pred_idx, integrated_grad_steps):
        """
        Method to calculated gradient magnitude in predictor according to gradient type

        Args:
            labeled_instance (string): labeled instance text
            pred_idx (integer): prediction index in labeled instance
            integrated_grad_steps (integer): amount of step to take while calculating gradient

        Returns:
            grad_signed (list): vector indicating whether magnitude sign.
            grad_magnitudes (list): vector of gradient magnitude for each token.
        """
        if self.grad_type == "integrated_l1":
            grads = self._get_integrated_gradients(labeled_instance, pred_idx, steps=integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(abs(grad), axis = 1)
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "integrated_signed":
            grads = self._get_integrated_gradients(labeled_instance, pred_idx, steps=integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(grad, axis = 1)
            grad_magnitudes = self.sign_direction * grad_signed

        elif self.grad_type == "integrated_l2":
            grads = self._get_integrated_gradients(labeled_instance, pred_idx, steps=integrated_grad_steps)
            grad = grads["grad_input_1"][0]
            grad_signed = [g.dot(g) for g in grad]
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "normal_l1":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(abs(grad), axis = 1)
            grad_magnitudes = grad_signed.copy()

        elif self.grad_type == "normal_signed":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = np.sum(grad, axis = 1)
            grad_magnitudes = self.sign_direction * grad_signed

        elif self.grad_type == "normal_l2":
            grads = self._get_gradients_by_prob(labeled_instance, pred_idx)[0]
            grad = grads["grad_input_1"][0]
            grad_signed = [g.dot(g) for g in grad]
            grad_magnitudes = grad_signed.copy()

        return grad_signed, grad_magnitudes

    def sanity_check(self, predic_tok_end_idx, predic_tok_start_idx, grad_magnitudes, all_predic_toks):
        """
        Sanity Check for len magnitudes and predictor tokens

        Args:
            predic_tok_end_idx (integer): predictor tokens end index
            predic_tok_start_idx (integer): predictor tokens start index
            grad_magnitudes (list): list of magnitudes
            all_predic_toks (list): predictor tokenized input
        """
        max_length = self.predictor.tokenizer.model_max_length

        if predic_tok_end_idx is not None:
            if predic_tok_start_idx is not None:
                assert(len(grad_magnitudes) == \
                        predic_tok_end_idx - predic_tok_start_idx)
            else:
                assert(len(grad_magnitudes) == predic_tok_end_idx)
        elif max_length is not None and (len(grad_magnitudes)) >= max_length:
            assert(max_length == (len(grad_magnitudes)))
        else:
            assert(len(all_predic_toks) == (len(grad_magnitudes)))

    def get_important_editor_tokens(self, 
                                    editable_seq,
                                    pred_idx,
                                    editor_tokenized,
                                    labeled_instance=None,
                                    predictor_tok_start_idx=None, 
                                    predictor_tok_end_idx=None, 
                                    num_return_toks=None):
        """ Gets Editor tokens that correspond to Predictor toks
        with highest gradient values (with respect to pred_idx).

        editable_seq:
            Original inp to mask.
        pred_idx:
            Index of label (in Predictor label space) to take gradient of.
        editor_tokenized:
            Tokenized words using Editor tokenizer
        labeled_instance:
            Instance object for Predictor
        predic_tok_start_idx:
            Start index of Predictor tokens to consider masking.
            Helpful for when we only want to mask part of the input,
                as in RACE (only mask article). In this case, editable_seq
                will contain a subinp of the original input, but the
                labeled_instance used to get gradient values will correspond
                to the whole original input, and so predic_tok_start_idx
                is used to line up gradient values with tokens of editable_seq.
        predic_tok_end_idx:
            End index of Predictor tokens to consider masking.
            Similar to predic_tok_start_idx.
        num_return_toks: int
            If set to value k, return k Editor tokens that correspond to
                Predictor tokens with highest gradients.
            If not supplied, use self.mask_frac to calculate # tokens to return
        """
        integrated_grad_steps = self.num_integrated_grad_steps
        tokenized_editable_seq = self.predictor.tokenizer(editable_seq,
                                                          truncation=True,
                                                          max_length=self.predictor.tokenizer.model_max_length)
        all_predic_toks = tokenized_editable_seq.tokens()
        # TODO: Does NOT work for RACE
        # If labeled_instance is not supplied, create one
        if labeled_instance is None:
            labeled_instance = self.predictor(editable_seq)[0][0]
            labeled_instance['sentence'] = editable_seq

        grad_type_options = ["integrated_l1", "integrated_signed", "normal_l1",
                             "normal_signed", "normal_l2", "integrated_l2"]
        if self.grad_type not in grad_type_options:
            raise ValueError("Invalid value for grad_type")
        # Grad_magnitudes is used for sorting; highest values ordered first.
        # -> For signed, to only mask most neg values, multiply by -1
        grad_signed, grad_magnitudes = self._get_gradient_magnitudes(labeled_instance,
                                                                     pred_idx,
                                                                     integrated_grad_steps)

        # Include only gradient values for editable parts of the inp
        if predictor_tok_end_idx is not None:
            if predictor_tok_start_idx is not None:
                grad_magnitudes = grad_magnitudes[predictor_tok_start_idx:predictor_tok_end_idx]
                grad_signed = grad_signed[predictor_tok_start_idx:predictor_tok_end_idx]
            else:
                grad_magnitudes = grad_magnitudes[:predictor_tok_end_idx]
                grad_signed = grad_signed[:predictor_tok_end_idx]

        # Order Predictor tokens from largest to smallest gradient values
        ordered_predic_tok_indices = np.argsort(grad_magnitudes)[::-1]
        ordered_word_indices_by_grad = [self._get_word_positions(tokenized_editable_seq.token_to_chars(idx),
                                                                 editor_tokenized)[0] for idx in ordered_predic_tok_indices if all_predic_toks[idx] not in self.predictor_special_toks]
        ordered_word_indices_by_grad = [item for sublist in ordered_word_indices_by_grad for item in sublist]
        # Sanity checks
        self.sanity_check(predictor_tok_end_idx, predictor_tok_start_idx, grad_magnitudes, all_predic_toks)

        # Get num words to return
        if num_return_toks is None:
            num_return_toks = math.ceil(self.mask_frac * len(ordered_word_indices_by_grad))
        highest_editor_tok_indices = []
        for idx in ordered_word_indices_by_grad:
            if idx not in highest_editor_tok_indices:
                highest_editor_tok_indices.append(idx)
                if len(highest_editor_tok_indices) == num_return_toks:
                    break
        # why is this done?? probably an error
        # highest_predic_tok_indices = ordered_predic_tok_indices[:num_return_toks]
        return highest_editor_tok_indices

    def _get_mask_indices(self, **kwargs):
        """ Helper function to get indices of Editor tokens to mask. """
        editable_seq = kwargs.pop('editable_seq')
        pred_idx = kwargs.pop('pred_idx')
        kwargs.pop('editor_tokens')
        editor_tokenized = kwargs.pop('editor_tokenized')
        editor_mask_indices = self.get_important_editor_tokens(editable_seq, pred_idx, editor_tokenized, **kwargs)
        return editor_mask_indices
