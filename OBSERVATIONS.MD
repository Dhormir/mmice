# Repository Code Observations

On this file we signal all code errors and bad implemented methods, in order to visualize better the code imrpovements generated.
Each section its related to each file, in case we decided to split a file this will be signaled by dividing the file section into subsections.

* We must explicit truncation when working with pipelines models and tokenizer.

## stage_One.py

TBA

## masker.py

On class Masker:

* _get_grouped_mask_indices: remove unused pred_idx parameter (this is gradient masker specific), this can be better implemented by just doing overloading.
    Probably we must move this from the parent to the child.
    Fixed: We now take advantage of the **kwargs, instead of unnecesarily cluttering the parent implementations, using kwargs allows for better code readability making the use of args related to a certain kind of masker instead of forcing all implementations to share the same arguments

## dataset.py

TBA

## utils.py

TBA

## editor.py

TBA

## edit_finder.py

TBA
