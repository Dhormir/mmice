from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import logging

# Local imports
from .maskers.gradient_masker import GradientMasker
from .maskers.random_masker import RandomMasker
from .dataset import StageOneDataset, RaceStageOneDataset
from .utils import logger, write_args, load_base_editor

FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT)
logger.setLevel(logging.INFO)

# For better pytorch performance on distributed GPU
ACCELERATOR = Accelerator()


def train_epoch(epoch, editor_tokenizer, editor_model, train_data_loader, optimizer):
    """
    Runs training for epoch.
    
    Args:
        epoch (integer): Current epoch number.
        editor_tokenizer (transformers.tokenizers.Tokenizer): Pretrained editor tokenizer.
        editor_model (transformers.PreTrainedModel): Pretrained editor model.
        train_data_loader (torch.utils.data.DataLoader): Dataloader for efficient batching.
        optimizer (torch.optim.Optimizer): Optimizer used in training.
        progress_bar (tqdm): Progress bar for visualization of progress across epochs.

    Returns:
        total_loss (integer): Loss sum of training in epoch.
    """
    editor_model.train()
    total_loss = 0
    logger.info(f"Training epoch: {epoch}")
    # Train Epoch Progress bar
    progress_bar = tqdm(train_data_loader,
                        total=len(train_data_loader),
                        disable=not ACCELERATOR.is_local_main_process,)
    progress_bar.set_description('Training loop progress')

    for i, batch in enumerate(train_data_loader, 1):
        # We might want to check this?
        lm_labels = batch['target_ids']
        lm_labels[lm_labels == editor_tokenizer.pad_token_id] = -100
        ids = batch['source_ids']
        # outputs = editor_model(input_ids=ids, labels=lm_labels, attention_mask=batch['source_mask'])
        outputs = editor_model(input_ids=ids, labels=lm_labels, )
        loss = outputs.loss
        total_loss += loss.item()
        ACCELERATOR.backward(loss)
        # Gradient acummulation?
        optimizer.step()
        optimizer.zero_grad()
        del lm_labels
        del ids
        progress_bar.update()
        progress_bar.set_postfix_str(f"Loss: {total_loss/i:.4f}")
    progress_bar.close()
    avg_loss = total_loss/len(train_data_loader)
    logger.info(f'Epoch: {epoch},' +
                f' Avg Batch Loss:  {avg_loss}')
    return avg_loss


def validate_epoch(epoch, editor_tokenizer, editor_model, val_data_loader):
    """
    Runs validation for epoch.
    
    Args:
        epoch (integer): _description_
        editor_tokenizer (transformers.tokenizers.Tokenizer): Pretrained editor tokenizer.
        editor_model (transformers.PreTrainedModel): Pretrained editor model.
        val_data_loader (torch.utils.data.DataLoader): Dataloader for efficient batching.

    Returns:
        total_loss (integer): Loss sum of validation in epoch.
    """
    editor_model.eval()
    total_loss = 0
    logger.info(f"Validating epoch: {epoch}")
    # This progress bar can be greatly improved.
    progress_bar = tqdm(val_data_loader,
                        total=len(val_data_loader),
                        desc='Validation loop progress',)
    for i, batch in enumerate(val_data_loader, 1):
        lm_labels = batch['target_ids']
        lm_labels[lm_labels == editor_tokenizer.pad_token_id] = -100
        ids = batch['source_ids']
        # outputs = editor_model(input_ids=ids, labels=lm_labels, attention_mask=batch['source_mask'])
        outputs = editor_model(input_ids=ids, labels=lm_labels, )
        loss = outputs.loss
        total_loss += loss.item()

        del lm_labels
        del ids
        progress_bar.update()
        progress_bar.set_postfix_str(f"Loss: {total_loss/i:.4f}")
    progress_bar.close()
    avg_loss = total_loss/len(val_data_loader)
    logger.info(f'Epoch: {epoch},' +
                f' Avg Validation Batch Loss:  {avg_loss}')
    return avg_loss


def get_datasets(predictor, dataset_reader, masker, data_dir, train_inputs, val_inputs,
                 train_labels, val_labels, editor_tokenizer, args):
    """
    Writes data for Editor fine-tuning.

    Args:
        predictor (_type_): Some predictor model that we want to explain.
        dataset_reader (_type_): _description_
        masker (Masker): Child of masker class. Type of mask to be used on input.
            This supports any kind of Masker object as long as it has the pertinent class methods implemented.
        data_dir (string): Data directory location.
        train_inputs (_type_): _description_
        val_inputs (_type_): _description_
        train_labels (_type_): _description_
        val_labels (_type_): _description_
        editor_tokenizer (transformers.tokenizers.Tokenizer): Pretrained editor tokenizer.
        args (_type_): _description_

    Returns:
        train_dataset (StageOneDataset): Train dataset transformed to stage one format.
        val_dataset (StageOneDataset): Validation dataset transformed to stage one format.
    """
    train_data_path = os.path.join(data_dir, "train_data.csv")
    val_data_path = os.path.join(data_dir, "val_data.csv")

    # If data already exists for experiment, read data
    if os.path.exists(train_data_path) and os.path.exists(val_data_path):
        logger.info("Data for Editor fine-tuning already exist.")
        logger.info(f"Loading train data from: {train_data_path}")
        logger.info(f"Loading val data from: {val_data_path}")

        train_csv = pd.read_csv(train_data_path, sep="\t")
        val_csv = pd.read_csv(val_data_path, sep="\t")

        train_dataset = StageOneDataset(editor_tokenizer,
                                        max_length=args.model.model_max_length,
                                        masked_strings=train_csv['inputs'],
                                        targets=train_csv['targets'],
                                        lang=args.meta.lang)
        val_dataset = StageOneDataset(editor_tokenizer,
                                      max_length=args.model.model_max_length,
                                      masked_strings=val_csv['inputs'],
                                      targets=val_csv['targets'],
                                      lang=args.meta.lang)

    # Else, create data by calling create_inputs() function in dataset.py
    else:
        logger.info("Creating masked data for Editor fine-tuning...")
        logger.info("Target label (options are 'pred' or 'gold'): " +
                    f"{args.misc.target_label}")
        # For RACE, pass dataset_reader to create_inputs() to correctly truncate
        if args.meta.task == "race":
            train_dataset = RaceStageOneDataset(editor_tokenizer, max_length=args.model.model_max_length,
                                                lang=args.meta.lang)
            train_dataset.create_inputs(dataset_reader, train_inputs, train_labels,
                                        predictor, masker, target_label=args.misc.target_label)
            val_dataset = RaceStageOneDataset(editor_tokenizer, max_length=args.model.model_max_length,
                                              lang=args.meta.lang)
            val_dataset.create_inputs(dataset_reader, val_inputs, val_labels, predictor, masker, target_label=args.misc.target_label)
        else:
            train_dataset = StageOneDataset(editor_tokenizer, max_length=args.model.model_max_length,
                                            lang=args.meta.lang)
            val_dataset = StageOneDataset(editor_tokenizer, max_length=args.model.model_max_length,
                                          lang=args.meta.lang)
            train_dataset.create_inputs(train_inputs, train_labels, predictor,
                                        masker, target_label=args.misc.target_label)
            val_dataset.create_inputs(val_inputs, val_labels, predictor,
                                      masker, target_label=args.misc.target_label)
        logger.info("Done creating data.")

        # Write data
        logger.info(f"Writing train data to: {train_data_path}")
        train_masked_df = pd.DataFrame({'inputs':train_dataset.masked_strings, 'targets':train_dataset.targets})
        train_masked_df.to_csv(train_data_path, sep="\t")
        logger.info(f"Writing val data to: {val_data_path}")
        val_masked_df = pd.DataFrame({'inputs':val_dataset.masked_strings, 'targets':val_dataset.targets})
        val_masked_df.to_csv(val_data_path, sep="\t")

    return train_dataset, val_dataset


def get_stage_one_masker(args, editor_tokenizer, predictor):
    """
    Helper function for loading appropriate masker, random or grad.

    Args:
        args (_type_): Arguments dictionary.
        predictor (_type_): Some predictor model that we want to explain.
        (Probably we can make a wrapper around pipelines, pretrained pytorch and tensorflow models, or something like that)

    Returns:
        masker : (Masker): Child of masker class. Type of mask to be used on input.
        This support any kind of Masker object as long as it has the pertinent class methods implemented.
        Can be expanded in the future to new maskers.
    """
    logger.info(f"Creating masker of type: {args.mask.mask_type}")

    if args.mask.mask_type == "random":
        logger.info("Loading Random masker...")
        masker = RandomMasker(None, editor_tokenizer, args.model.model_max_length)
    elif args.mask.mask_type == "grad":
        logger.info("Loading Gradient Masker...")
        # In stage 1, if signed gradients, mask tokens pushing *towards* target
        sign_direction = 1 if "signed" in args.mask.grad_type else None 
        masker = GradientMasker(None, editor_tokenizer, predictor,
                                args.model.model_max_length,
                                grad_type=args.mask.grad_type,
                                sign_direction=sign_direction)
    logger.info("Done.")
    return masker


def get_task_data(args, dataset_reader):
    """
    Helper function for loading original data of task. 
    Calls get_inputs() function of dataset reader dataset_reader.

    Args:
        args (_type_): Arguments dictionary.
        dataset_reader (_type_): _description_

    Returns:
        train_inputs (_type_): _description_
        val_inputs (_type_): _description_
        train_labels (_type_): _description_
        val_labels (_type_): _description_
    """
    if args.meta.task in ["imdb", "newsgroups", "chileanhate", "42k_hcuch"]:
        train_data, val_data = dataset_reader.train_test_split(train_size=args.train.data_split_ratio).values()
        train_inputs, train_labels = train_data["text"], train_data["label"]
        val_inputs, val_labels = val_data["text"], val_data["label"]
    else:
        logger.error("Unsupported dataset")
        raise Exception("Unsupported Task dataset")
    logger.info(f"Num train for Editor fine-tuning: {len(train_inputs)}")
    logger.info(f"Num val for Editor fine-tuning: {len(val_inputs)}")

    return train_inputs, val_inputs, train_labels, val_labels


def run_train_editor(predictor, dataset_reader, args):
    """
    Runs Editor training.

    Args:
        predictor (_type_): Some predictor model that we want to explain.
        dataset_reader (_type_): _description_
        args (_type_): Arguments dictionary.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    torch.backends.cudnn.deterministic = True

    editor_tokenizer, editor_model = load_base_editor(model_name=args.model.model_name,
                                                      max_length=args.model.model_max_length)
    # We use the accelerator to prepare the model
    editor_model = ACCELERATOR.prepare(editor_model)

    # DIRECTORIES
    task_dir = os.path.join(args.meta.results_dir, args.meta.task)
    stage_one_dir = os.path.join(task_dir, f"editors/{args.meta.stage1_exp}")
    data_dir = os.path.join(stage_one_dir, f'{args.mask.mask_type}/editor_train_data')
    checkpoint_dir = os.path.join(stage_one_dir, 'checkpoints')
    
    # LOGS
    logger.info(f"Task dir: {task_dir}")
    logger.info(f"Stage one dir: {stage_one_dir}")
    logger.info(f"Stage one training data dir: {data_dir}")
    logger.info(f"Checkpoints dir: {checkpoint_dir}")

    for dir in [task_dir, data_dir, stage_one_dir, checkpoint_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # setting log file    
    meta_log_file = os.path.join(stage_one_dir, "meta_log.txt")
    # add output to log file
    logger.addHandler(logging.FileHandler(meta_log_file))
        
    # Save args
    args_path = os.path.join(stage_one_dir, "stage_one_args.json")
    write_args(args_path, args)

    # We get our masker
    masker = get_stage_one_masker(args, editor_tokenizer, predictor)

    # Defining the parameters for creation of dataloaders
    train_params = {'batch_size': args.train.train_batch_size,
                    'shuffle': True,
                    'num_workers': 0,
                    }
    val_params = {'batch_size': args.train.val_batch_size,
                  'shuffle': False,
                  'num_workers': 0,
                  }

    # Load original task data
    train_inputs, val_inputs, train_labels, val_labels = get_task_data(args, dataset_reader)
    # Get datasets for Editor training
    train_dataset, val_dataset = get_datasets(predictor, dataset_reader,
                                              masker, data_dir,
                                              train_inputs, val_inputs,
                                              train_labels, val_labels,
                                              editor_tokenizer, args)
    # We free all used gpu memory on creating the inputs and free the used predictor from memory
    torch.cuda.empty_cache()
    del(predictor)
    # We use the accelerator to prepare the optimizer
    optimizer = ACCELERATOR.prepare(torch.optim.Adam(params=editor_model.parameters(), lr=args.train.lr))
    # We use the accelerator to prepare the dataloaders
    train_data_loader = ACCELERATOR.prepare(DataLoader(train_dataset, **train_params))
    val_data_loader = ACCELERATOR.prepare(DataLoader(val_dataset, **val_params))

    # Training loop
    logger.info('Initiating Editor Fine-Tuning.')
    # Training Progress bar
    progress_bar = tqdm(range(args.train.num_epochs),
                        disable=not ACCELERATOR.is_local_main_process)
    progress_bar.set_description('Epoch Training progress')
    
    # Path to best validation model
    best_path = os.path.join(checkpoint_dir, 'best.pth')
    best_val_loss = 1e6
    for epoch in progress_bar:
        path = os.path.join(checkpoint_dir, f"{epoch}.pth")
        if os.path.exists(path):
            logger.info(f"Found checkpoint for epoch. Loading from: {path}")
            ACCELERATOR.load_state(path)
        else:
            avg_train_loss = train_epoch(epoch, editor_tokenizer, editor_model,
                                         train_data_loader, optimizer)
            logger.info(f"Epoch {epoch} Avg Train Batch Loss: {avg_train_loss:.4f}. Saving weights to: {path}")
            logger.info("Saving Editor checkpoint to: " + path)

            ACCELERATOR.save_state(path, safe_serialization=False)
            val_loss = validate_epoch(epoch, editor_tokenizer, editor_model, val_data_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"Lowest loss. Saving weights to: {best_path}")
                ACCELERATOR.save_state(best_path, safe_serialization=False)
            progress_bar.set_postfix_str(f"Avg Train Loss: {avg_train_loss:.4f}, Best Validation Loss: {best_val_loss:.4f}")
    progress_bar.close()
