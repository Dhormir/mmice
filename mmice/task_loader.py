import pandas as pd
from datasets import Dataset, Sequence, Value, Features, load_dataset
from torchvision import transforms

BIOMED_CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

# Task loader script in case someone wants to expand and add a new task
# just create a loader here that returns a Hugginface Dataset object
# with text and label columns


def load_chilean_hate(data_files=None, column_names=["text", "Odio"]):
    kwargs = {"encoding": "utf-8", "index_col": 0, "sep": ","}
    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]

    data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    data = data.rename_column("Odio", "label")
    return data.train_test_split(train_size=0.89, seed=42)


def load_42k_hcuch(
    data_files=None,
    column_names=["hallazgos", "impresion", "condensacion", "nodulos", "quistes"],
):
    kwargs = {
        "encoding": "utf-8",
        "sep": "|",
        "dtype": {"condensacion": int, "nodulos": int, "quistes": int},
    }

    def combine_columns(example):
        example["label"] = [
            example["condensacion"],
            example["nodulos"],
            example["quistes"],
        ]
        return example

    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]
    df["text"] = df["hallazgos"]  # + " " +  df['impresion']

    data = Dataset.from_pandas(df).remove_columns(
        ["hallazgos", "impresion", "__index_level_0__"]
    )
    data = data.map(combine_columns)
    new_features = data.features.copy()
    new_features["label"] = Sequence(feature=Value("int8"), length=3)

    data = data.cast(new_features)
    data = data.remove_columns(["condensacion", "nodulos", "quistes"])
    # For multilabel models we will make it focus only on the highest probability label
    return data.train_test_split(train_size=0.75, seed=42)


def load_semeval_hate(data_files=None, column_names=["text", "HS"]):
    kwargs = {"encoding": "utf-8"}
    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]

    data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    data = data.rename_column("HS", "label")
    data = data.shuffle(42)
    return data.train_test_split(train_size=0.75, seed=42)


def load_mimic_cxr(split="train", transform=BIOMED_CLIP_TRANSFORM):
    MIMIC_LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Lung Opacity",
        "Pleural Effusion",
        "Pneumonia",
        "Pneumothorax",
    ]
    print("Loading dataset")
    ds = load_dataset(
        "BoSsa-Projects/MIMIC-CXR-1024",
        split="train[:10000]" if split == "train" else split,
        cache_dir="/content/drive/MyDrive/mmice_data",
    )

    # Normalize columns to match MMiCE convention
    ds = ds.rename_column("report", "text")
    ds = ds.map(lambda row: {"label": [int(row[col]) for col in MIMIC_LABEL_COLS]})
    if transform is not None:

        def apply_transform(batch):
            if "image" in batch:
                batch["image"] = [transform(img) for img in batch["image"]]
            return batch

        ds.set_transform(apply_transform)
    return ds
