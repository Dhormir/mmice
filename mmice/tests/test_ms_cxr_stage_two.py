"""
Tests for running Stage 2 on MS-CXR with the MIMIC-CXR-trained models.

Covers the loader, task dispatch, the new CLI flags, the output-dir split,
row alignment after filtering, the extra metadata columns and resume.
No network access: load_dataset is monkeypatched with a tiny in-memory dataset.

Run:
    uv run pytest mmice/tests/test_ms_cxr_stage_two.py -v
"""

import csv
import json
import os
import sys

import numpy as np
import pytest
import torch
from datasets import Dataset, Features, Image, Sequence, Value
from munch import Munch
from PIL import Image as PILImage

import mmice.task_loader as task_loader
import mmice.utils as utils
from mmice.stage_two import (
    edit_indices,
    filter_editable_rows,
    get_extra_values,
    get_stage_two_dir,
)

# ─── Fixtures ───

TEXTS = [
    "small left pleural effusion",
    "",  # dropped by the filter
    "cardiomegaly",
    "12345",  # dropped: no letters
    "right lower lobe opacity",
]


def _fake_ms_cxr():
    """Tiny stand-in with the same columns as BoSsa-Projects/MS-CXR."""
    n = len(TEXTS)
    features = Features(
        {
            "image": Image(),
            "text": Value("string"),
            "objects": {
                "bbox": Sequence(Sequence(Value("float32"))),
                "category_name": Sequence(Value("string")),
            },
            "dicom_id": Value("string"),
            "subject_id": Value("int64"),
            "study_id": Value("int64"),
            "split": Value("string"),
        }
    )
    return Dataset.from_dict(
        {
            # Non-square grayscale images, like real CXRs
            "image": [PILImage.new("L", (300, 260), color=i * 40) for i in range(n)],
            "text": TEXTS,
            "objects": [
                {"bbox": [[float(i), 2.0, 30.0, 40.0]], "category_name": [f"cat{i}"]}
                for i in range(n)
            ],
            "dicom_id": [f"dicom_{i}" for i in range(n)],
            "subject_id": list(range(n)),
            "study_id": list(range(n)),
            "split": ["test"] * n,
        },
        features=features,
    )


@pytest.fixture
def patched_load_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, split=None, cache_dir=None, **kw):
        calls.append({"name": name, "split": split, "cache_dir": cache_dir})
        return _fake_ms_cxr()

    monkeypatch.setattr(task_loader, "load_dataset", fake_load_dataset)
    return calls


@pytest.fixture
def ms_cxr(patched_load_dataset):
    return task_loader.load_ms_cxr(split="test", cache_dir="/tmp/cache")


# ─── Loader ───


class TestLoadMsCxr:
    def test_always_loads_train_split(self, patched_load_dataset):
        task_loader.load_ms_cxr(split="test", cache_dir="/tmp/cache")
        assert patched_load_dataset[0]["name"] == "BoSsa-Projects/MS-CXR"
        assert patched_load_dataset[0]["split"] == "train"
        assert patched_load_dataset[0]["cache_dir"] == "/tmp/cache"

    def test_image_is_biomedclip_tensor(self, ms_cxr):
        img = ms_cxr[0]["image"]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)  # grayscale -> 3 channels, cropped

    def test_metadata_untouched(self, ms_cxr):
        row = ms_cxr[0]
        assert row["objects"]["bbox"] == [[0.0, 2.0, 30.0, 40.0]]
        assert row["dicom_id"] == "dicom_0"

    def test_text_column_access_with_transform(self, ms_cxr):
        # stage_two reads dr["text"] while the image transform is active
        assert ms_cxr["text"] == TEXTS

    def test_transform_survives_shuffle_and_select(self, ms_cxr):
        # stage_two does dr.shuffle(seed=42).select(range(n_samples))
        sub = ms_cxr.shuffle(seed=42).select(range(2))
        assert isinstance(sub[0]["image"], torch.Tensor)

    def test_no_transform_keeps_pil(self, patched_load_dataset):
        ds = task_loader.load_ms_cxr(transform=None)
        assert isinstance(ds[0]["image"], PILImage.Image)


# ─── Task dispatch ───


class TestDatasetReaderDispatch:
    def test_ms_cxr_dispatch_uses_data_dir(self, monkeypatch):
        seen = {}

        def fake_loader(split, cache_dir):
            seen.update(split=split, cache_dir=cache_dir)
            return "sentinel"

        monkeypatch.setattr(utils, "load_ms_cxr", fake_loader)
        out = utils.get_dataset_reader(
            "BoSsa-MS-CXR", split="test", data_dir="/drive/data"
        )
        assert out == "sentinel"
        assert seen == {"split": "test", "cache_dir": "/drive/data"}

    def test_unknown_task_still_raises(self):
        with pytest.raises(NotImplementedError):
            utils.get_dataset_reader("not-a-task")


# ─── CLI args ───

BASE_ARGV = [
    "-task",
    "BoSsa-MIMIC-CXR-1024",
    "-editor_path",
    "/editor",
    "-stage2_exp",
    "exp",
]


class TestStageTwoArgs:
    def _parse(self, monkeypatch, extra):
        monkeypatch.setattr(sys, "argv", ["prog"] + BASE_ARGV + extra)
        return utils.get_args("stage2")

    def test_defaults(self, monkeypatch):
        args = self._parse(monkeypatch, [])
        assert args.meta.eval_task is None
        assert args.misc.extra_columns == []
        assert args.misc.n_samples == 0  # 0 means "use all samples"

    def test_eval_task_and_extra_columns(self, monkeypatch):
        args = self._parse(
            monkeypatch,
            [
                "-eval_task",
                "BoSsa-MS-CXR",
                "-extra_columns",
                "objects",
                "bbox",
                "dicom_id",
            ],
        )
        assert args.meta.task == "BoSsa-MIMIC-CXR-1024"
        assert args.meta.eval_task == "BoSsa-MS-CXR"
        assert args.misc.extra_columns == ["objects", "bbox", "dicom_id"]

    def test_extra_columns_does_not_swallow_following_flags(self, monkeypatch):
        args = self._parse(
            monkeypatch, ["-extra_columns", "objects", "bbox", "-n_samples", "100"]
        )
        assert args.misc.extra_columns == ["objects", "bbox"]
        assert args.misc.n_samples == 100

    def test_invalid_eval_task_rejected(self, monkeypatch):
        with pytest.raises(SystemExit):
            self._parse(monkeypatch, ["-eval_task", "imdb"])

    def test_stage_one_has_no_eval_task(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "-task",
                "BoSsa-MIMIC-CXR-1024",
                "-stage1_exp",
                "s1",
                "-eval_task",
                "BoSsa-MS-CXR",
            ],
        )
        with pytest.raises(AssertionError, match="Unrecognized"):
            utils.get_args("stage1")


# ─── Output directory ───


def _args(task, eval_task, results_dir="/r", exp="exp"):
    return Munch(
        meta=Munch(
            task=task, eval_task=eval_task, results_dir=results_dir, stage2_exp=exp
        )
    )


class TestStageTwoDir:
    def test_same_task_keeps_old_layout(self):
        d = get_stage_two_dir(_args("BoSsa-MIMIC-CXR-1024", None))
        assert d == os.path.join("/r", "BoSsa-MIMIC-CXR-1024", "edits", "exp")

    def test_eval_task_equal_to_task_keeps_old_layout(self):
        d = get_stage_two_dir(_args("BoSsa-MIMIC-CXR-1024", "BoSsa-MIMIC-CXR-1024"))
        assert d == os.path.join("/r", "BoSsa-MIMIC-CXR-1024", "edits", "exp")

    def test_cross_task_gets_own_subdir(self):
        d = get_stage_two_dir(_args("BoSsa-MIMIC-CXR-1024", "BoSsa-MS-CXR"))
        assert d == os.path.join(
            "/r", "BoSsa-MIMIC-CXR-1024", "edits", "BoSsa-MS-CXR", "exp"
        )

    def test_same_exp_name_never_collides(self):
        a = get_stage_two_dir(_args("BoSsa-MIMIC-CXR-1024", None))
        b = get_stage_two_dir(_args("BoSsa-MIMIC-CXR-1024", "BoSsa-MS-CXR"))
        assert a != b


# ─── Row alignment ───


class TestFilterEditableRows:
    def test_drops_empty_and_letterless(self, ms_cxr):
        out = filter_editable_rows(ms_cxr, "BoSsa-MS-CXR")
        assert out["text"] == [TEXTS[0], TEXTS[2], TEXTS[4]]

    def test_text_image_metadata_stay_aligned(self, ms_cxr):
        """The old list-filter bug: dr[i] pointed at a different row than inputs[i]."""
        out = filter_editable_rows(ms_cxr, "BoSsa-MS-CXR")
        inputs = out["text"]
        for i, text in enumerate(inputs):
            orig_idx = TEXTS.index(text)
            row = out[i]
            assert row["dicom_id"] == f"dicom_{orig_idx}"
            assert row["objects"]["bbox"][0][0] == float(orig_idx)

    def test_transform_kept_after_filter(self, ms_cxr):
        out = filter_editable_rows(ms_cxr, "BoSsa-MS-CXR")
        assert isinstance(out[0]["image"], torch.Tensor)

    def test_race_untouched(self):
        ds = Dataset.from_dict({"text": ["", "a"]})
        assert filter_editable_rows(ds, "race") is ds

    def test_all_kept_is_noop(self):
        ds = Dataset.from_dict({"text": ["a", "b"]})
        assert filter_editable_rows(ds, "BoSsa-MS-CXR")["text"] == ["a", "b"]


# ─── Extra columns ───


class TestGetExtraValues:
    def test_top_level_and_nested_bbox(self, ms_cxr):
        row = ms_cxr[2]
        objects, bbox, dicom = get_extra_values(row, ["objects", "bbox", "dicom_id"])
        assert json.loads(objects)["category_name"] == ["cat2"]
        assert json.loads(bbox) == [[2.0, 2.0, 30.0, 40.0]]
        assert json.loads(dicom) == "dicom_2"

    def test_top_level_wins_over_objects(self):
        row = {"bbox": [1], "objects": {"bbox": [2]}}
        assert get_extra_values(row, ["bbox"]) == ["[1]"]

    def test_missing_column_is_null(self):
        assert get_extra_values({"text": "x"}, ["bbox"]) == ["null"]

    def test_no_columns(self):
        assert get_extra_values({"text": "x"}, []) == []

    def test_values_are_tsv_safe(self):
        row = {"objects": {"note": "line1\nline2\tx"}}
        (val,) = get_extra_values(row, ["objects"])
        assert "\n" not in val and "\t" not in val
        assert json.loads(val)["note"] == "line1\nline2\tx"

    def test_non_json_types_do_not_crash(self):
        row = {"x": np.float32(1.5), "t": torch.tensor([1, 2])}
        vals = get_extra_values(row, ["x", "t"])
        assert len(vals) == 2


# ─── Resume with the wider schema ───


BASE_FIELDS = [
    "data_idx",
    "sorted_idx",
    "orig_pred",
    "new_pred",
    "contrast_pred",
    "orig_contrast_prob_pred",
    "new_contrast_prob_pred",
    "orig_input",
    "edited_input",
    "orig_editable_seg",
    "edited_editable_seg",
    "minimality",
    "num_edit_rounds",
    "mask_frac",
    "duration",
    "error",
]


def _success_row(idx, extra):
    return [
        idx,
        0,
        "Edema",
        "NO Edema",
        "NO Edema",
        0.2,
        0.7,
        "orig",
        "edit",
        "orig",
        "edit",
        0.1,
        1,
        0.3,
        1.0,
        False,
    ] + extra


def _stage_two_writer(f):
    # Same settings as run_edit_test
    return csv.writer(f, delimiter="\t", lineterminator="\n")


class TestResumeWithExtraColumns:
    def test_stage_two_writer_uses_unix_newlines(self):
        """csv.writer defaults to CRLF but edit_indices splits on LF, so the
        last column (now an extra column like bbox) would end in a stray CR."""
        import inspect
        import mmice.stage_two as st

        src = inspect.getsource(st.run_edit_test)
        assert 'lineterminator="\\n"' in src
        # Without newline="", Windows text mode turns every LF back into CRLF
        assert 'newline=""' in src

    def test_edit_indices_skips_done_rows(self, tmp_path, ms_cxr):
        out_file = tmp_path / "edits.csv"
        extra_cols = ["objects", "bbox"]
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            w = _stage_two_writer(f)
            w.writerow(BASE_FIELDS + extra_cols)
            for idx in (0, 2):
                w.writerow(_success_row(idx, get_extra_values(ms_cxr[idx], extra_cols)))

        remaining = edit_indices(str(out_file), ["a"] * 5)
        assert sorted(remaining.tolist()) == [1, 3, 4]

    def test_file_parses_with_expected_columns(self, tmp_path, ms_cxr):
        import pandas as pd

        out_file = tmp_path / "edits.csv"
        extra_cols = ["objects", "bbox"]
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            w = _stage_two_writer(f)
            w.writerow(BASE_FIELDS + extra_cols)
            w.writerow(_success_row(0, get_extra_values(ms_cxr[0], extra_cols)))

        df = pd.read_csv(out_file, sep="\t", lineterminator="\n")
        assert list(df.columns) == BASE_FIELDS + extra_cols
        assert json.loads(df.loc[0, "bbox"]) == [[0.0, 2.0, 30.0, 40.0]]
