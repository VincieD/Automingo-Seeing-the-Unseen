from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image as PILImage

from datasets import load_dataset, Dataset, DatasetDict, Image as HFImage


@dataclass
class DatasetWrapper:
    """
    Loads a HF dataset and produces train/val datasets formatted for evaluation.

    Output columns after preprocessing:
      - images: List[PIL.Image] length == 5 (optionally padded)
      - answer
      - question
      - answers_reasoning
      - multi_choice_answers
    """
    dataset_name: str = "ibarcelo/Automingo_75"
    image_cols: Tuple[str, ...] = tuple(f"image_{i}" for i in range(1, 6))
    pad_missing_images: bool = True
    default_pad_size: Tuple[int, int] = (224, 224)

    # Speed/behavior controls (defaults preserve your current behavior)
    only_split: Optional[str] = None          # e.g. "validation" to avoid loading train at all
    map_num_proc: int = 1
    map_batch_size: int = 1000
    map_writer_batch_size: int = 1000

    def __post_init__(self) -> None:
        self.train_dataset, self.val_dataset = self._load_splits()

        self.train_dataset = self._ensure_images_decode_to_pil(self.train_dataset)
        if self.val_dataset is not None:
            self.val_dataset = self._ensure_images_decode_to_pil(self.val_dataset)

        self.train_dataset = self._preprocess(self.train_dataset)
        if self.val_dataset is not None:
            self.val_dataset = self._preprocess(self.val_dataset)

    def _load_splits(self) -> Tuple[Dataset, Optional[Dataset]]:
        # If you only want one split (typical for evaluation), load ONLY that split.
        # This avoids downloading/processing train entirely.
        if self.only_split:
            val = load_dataset(self.dataset_name, split=self.only_split)
            train = val.select([])  # empty Dataset, keeps get_datasets() API intact
            return train, val

        ds = load_dataset(self.dataset_name)

        if isinstance(ds, DatasetDict):
            train = ds["train"] if "train" in ds else next(iter(ds.values()))

            val = None
            for key in ("validation", "val", "dev", "test"):
                if key in ds:
                    val = ds[key]
                    break
            return train, val

        return ds, None

    def _ensure_images_decode_to_pil(self, ds: Dataset) -> Dataset:
        for col in self.image_cols:
            if col in ds.column_names:
                ds = ds.cast_column(col, HFImage())
        return ds

    def _pad_to_five(self, images: List[Any]) -> List[Any]:
        if not self.pad_missing_images:
            return images

        if len(images) >= 5:
            return images[:5]

        size = None
        for im in images:
            if hasattr(im, "size"):
                size = im.size
                break
        if size is None:
            size = self.default_pad_size

        pad_img = PILImage.new("RGB", size, color=(0, 0, 0))
        return images + [pad_img] * (5 - len(images))

    def _preprocess(self, ds: Dataset) -> Dataset:
        image_cols = list(self.image_cols)

        def to_five_image_sequence(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            new_images = []
            new_questions = []
            new_answers = []
            new_reasoning_texts = []
            new_multi_choice_answers = []

            batch_size = len(batch["question"])

            answers_col = batch["ground_truth_answer"] if "ground_truth_answer" in batch else [""] * batch_size
            reasons_col = batch["ground_truth_reasoning"] if "ground_truth_reasoning" in batch else [""] * batch_size
            d1 = batch["distractor_1"] if "distractor_1" in batch else [""] * batch_size
            d2 = batch["distractor_2"] if "distractor_2" in batch else [""] * batch_size
            d3 = batch["distractor_3"] if "distractor_3" in batch else [""] * batch_size

            for i in range(batch_size):
                question = (batch["question"][i] or "").strip()

                answer = (answers_col[i] or "").strip()
                reasoning = (reasons_col[i] or "").strip()
                multi_choice_answer = [(d1[i] or "").strip(), (d2[i] or "").strip(), (d3[i] or "").strip()]

                if reasoning:
                    completion_text = f"{answer}, {reasoning}" if answer else f"Reasoning:\n{reasoning}"
                else:
                    completion_text = answer

                imgs = []
                for col in image_cols:
                    if col not in batch:
                        continue
                    img = batch[col][i]
                    if img is None or img == "":
                        continue
                    imgs.append(img)

                imgs = self._pad_to_five(imgs)

                new_images.append(imgs)
                new_questions.append(question)
                new_answers.append(answer)
                new_reasoning_texts.append(completion_text)
                new_multi_choice_answers.append(multi_choice_answer)

            return {
                "images": new_images,
                "answer": new_answers,
                "question": new_questions,
                "answers_reasoning": new_reasoning_texts,
                "multi_choice_answers": new_multi_choice_answers,
            }

        return ds.map(
            to_five_image_sequence,
            batched=True,
            batch_size=self.map_batch_size,
            writer_batch_size=self.map_writer_batch_size,
            remove_columns=ds.column_names,
        )

    def get_datasets(self) -> Tuple[Dataset, Optional[Dataset]]:
        return self.train_dataset, self.val_dataset