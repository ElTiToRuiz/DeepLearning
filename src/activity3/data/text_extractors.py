"""
Generic helper that scores a list of tweets with any HuggingFace
sequence-classification model and produces per-tweet probability columns,
then aggregates them by day. Used by sentiment, emotion and toxicity
modules so each one stays a thin config wrapper.
"""
import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.activity3.config import DATASET_DIR, MAX_TEXT_TOKENS, SENTIMENT_BATCH, get_device


class _TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def _score_with_model(tweets_df: pd.DataFrame,
                     model_name: str,
                     output_columns: List[str],
                     batch_size: int = SENTIMENT_BATCH,
                     max_tokens: int = MAX_TEXT_TOKENS) -> pd.DataFrame:
    """Pass every row of tweets_df through `model_name` and append the
    softmax probabilities as new columns named by `output_columns`.

    output_columns must match the model's id2label order so we can keep
    them aligned. If the model has extra labels we drop them (sigmoid
    multi-head outputs handled separately if needed).
    """
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    is_multilabel = (model.config.problem_type == "multi_label_classification"
                     if hasattr(model.config, "problem_type") and model.config.problem_type
                     else False)

    ds = _TextDataset(tweets_df["text"].fillna("").astype(str), tokenizer, max_tokens)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    accum = {c: [] for c in output_columns}
    n_cols = len(output_columns)

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(loader, desc=f"Scoring {model_name}",
                                              leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask).logits[:, :n_cols]
            if is_multilabel:
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            for j, c in enumerate(output_columns):
                accum[c].extend(probs[:, j].tolist())

    out = tweets_df.copy().reset_index(drop=True)
    for c in output_columns:
        out[c] = accum[c]
    return out


def cached_score(tweets_df: pd.DataFrame, cache_filename: str,
                 model_name: str, output_columns: List[str],
                 force: bool = False, **kwargs) -> pd.DataFrame:
    """Run `_score_with_model` if not already cached, save the result to
    parquet, and return it."""
    cache_path = os.path.join(DATASET_DIR, cache_filename)
    if (not force) and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    os.makedirs(DATASET_DIR, exist_ok=True)
    scored = _score_with_model(tweets_df, model_name, output_columns, **kwargs)
    scored.to_parquet(cache_path, index=False)
    return scored


def aggregate_daily(scored_df: pd.DataFrame, feature_columns: List[str],
                    include_text: bool = True,
                    text_sep: str = " [SEP] ") -> pd.DataFrame:
    """Group by date and take the mean of each feature column. If
    include_text is True (default) also returns n_tweets and the
    joined text (used by the LogReg / TF-IDF baseline). Set False when
    you just want the score columns so several daily frames can be
    merged without colliding."""
    agg_dict = {c: (c, "mean") for c in feature_columns}
    if include_text:
        agg_dict["n_tweets"] = ("text", "size")
        agg_dict["text_concat"] = ("text", lambda xs: text_sep.join(map(str, xs)))
    return scored_df.groupby("date", as_index=False).agg(**agg_dict)
