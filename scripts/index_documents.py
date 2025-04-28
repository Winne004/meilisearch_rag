# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas",
#      "fsspec",
#     "huggingface_hub",
#     "requests",
# ]
# ///

import pandas as pd
import requests


def index_documents(articles: list[dict[str, str]]) -> None:
    r = requests.post(
        "http://127.0.0.1:8000/index/document",
        json=articles,
        timeout=30,
    )

    r.raise_for_status()
    print(f"Uploaded: {len(articles)}")


splits = {"train": "train.jsonl", "test": "test.jsonl"}
df = pd.read_json("hf://datasets/SetFit/bbc-news/" + splits["train"], lines=True)  # type: ignore

print(df.head(10))
df = df.reset_index()

articles = []
for index, row in df.iterrows():  # type: ignore
    articles.append(  # type: ignore
        {
            "id": str(index),
            "body": row["text"],
            "url": row["url"] if "url" in row else None,
        },
    )
index_documents(articles)  # type: ignore
