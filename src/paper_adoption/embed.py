"""Turn every paper's title + abstract into a vector of numbers with a sentence-transformer model.

    .venv/bin/python -m paper_adoption.embed                       # all-MiniLM-L6-v2 on papers.csv (iteration 2)
    .venv/bin/python -m paper_adoption.embed specter2              # SPECTER2 on papers.csv (iteration 4)
    .venv/bin/python -m paper_adoption.embed specter2 holdout_ holdout.csv
    .venv/bin/python -m paper_adoption.embed specter2 full_biotech_ full_papers.csv biotech   # one field of the full table (biotech track, iteration 4)

Why: a model cannot read text; it needs numbers. TF-IDF gives one number per word. A sentence-transformer gives a dense
"meaning" vector in which papers about similar things end up close together. Two models:
  * all-MiniLM-L6-v2: small (90 MB), general English, reads at most 256 word-pieces (~200 words). The iteration-2 baseline.
  * SPECTER2 (allenai/specter2_base): a scientific-paper encoder trained on citation relations between papers, reads
    512 word-pieces, takes "title [SEP] abstract" and uses the [CLS] vector. The natural candidate for this task.
The vectors are saved once, so the notebooks load them instead of recomputing:
    data/processed/<prefix>embeddings_<name>.npy      one row per paper, same order as the ids file
    data/processed/<prefix>embeddings_<name>_ids.csv  the paper_id of every row

References: Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER: Document-level representation
learning using citation-informed transformers. ACL 2020. https://doi.org/10.18653/v1/2020.acl-main.207 ; Singh, A., D'Arcy, M.,
Cohan, A., Downey, D., & Feldman, S. (2023). SciRepEval. arXiv. https://doi.org/10.48550/arXiv.2211.13308 ; Reimers, N., &
Gurevych, I. (2019). Sentence-BERT. EMNLP-IJCNLP 2019. https://doi.org/10.18653/v1/D19-1410 ; Wang, W., Wei, F., Dong, L.,
Bao, H., Yang, N., & Zhou, M. (2020). MiniLM. NeurIPS 33. https://doi.org/10.48550/arXiv.2002.10957
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models

from . import config as C

MODELS = {  # name -> (model id on the Hugging Face hub, pooling, max word-pieces)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", 256),
    "specter2": ("allenai/specter2_base", "cls", 512),
}
MODEL = MODELS["minilm"][0]      # kept for the iteration-3 notebook, which embeds the hold-out set on the fly


def load_model(name: str) -> tuple[SentenceTransformer, str]:
    model_id, pooling, max_len = MODELS[name]
    if pooling == "mean":
        return SentenceTransformer(model_id), ". "
    transformer = models.Transformer(model_id, max_seq_length=max_len)                       # the BERT-like encoder
    pool = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="cls")  # take the [CLS] vector
    return SentenceTransformer(modules=[transformer, pool]), " [SEP] "


def main(name: str = "minilm", prefix: str = "", table: str = "papers.csv", field: str | None = None) -> None:
    papers = pd.read_csv(C.PROCESSED_DIR / table, usecols=["paper_id", "field", "title", "abstract"], low_memory=False)
    if field:                                                      # keep one field only (e.g. "biotech")
        papers = papers[papers["field"] == field].reset_index(drop=True)
    model, joiner = load_model(name)
    texts = (papers["title"].fillna("") + joiner + papers["abstract"].fillna("")).tolist()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)  # unit length -> cosine = dot product
    np.save(C.PROCESSED_DIR / f"{prefix}embeddings_{name}.npy", vectors.astype(np.float32))
    papers[["paper_id"]].to_csv(C.PROCESSED_DIR / f"{prefix}embeddings_{name}_ids.csv", index=False)
    print(f"{vectors.shape[0]} papers x {vectors.shape[1]} numbers  ->  {C.PROCESSED_DIR / (prefix + 'embeddings_' + name + '.npy')}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(name=args[0] if args else "minilm", prefix=args[1] if len(args) > 1 else "", table=args[2] if len(args) > 2 else "papers.csv",
         field=args[3] if len(args) > 3 else None)
