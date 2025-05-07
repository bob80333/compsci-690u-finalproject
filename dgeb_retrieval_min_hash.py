# using rensa library for fast minhash implementation
from rensa import RMinHash
import numpy as np

import argparse


# based off rensa library example
def rensa_minhash(text, num_perm=512, k_gram=7):
    gram_list = list(text)
    k_grams = [
        "".join(gram_list[i : i + k_gram]) for i in range(len(gram_list) - k_gram + 1)
    ]
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(k_grams)
    return m


# this version adds positional information to the k_grams
# it was done in the hope that the positional information would help with the retrieval task
# it makes the results much worse, so it is not used
def rensa_positional_minhash(text, num_perm=128, k_gram=3):
    gram_list = list(text)
    # just add the starting index of the k_gram to the end of the k_gram to give positional info
    k_grams = [
        "".join(gram_list[i : i + k_gram] + [str(i)])
        for i in range(len(gram_list) - k_gram + 1)
    ]
    m = RMinHash(num_perm=num_perm, seed=42)
    m.update(k_grams)
    return m


# from dgeb github readme example:
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks.tasks import Modality

import datasets

from torch.utils.data import DataLoader
import tqdm
from typing import List, Literal, Optional
from transformers import (
    DefaultDataCollator,
)
import logging

logger = logging.getLogger(__name__)

import torch


class MyModel(BioSeqTransformer):

    def __init__(
        self,
        model_name: str = "RMinHash",
        k_mers: int = 7,
        num_perm: int = 512,
        # leaving these to not break the dgeb library
        # but not using them
        layers: Optional[List[int] | Literal["mid"] | Literal["last"]] = None,
        devices: List[int] = [0],
        num_processes: int = 16,
        max_seq_length: int = 1024,
        l2_norm: bool = False,
        batch_size: int = 128,
        pool_type: str = "mean",
    ):
        # super().__init__(model_name)

        self.model_name = model_name

        self.id = self.__class__.__name__

        self.num_param = 0
        self.data_collator = DefaultDataCollator()
        self.gpu_count = 0
        self.l2_norm = 0

        self.device = "cpu"
        self.num_processes = 0
        self.max_seq_length = max_seq_length
        self.batch_size = 8
        self.pool_type = pool_type

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return 0

    @property
    def embed_dim(self) -> int:
        return 128

    # overwriting the encode method to use the rensa library for minhash
    def encode(self, sequences, **kwargs) -> np.ndarray:
        """Returns a list of embeddings for the given sequences.
        Args:
            sequences (`List[str]`): List of sequences to encode
        Returns:
            `np.ndarray`: Embeddings for the given sequences of shape [num_sequences, num_layers, embedding_dim].
        """

        dataset = datasets.Dataset.from_dict({"input_seqs": sequences})

        print(len(dataset))

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_processes,
            # collate_fn=self.data_collator,
        )

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(
            data_loader, desc="encoding", mininterval=10, disable=len(sequences) < 128
        ):
            batch_dict = {k: v for k, v in batch_dict.items()}

            # get the input sequences
            input_seqs = batch_dict["input_seqs"]
            # apply the rensa minhash function
            for seq in input_seqs:
                seq = rensa_positional_minhash(seq)
                encoded_embeds.append(seq)

        return encoded_embeds


# overwrite the run_retrieval_task function to allow monkey patching the RetrievalEvaluator to add evaluation for MinHash, not just cosine similarity

"""
Retrieval tasks find functionally relevant genes in a corpus of genes based on a query gene.
Typically corpus is derived from a different phylogenetic group than the query genes.
"""

import logging
from collections import defaultdict

from dgeb.evaluators import RetrievalEvaluator
from dgeb.modality import Modality
from dgeb.models import BioSeqTransformer
from dgeb.tasks import TaskMetadata, TaskResult

logger = logging.getLogger(__name__)


def min_hash_compare(a, b):
    """create similarity matrix for 2 lists of minhashes"""

    output_matrix = torch.zeros((len(a), len(b)))
    for i, a_hash in enumerate(a):
        for j, b_hash in enumerate(b):
            output_matrix[i][j] = a_hash.jaccard(b_hash)

    return output_matrix


def run_retrieval_task(model: BioSeqTransformer, metadata: TaskMetadata) -> TaskResult:
    """Evaluate retrieval task. Utilizes the Retrieval evaluator."""
    if len(metadata.datasets) != 2:
        raise ValueError("Retrieval tasks require 3 datasets: corpus, query and qrels.")
    corpus_ds = metadata.datasets[0].load()["train"]
    query_ds = metadata.datasets[0].load()["test"]
    qrels = metadata.datasets[1].load()
    corpus_embeds = model.encode(corpus_ds["Sequence"])
    query_embeds = model.encode(query_ds["Sequence"])
    qrels_dict = defaultdict(dict)

    def qrels_dict_init(row):
        qrels_dict[str(row["query_id"])][str(row["corpus_id"])] = int(row["fuzz_ratio"])

    # Populate `qrels_dict` from the dataset.
    # See https://github.com/cvangysel/pytrec_eval for qrels format.
    qrels.map(qrels_dict_init)
    qrels = qrels_dict
    layer_results = defaultdict(dict)
    evaluator = RetrievalEvaluator(
        corpus_embeds,
        query_embeds,
        corpus_ds["Entry"],
        query_ds["Entry"],
        qrels,
        score_function="min_hash_compare",
    )
    # monkey patch the evaluator to have the min_hash_compare function
    evaluator.score_functions["min_hash_compare"] = min_hash_compare
    layer_results["layers"]["nolayer"] = evaluator()
    logger.info(
        f"Layer: nolayer, Retrieval results: {layer_results['layers']['nolayer']}"
    )
    return TaskResult.from_dict(metadata, layer_results, model.metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval task")
    parser.add_argument(
        "--k-mers",
        type=int,
        default=7,
        help="Length of k-mers to use for minhashing",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=512,
        help="Number of permutations to use for minhashing",
    )
    args = parser.parse_args()

    model = MyModel(model_name="RMinHash", k_mers=args.k_mers, num_perm=args.num_perm)
    tasks = dgeb.get_tasks_by_modality(model.modality)
    tasks = [task for task in tasks if "retrieval" in task.metadata.type]
    # patch the tasks to use custom run_retrieval_task
    for task in tasks:
        task.run = lambda self, model: run_retrieval_task(model, self.metadata)
    # run the evaluation
    evaluation = dgeb.DGEB(tasks=tasks)
    evaluation.run(model)
