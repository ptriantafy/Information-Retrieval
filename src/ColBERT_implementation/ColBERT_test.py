import os

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

queries = os.path.join('data/tsv/queries.tsv')
collection = os.path.join('data/tsv/docs.tsv')
queries = Queries(path=queries)
collection = Collection(path=collection)

print (f'Loaded {len(queries)} queries and {len(collection):,} passages')

