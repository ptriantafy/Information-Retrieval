import os
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

queries = os.path.join('data/tsv/queries.tsv')
collection = os.path.join('data/tsv/docs.tsv')
checkpoint = os.path.join('src/ColBERT_implementation/checkpoint/colbertv2.0')
index_name = "cfcolection.nbits=2"

queries = Queries(path=queries)
collection = Collection(path=collection)
print (f'Loaded {len(queries)} queries and {len(collection):,} passages')

if __name__ == '__main__': #without this it won't run...
    with Run().context(RunConfig(nranks=1, experiment="cfcollection")):
        config = ColBERTConfig(
            nbits=2,
            root="/experiments",
        )
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite=True)