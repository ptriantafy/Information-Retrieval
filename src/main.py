import os

from VectorSpace.preprocess import Preprocessing, InvertedIndex, STEM, LEMMATIZE
from VectorSpace.vsm import VectorSpaceModel


def main():

    curr_path = os.path.dirname(__file__)

    # Initialize Preprocessing class
    preproc = Preprocessing()
    # Initialize InvertedIndex class
    inv = InvertedIndex()


    # Flatten, remove stopwords, and stem/lemmatize the text
    preproc.flatten_text(os.path.join(curr_path,'../data/docs/raw'))
    preproc.remove_stop_words(os.path.join(curr_path,'../data/docs/flattened'))
    preproc.normalise_text(STEM, os.path.join(curr_path,'../data/docs/no_stop_words'))

    preproc.normalise_text(STEM, os.path.join(curr_path,'../data/docs/no_stop_words'))

  
    inverted_index = inv.generate_inverted_index(os.path.join(curr_path,'../data/docs/normalized'), True)
    inverted_index = inv.import_inverted_index(os.path.join(curr_path,'../data/inverted_index.txt'))


    # Initialize VectorSpaceModel class
    vsm = VectorSpaceModel(inverted_index, os.path.join(curr_path,'../data/docs/normalized'))
    vsm.generate_document_vectors(os.path.join(curr_path,'../data/docs/normalized'),True,os.path.join(curr_path,'saves/document_vectors.npz'))
    vsm.compare_queries(queries_path=os.path.join(curr_path,'../data/queries/normalized')\
                        ,docs_sparse_matrix=vsm.load_precomputed_vsm(os.path.join(curr_path,'saves/document_vectors.npz')))
       
    vsm.print_metrics(vsm.load_precomputed_vsm(os.path.join(curr_path,'saves/document_vectors.npz')))




if __name__ == '__main__':
    main()