import os
import matplotlib.pyplot as plt
import numpy as np

from VectorSpace.preprocess import Preprocessing, InvertedIndex, STEM, LEMMATIZE
from VectorSpace.vsm import VectorSpaceModel
from colbert_metrics import get_relevant, get_true_relevant, colbert_metrics

curr_path = os.path.dirname(__file__)



def precision_recall_plot(relevant_docs:list, true_relevant:list, k:int, axes:plt.axes, label:str =''):
    '''
    Plots the precision recall curve for a given list of precision and recall values
    '''
        
    true_positives = 0
    tp = np.zeros(k)
    tp_fp = np.arange(1, k+1)

    for i,doc in enumerate(relevant_docs):
        
        if i >= k:
            break

        if doc in true_relevant:
            true_positives += 1
        tp[i] = true_positives

    precision = tp/tp_fp
    recall = tp/len(true_relevant)

    axes.plot(recall, precision,linestyle='-',label=label)

    axes.legend()



def main():


    # Initialize Preprocessing class
    preproc = Preprocessing()
    # Initialize InvertedIndex class
    inv = InvertedIndex()


    # Flatten, remove stopwords, and stem/lemmatize the text
    preproc.flatten_text(os.path.join(curr_path,'../data/docs/raw'))
    preproc.remove_stop_words(os.path.join(curr_path,'../data/docs/flattened'))
    preproc.normalise_text(STEM, os.path.join(curr_path,'../data/docs/no_stop_words'))

    preproc.save_queries_separately(os.path.join(curr_path,'../data/queries/queries.txt'))
    preproc.flatten_text(os.path.join(curr_path,'../data/queries/raw'))
    preproc.remove_stop_words(os.path.join(curr_path,'../data/queries/flattened'))
    preproc.normalise_text(STEM, os.path.join(curr_path,'../data/queries/no_stop_words'))

  
    inverted_index = inv.generate_inverted_index(os.path.join(curr_path,'../data/docs/normalized'), True)
    # inverted_index = inv.import_inverted_index(os.path.join(curr_path,'../data/inverted_index.txt')) #not working
    

    # Initialize VectorSpaceModel class
    vsm_1 = VectorSpaceModel(inverted_index, os.path.join(curr_path,'../data/docs/normalized'))
    # vsm_1.generate_document_vectors(os.path.join(curr_path,'../data/docs/normalized'),True,os.path.join(curr_path,'saves/document_vectors.npz'))

    vsm_2 = VectorSpaceModel(inverted_index, os.path.join(curr_path,'../data/docs/normalized'),tf_mode=4, idf_mode=1)
    # vsm_2.generate_document_vectors(os.path.join(curr_path,'../data/docs/normalized'),True,os.path.join(curr_path,'saves/document_vectors_2.npz'))
   

    vsm_1.print_metrics(vsm_1.load_precomputed_vsm(path_to_file='../saves/document_vectors.npz'))
    vsm_2.print_metrics(vsm_2.load_precomputed_vsm(path_to_file='../saves/document_vectors_2.npz'))
    colbert_metrics(os.path.join(curr_path,'../data/results/colbert_results_3.txt'))
       
    k = 10

    for i,query in enumerate(sorted(os.listdir(os.path.join(curr_path,'../data/queries/normalized')))):

        fig, axes = plt.subplots()
   
    
        # print("VSM_1")
        q_vec = vsm_1.query_vectorize(os.path.join('data/queries/normalized',query))
        relevant_docs = vsm_1.compare_query(query_vector=q_vec, docs_sparse_matrix=vsm_1.load_precomputed_vsm(path_to_file='../saves/document_vectors.npz'), k_res=k)
        true_relevant = vsm_1.get_true_relevant(i+1)
        precision_recall_plot(relevant_docs, true_relevant, k,axes, label='VSM_1')

        # print("VSM_2")
        q_vec = vsm_2.query_vectorize(os.path.join('data/queries/normalized',query))
        relevant_docs = vsm_2.compare_query(query_vector=q_vec, docs_sparse_matrix=vsm_1.load_precomputed_vsm(path_to_file='../saves/document_vectors.npz'), k_res=k)
        true_relevant = vsm_2.get_true_relevant(i+1)
        precision_recall_plot(relevant_docs, true_relevant, k,axes, label='VSM_2')

        # print("ColBERT")

        relevant_docs = get_relevant('../data/results/colbert_results_3.txt')
        true_relevant = get_true_relevant(i+1)
        precision_recall_plot(relevant_docs[i], true_relevant, k,axes, label='ColBERT')
        axes.set_xlabel('Recall')
        axes.set_ylabel('Precision')

        fig.suptitle(f'Query {i+1}')
        
    plt.show()


if __name__ == '__main__':
    main()