import numpy as np
import scipy as sp
import os
import time
import re
import math
import matplotlib.pyplot as plt


class VectorSpaceModel:

    

    def __init__(self, inverted_index: dict, file_path:str = "data/docs/normalized",tf_mode:int = 3, idf_mode:int=3) -> None:
        self.inverted_index = inverted_index
        self.list_of_docs = sorted(os.listdir(file_path))  
        self.file_path = file_path
        self.TF_MODE = tf_mode
        self.IDF_MODE = idf_mode


    #///////////// Private Methods ///////////////

    def __binary_tf(self,frequency):
        return 1 if frequency > 0 else 0


    def __raw_tf(self,frequency):
        return frequency

   
    def __log_tf(self,frequency):
        return 1 + math.log(frequency) if frequency > 0 else 0


    def __double_normalization_tf(self,frequency, max_frequency):
        return 0.5 + 0.5 * (frequency / max_frequency) if max_frequency > 0 else 0


    def __k_normalization_tf(self,frequency, max_frequency, K=0.5):
        return K + (1 - K) * (frequency / max_frequency) if max_frequency > 0 else 0


    def __idf_mono(self):
        return 1


    def __idf_inverse(self,n_docs, doc_freq):
        return math.log(n_docs / doc_freq) if doc_freq > 0 else 0


    def __idf_log(self,n_docs, doc_freq):
        return math.log(1 + (n_docs / doc_freq)) if doc_freq > 0 else 0


    def __idf_max_inverse(self,n_docs, doc_freq, max_freq):
        return math.log(1 + (max_freq / doc_freq)) if doc_freq > 0 else 0


    def __term_occurances_in_document(self,term:str, document:str) -> int:
        '''
        Find term occurances per document. Uses the InvertedIndex class to look up the index
            - `term` Used to lookup for occurances
            - `document` the document to search for term occurances
        '''
        inverted_index = self.inverted_index
        return len(inverted_index[term][document])
    

    def __total_terms_in_document(self,document:str) -> int:
        '''
        Find the number of terms in the document
        '''
        return len(document.split())


    def __total_documents_with_term(self,term:str) -> int:
        '''
        Finds the number of documents that contain the term
        '''
        inverted_index = self.inverted_index
        return len(inverted_index[term])
    

    def __total_documents(self) -> int:
        '''
        Returns total number of documents inside the specified path
        '''
        return len(self.list_of_docs)
    

    def __get_max_frequency(self,document:str) -> str:
        '''
        Returns the maximum frequency of a term in a document
        '''
        with open(os.path.join(self.file_path, document), 'r') as f:    
            doc = f.read()
            max_frequency = 0
            for term in set(doc.split()):
                max_f_tmp = self.__term_occurances_in_document(term,document)
                if max_f_tmp > max_frequency:
                    max_frequency = max_f_tmp
            return max_frequency


    #///////////// Public Methods ///////////////
    def term_frequency(self,term:str, document:str) -> float:
        '''Calculates the t_f parameter for a specific term and document'''
        mode = self.TF_MODE
        if mode == 0:
            return self.__term_occurances_in_document(term,document)/self.__total_terms_in_document(document)
        if mode == 1:
            return self.__binary_tf(self.__term_occurances_in_document(term,document))
        elif mode == 2:
            return self.__raw_tf(self.__term_occurances_in_document(term,document))
        elif mode == 3:
            return self.__log_tf(self.__term_occurances_in_document(term,document))
        elif mode == 4:
            max_frequency = self.__get_max_frequency(document)
            return self.__double_normalization_tf(self.__term_occurances_in_document(term,document), max_frequency)
        elif mode == 5:
            max_frequency = self.__get_max_frequency(document)
            return self.__k_normalization_tf(self.__term_occurances_in_document(term,document), max_frequency)
    

    def inverse_document_frequency(self,term:str) -> float:
        '''Calculates the idf '''
        mode = self.IDF_MODE
        if mode == 0:
            return np.log(self.__total_documents()/self.__total_documents_with_term(term))
        if mode == 1:
            return self.__idf_mono()
        if mode == 2:
            return self.__idf_inverse(self.__total_documents(), self.__total_documents_with_term(term))
        if mode == 3:
            return self.__idf_log(self.__total_documents(), self.__total_documents_with_term(term))
        if mode == 4:
            try:
                max_frequency = max([self.__term_occurances_in_document(term,doc) for doc in self.list_of_docs])
            except:
                max_frequency = 0
            return self.__idf_max_inverse(self.__total_documents(), self.__total_documents_with_term(term), max_frequency)
        


    def term_score(self,term: str,document: str) -> float:
        return self.term_frequency(term,document)*self.inverse_document_frequency(term)


    def optimize(self):
        for i in range(0,6):
            for j in range(0,5):
                # try:
                    
                    avg_f_measure = 0
                    self.TF_MODE = i
                    self.IDF_MODE = j
                    sp_matrix = self.generate_document_vectors(os.path.join(os.path.dirname(__file__),'../../data/docs/normalized'))
                    for query in (os.listdir('data/queries/normalized')):
                        with open(os.path.join('data/queries/normalized', query), 'r') as f:
                            query_vector = self.query_vectorize(os.path.join('data/queries/normalized', query))
                            cos_similarities = self.get_cos_similarities(sp_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
                            pred_relev = self.get_top_k_indices(cos_similarities, 20)
                            pred_relev = self.index_to_doc(pred_relev)
                            pred_relev = self.extract_doc_id(pred_relev)
                            true_relev = self.get_true_relevant(self.extract_doc_id(query))
                            precision = self.precision(pred_relev, true_relev)
                            recall = self.recall(pred_relev, true_relev)
                            f_measure = self.harmonic_mean(precision, recall)
                            avg_f_measure += f_measure
                    avg_f_measure = avg_f_measure/19
                    print(f"Average f measure for tf mode {i} and idf mode {j} is {avg_f_measure}")
                # except:
                #     print(f"Error with tf mode {i} and idf mode {j}")
                #     continue
                
        

    def vectorize(self, file_path: str) -> np.array:
        '''
        Converts a text (query or document content) to a vector.
        '''

        with open(file_path, 'r') as f:
            text = f.read()
        
            inverted_index = self.inverted_index
            vector = np.zeros(len(inverted_index))
            terms = text.split()
            unique_terms = set(terms)
            
            for term in unique_terms:
                try:
                    vector[list(inverted_index.keys()).index(term)] = self.term_score(term, os.path.basename(file_path))
                except ValueError:
                    print(f"Term {term} not found in the inverted index")
        
        sparse_vector = sp.sparse.csr_matrix(vector)
        return sparse_vector


    def query_vectorize(self, file_path:str) -> np.array:
        ''''''
        # start_time = time.time()
        with open(file_path, 'r') as f:
            query = f.read()
            inverted_index = self.inverted_index
            query_vector = np.zeros(len(inverted_index))
            words = query.split()
            for term in set(words):
                try:
                    tf = words.count(term)/len(words)
                    # tf = (1 + np.log(query.split().count(term)/len(query.split()))) #normalized
                    query_vector[list(inverted_index.keys()).index(term)] = self.inverse_document_frequency(term) * tf
                except:
                    pass

        sparse_query_vector = sp.sparse.csr_matrix(query_vector)
        # print("--- %f seconds to create vector---" % (time.time() - start_time))
        return sparse_query_vector


    def generate_document_vectors(self,file_path:str,save_to_npz:bool = False, save_path = 'saves/document_sparse_vectors.npz') -> sp.sparse.csr_matrix:
        '''
        This the main VSM generator funcion. Generates the document vectors for all the documents in the file_path
        '''
        document_vectors = sp.sparse.csr_matrix((0, len(self.inverted_index)))  # Initialize an empty sparse matrix
        # sort files by name
        for file in self.list_of_docs:
            vec = self.vectorize(os.path.join(file_path, file))
            document_vectors = sp.sparse.vstack((document_vectors,vec))
            # print(f"Document vector for {file} generated")
        if save_to_npz:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            print("Saving document vectors to: ",save_path)
            sp.sparse.save_npz(save_path, document_vectors)
        return document_vectors
    
    #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def get_cos_similarities(self, docs:sp.sparse.csr_matrix, query:sp.sparse.csr_matrix) -> np.array:
       
        dot_products = np.dot(docs, query.T).flatten()
        norms = np.linalg.norm(docs, axis=1) * np.linalg.norm(query)

        cos_similarities = np.abs(np.divide(dot_products, norms))

        return cos_similarities
    
    
    def get_top_k_indices(self, cos_similarities:np.array, k:int) -> list:
        '''
        Returns the indices of the top k documents from the cosine similarities array
        '''
        return (np.argsort(cos_similarities)[-k:][::-1]).tolist()


    def index_to_doc(self, index:int|list) -> str|list:
        '''
        Indices and doc names do not have an 1:1 mapping, this function returns the doc name for a given index
        '''
        if isinstance(index,int):
            return self.list_of_docs[index]
        elif isinstance(index,list):
            return [self.list_of_docs[i] for i in index]
        

    def extract_doc_id(self, file: str|list) -> int:
        '''
        Extracts the numeric part of a file name and returns it as an integer.
        Example: "00153.txt" -> 153
        '''
        if isinstance(file, str):
            match = re.search(r'\d+', file)
            if match:
                return int(match.group())
            else:
                raise ValueError("No numeric part found in the file name")
            
        elif isinstance(file, list):
            return [self.extract_doc_id(f) for f in file]


    def load_precomputed_vsm(self, path_to_file:str = 'saves/document_vectors.npz' ) -> sp.sparse.csr_matrix:
        '''
        Loads the precomputed document vectors from a .npz file 
        '''
        docs_sparse_matrix = sp.sparse.load_npz(os.path.join(os.path.dirname(__file__), path_to_file ))
        return docs_sparse_matrix


    def compare_query(self, query_vector , docs_sparse_matrix: sp.sparse.csr_matrix = None, k_res: int=5 ) -> list:
        '''
        Compares a query vector to the documents in the docs_sparse_matrix and return a list of top k relevant documents
        '''
        relevant_docs = []

        cos_similarities = self.get_cos_similarities(docs_sparse_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
                
        for d_i in self.get_top_k_indices(cos_similarities, k_res):
            relevant_docs.append( self.index_to_doc(d_i))

        return self.extract_doc_id(relevant_docs)


    def compare_queries(self, queries_path: str = 'data/queries/normalized', docs_sparse_matrix: sp.sparse.csr_matrix = None) -> list:
        '''
        Compares the queries in the queries_path to the documents in the docs_sparse_matrix
        '''
        relevant_docs = []
        results = []
        # with open(os.path.join(os.path.dirname(__file__), '../res/results.txt'), 'w+') as f:
        #     f.write("") # clear results.txt
        for query in (os.listdir(queries_path)):

            results.append(f"\nquery file: {query}\n")

            with open(os.path.join(queries_path, query), 'r') as f:
                query_vector = self.query_vectorize(os.path.join(queries_path, query))
                cos_similarities = self.get_cos_similarities(docs_sparse_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
                # print documents in a pretty manner and write them in results.txt
                for order, i in enumerate(self.get_top_k_indices(cos_similarities, 20)):
                    results.append(f"{order+1}.  {self.list_of_docs[i]}: {cos_similarities[i]}\n")
                    relevant_docs.append(self.list_of_docs[i])
            results.append("\n")
        #Terminal print 
        output = ''.join(results)
        print(output)

        file_path = os.path.join(os.path.dirname(__file__), '../../data/results/vsm_results.txt')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write to the file
        with open(file_path, 'w+') as f:
            f.write(output)

        return relevant_docs


    def get_true_relevant(self, file_index:int):
        '''
        Returns the relevant documents for a specific query as a list of integers
        '''
        with open(os.path.join(os.path.dirname(__file__), '../../data/Relevant_20.txt'), 'r') as f:
            lines = f.readlines()
            return [int(x) for x in lines[file_index - 1].split()[1:]]
        

    def precision_at_k(self, pred_rel:list, true_rel:list, k:int):
        '''
        Calculates the precision at k
        '''
        relevant = 0
        for prediction in pred_rel[:k]:
            if prediction in true_rel:
                relevant = relevant + 1
        return relevant/k
    
    def precision(self, pred_rel:list, true_rel:list):
        '''
        Calculates the precision at k
        '''
        relevant = 0
        for prediction in pred_rel:
            if prediction in true_rel:
                relevant = relevant + 1
        return relevant/len(pred_rel)
    

    def reciproral_ranking(self, pred_relev:list, true_relev:list):
        for i, prediction in enumerate(pred_relev):
            if prediction in true_relev:
                return 1/(i+1)
        return 0
    

    def recall(self, pred_rel:list, true_rel:list):
        relevant = 0
        for prediction in pred_rel:
            if prediction in true_rel:
                relevant = relevant + 1
        return relevant/len(true_rel)
    

    def harmonic_mean(self, precision:float, recall:float):
        if precision + recall == 0:
            return 0
        return 2*(precision*recall)/(precision + recall)
    

    def print_metrics(self,docs_sparse_matrix:sp.sparse.csr_matrix ):
        with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'w+') as f:
            f.write("")
        average_p_at_5 = 0
        average_p_at_10 = 0
        average_p_at_20 = 0
        m_r_r = 0
        average_f_score = 0
        for query in (os.listdir('data/queries/normalized')):
            # print(f"query file: {query}")
            with open(os.path.join('data/queries/normalized', query), 'r') as f:
                # with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'a') as f:
                    # f.write(f"\nquery file: {query}\n")
                query_vector = self.query_vectorize(os.path.join('data/queries/normalized', query))
                cos_similarities = self.get_cos_similarities(docs_sparse_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
                pred_relev = self.get_top_k_indices(cos_similarities, 20)
                pred_relev = self.index_to_doc(pred_relev)
                pred_relev = self.extract_doc_id(pred_relev)
                true_relev = self.get_true_relevant(self.extract_doc_id(query))
                precision = self.precision(pred_relev, true_relev)
                recall = self.recall(pred_relev, true_relev)
                precision_at_5 = self.precision_at_k(pred_relev, true_relev, 5)
                precision_at_10 = self.precision_at_k(pred_relev, true_relev, 10)
                precision_at_20 = self.precision_at_k(pred_relev, true_relev, 20)
                harmonic_mean = self.harmonic_mean(precision, recall)
                reciproral_ranking = self.reciproral_ranking(pred_relev, true_relev)
                average_p_at_5 += precision_at_5
                average_p_at_10 += precision_at_10
                average_p_at_20 += precision_at_20
                m_r_r += reciproral_ranking
                average_f_score += harmonic_mean
                # with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'a') as f:
                #     f.write(f"Precision: {precision}\nRecall: {recall}\nPrecision at 5: {precision_at_5}\nPrecision at 10: {precision_at_10}\nPrecision at 20: {precision_at_20}\nHarmonic mean: {harmonic_mean}\nReciproral ranking: {reciproral_ranking}\n")
                # print(f"Precision: {precision}\nRecall: {recall}\nPrecision at 5: {precision_at_5}\nPrecision at 10: {precision_at_10}\nPrecision at 20: {precision_at_20}\nHarmonic mean: {harmonic_mean}\nReciproral ranking: {reciproral_ranking}\n")
                # print()
        average_p_at_5 = average_p_at_5/19
        average_p_at_10 = average_p_at_10/19
        average_p_at_20 = average_p_at_20/19
        m_r_r = m_r_r/19
        average_f_score = average_f_score/19
        with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'a') as f:
            f.write(f"\nAverage precision at 5: {average_p_at_5}\nAverage precision at 10: {average_p_at_10}\nAverage precision at 20: {average_p_at_20}\nMean reciproral ranking: {m_r_r}\nAverage f score: {average_f_score}\n")
        print(f"\nAverage precision at 5: {average_p_at_5}\nAverage precision at 10: {average_p_at_10}\nAverage precision at 20: {average_p_at_20}\nMean reciproral ranking: {m_r_r}\nAverage f score: {average_f_score}\n")


    