import numpy as np
import scipy as sp
import os
import time
import re
import ast

class VectorSpaceModel:

    def __init__(self, inverted_index: dict, file_path:str = "data/docs/normalized") -> None:
        self.inverted_index = inverted_index
        self.list_of_docs = sorted(os.listdir(file_path))  


    #///////////// Private Methods ///////////////
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
    

    def __total_documents(self,folder_path:str = "data/docs/normalized") -> int:
        '''
        Returns total number of documents inside the specified path
        '''

        return len(self.list_of_docs)
    

    #///////////// Public Methods ///////////////
    def term_frequency(self,term:str, document:str) -> float:
        '''Calculates the t_f parameter for a specific term and document'''
        
        return self.__term_occurances_in_document(term,document)/self.__total_terms_in_document(document)
        # return 1 + np.log(self.__term_occurances_in_document(term,document)/self.__total_terms_in_document(document)) #normalized
    

    def inverse_document_frequency(self,term:str) -> float:
        '''Calculates the idf '''
        return np.log(self.__total_documents()/self.__total_documents_with_term(term))
        # return np.log(1 + self.__total_documents()/self.__total_documents_with_term(term)) #normalized


    def term_score(self,term: str,document: str) -> float:
        return self.term_frequency(term,document)*self.inverse_document_frequency(term)



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
            print(f"Document vector for {file} generated")
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




    def test_time(self,func:callable): #///TODO make it take fumction arguments
        start_time = time.time()
        func()
        print("--- %f seconds ---" % (time.time() - start_time))


    def load_precomputed_vsm(self, path_to_file:str = 'saves/document_vectors.npz' ) -> sp.sparse.csr_matrix:
        '''
        Loads the precomputed document vectors from a .npz file 
        '''
        docs_sparse_matrix = sp.sparse.load_npz(os.path.join(os.path.dirname(__file__), path_to_file ))
        return docs_sparse_matrix


    def compare_queries(self, queries_path:str = 'data/queries/normalized', docs_sparse_matrix:sp.sparse.csr_matrix = None):
        '''
        Compares the queries in the queries_path to the documents in the docs_sparse_matrix
        '''
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


    def extract_relevant(self, file_index:int):
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
    

    def precision(self, pred_rel:list, true_rel:list):
        relevant = 0
        for prediction in pred_rel:
            if prediction in true_rel:
                relevant = relevant + 1
        return relevant/len(pred_rel)


    def harmonic_mean(self, precision:float, recall:float):
        if precision + recall == 0:
            return 0
        return 2*(precision*recall)/(precision + recall)
    

    def colbert_metrics(self, file_path:str = '../tmp/colbert_results.txt'):
        '''
        Calculates the metrics for the colbert results
        '''
    
        with open(os.path.join(os.path.dirname(__file__),file_path), 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                query_number = int(parts[0])
                pred_docs = ast.literal_eval(parts[1])

                docs = self.extract_doc_id(self.index_to_doc(pred_docs))
                relevant_docs = self.extract_relevant(query_number)
                p_at_k = self.precision_at_k(pred_docs, relevant_docs, 20)
                reciprocal_rank = self.reciproral_ranking(pred_docs, relevant_docs)
                recall = self.recall(pred_docs, relevant_docs)
                precision = self.precision(docs, relevant_docs)
                harmonic_mean = self.harmonic_mean(precision, recall)
                print(query_number,precision)


    def print_metrics(self,docs_sparse_matrix:sp.sparse.csr_matrix ):
        with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'w+') as f:
            f.write("")
        for query in (os.listdir('data/queries/normalized')):
            print(f"query file: {query}")
            with open(os.path.join('data/queries/normalized', query), 'r') as f:
                with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'a') as f:
                    f.write(f"\nquery file: {query}\n")
                query_vector = self.query_vectorize(os.path.join('data/queries/normalized', query))
                cos_similarities = self.get_cos_similarities(docs_sparse_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
                pred_relev = self.get_top_k_indices(cos_similarities, 20)
                pred_relev = self.index_to_doc(pred_relev)
                pred_relev = self.extract_doc_id(pred_relev)
                true_relev = self.extract_relevant(self.extract_doc_id(query))
                precision = self.precision(pred_relev, true_relev)
                recall = self.recall(pred_relev, true_relev)
                precision_at_5 = self.precision_at_k(pred_relev, true_relev, 5)
                precision_at_10 = self.precision_at_k(pred_relev, true_relev, 10)
                precision_at_20 = self.precision_at_k(pred_relev, true_relev, 20)
                harmonic_mean = self.harmonic_mean(precision, recall)
                reciproral_ranking = self.reciproral_ranking(pred_relev, true_relev)
                with open(os.path.join(os.path.dirname(__file__), '../../data/results/vsm_metrics.txt'), 'a') as f:
                    f.write(f"Precision: {precision}\nRecall: {recall}\nPrecision at 5: {precision_at_5}\nPrecision at 10: {precision_at_10}\nPrecision at 20: {precision_at_20}\nHarmonic mean: {harmonic_mean}\nReciproral ranking: {reciproral_ranking}\n")
                print(f"Precision: {precision}\nRecall: {recall}\nPrecision at 5: {precision_at_5}\nPrecision at 10: {precision_at_10}\nPrecision at 20: {precision_at_20}\nHarmonic mean: {harmonic_mean}\nReciproral ranking: {reciproral_ranking}\n")
                print()
#///////main testing script///////

# vsm = VectorSpaceModel()
# # vsm.print_metrics()
# # vsm.print_results() 
# vsm.colbert_metrics()
