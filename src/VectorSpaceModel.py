import numpy as np
import scipy as sp
import os
import InvertedIndex 
# import pandas as pd

class VectorSpaceModel:


    #///////////// Private Methods ///////////////
    def __init__(self) -> None:
        ii_obj = InvertedIndex.InvertedIndex()
        self.inverted_index = ii_obj.generateInvertedIndex(True)



    #Methods are pretty self expalnatory
    def __termOccurancesInDocument(self,term, document) -> int:
        inverted_index = self.inverted_index
        return len(inverted_index[term][document])
    


    def __totalTermsInDocument(self,document) -> int:
        inverted_index = self.inverted_index
        total_terms = 0
        for term in inverted_index:
            try:
                total_terms += len(inverted_index[term][document])
            except:
                pass
        return total_terms



    def __totalDocumentsWithTerm(self,term) -> int:
        inverted_index = self.inverted_index
        return len(inverted_index[term])
    


    def __totalDocuments(self,folder_path = "data/docs/processed") -> int:
        files = os.listdir(folder_path)

        # Filter files that end with '.txt'
        txt_files = [file for file in files if file.endswith('.txt')]

        return len(txt_files)
    


    #///////////// Public Methods ///////////////
    def termFrequency(self,term, document) -> float:
        return self.__termOccurancesInDocument(term,document)/self.__totalTermsInDocument(document)
    


    def inverseDocumentFrequency(self,term) -> float:
        return np.log(self.__totalDocuments()/self.__totalDocumentsWithTerm(term))
    


    def termScore(self,term,document) -> float:
        return self.termFrequency(term,document)*self.inverseDocumentFrequency(term)
        
    # queries should not be included in inverted index
    def queryVector(self,query) -> np.array:
        inverted_index = self.inverted_index
        query_vector = []
        for term in inverted_index:
            found = 0
            # if done like below, it can spot single syllables like "eg" which can
            # also be parts of words like "egg" or "leg"
            for word in query.split():
                if len(np.nonzero(query_vector)) == len(query.split()):
                    break
                if term == word:
                    found += 1
                    # print(f"Term {term} found {found} times in query")
            query_vector.append(found/len(query.split())*self.inverseDocumentFrequency(term))
            #  break after appending len(query.split()) non zero columns
        sparse_query_vector = sp.sparse.csr_matrix(query_vector)
        return sparse_query_vector

    def documentVector(self,document) -> np.array:
        inverted_index = self.inverted_index
        document_vector = []
        for term in inverted_index:

            #if the term is in the document, add its score to the document vector
            try:
                document_vector.append(self.termScore(term,document))

            #else insert 0
            except:
                document_vector.append(0)
        

        sparse_document_vector = sp.sparse.csr_matrix(document_vector)
        return sparse_document_vector
    

    def generateDocumentVectors(self) -> np.array:
        document_vectors = []
        # sort files by name --why?
        for file in os.listdir("data/docs/processed"):
            document_vectors = sp.sparse.vstack((document_vectors, self.documentVector(file)))
            print(f"Document vector for {file} generated")
        document_vectors = np.delete(document_vectors, (0), axis=0)
        return document_vectors
    
    def getCosSimilarities(self, docs, query) -> np.array:
    
        dot_products = np.dot(docs, query.T)
        norms = np.linalg.norm(docs) * np.linalg.norm(query)
        dot_products = dot_products.flatten()
        print("max value of vector of dot products: ", dot_products.max())
        print("max value of vector of norms: ", norms.max())
        # norms = np.linalg.norm(docs) * np.linalg.norm(query)
        cos_similarities = np.divide(dot_products, norms)
        return cos_similarities
    
    def getTopKDocs(self, cos_similarities, k) -> np.array:
        return np.argsort(cos_similarities.flatten())[-k:][::-1]
    #Following functions are used for debugging purposes
    # def count_unique_words(self, file_path = "data/docs/processed/00001.txt"):
    #     unique_words = set()

    #     with open(file_path, 'r') as file:
    #         for line in file:
    #             # Split the line into words and add them to the set
    #             words = line.split()
    #             unique_words.update(words)

    #     return len(unique_words)



    # def invertedIndexLength(self) -> int: 
    #     inverted_index = self.inverted_index
    #     return len(inverted_index)  






#///////main testing script///////
vsm = VectorSpaceModel()
# doc_vec = vsm.generateDocumentVectors()

# sp.sparse.save_npz(os.path.join(os.path.dirname(__file__), 'tmp/document_vectors.npz'), doc_vec)

sparse_matrix = sp.sparse.load_npz(os.path.join(os.path.dirname(__file__), 'tmp/document_vectors.npz'))
print(sparse_matrix.shape)
print(vsm.getCosSimilarities(sparse_matrix.toarray()[1,:], sparse_matrix.toarray()[1,:]))

for file in sorted(os.listdir("data/Queries_Processed")):
    print(f"query file: {file}")
    with open(os.path.join(os.path.dirname(__file__), '../data/Queries_Processed', file), 'r') as f:
        query = f.read()
        # print(query)
        query_vector = vsm.queryVector(query)
        # exclude row 0 empty vector (don't know why)
        cos_similarities = vsm.getCosSimilarities(sparse_matrix.tocsr().toarray(), query_vector.tocsr().toarray())
        # print(cos_similarities.shape)
        for i in vsm.getTopKDocs(cos_similarities, 20):
            print(f"{i}: {cos_similarities[i]}")
        # print()
