import numpy as np
import scipy as sp
import os
import InvertedIndex 
import time
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
        return len(document.split())



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
        # return 1 + np.log(self.__termOccurancesInDocument(term,document)/self.__totalTermsInDocument(document)) #normalized
    


    def inverseDocumentFrequency(self,term) -> float:
        return np.log(self.__totalDocuments()/self.__totalDocumentsWithTerm(term))
        # return np.log(1 + self.__totalDocuments()/self.__totalDocumentsWithTerm(term)) #normalized
    


    def termScore(self,term,document) -> float:
        return self.termFrequency(term,document)*self.inverseDocumentFrequency(term)

    def queryVector(self, query) -> np.array:
        # start_time = time.time()
        inverted_index = self.inverted_index
        query_vector = np.zeros(len(inverted_index))
        for term in set(query.split()):
            try:
                tf = 1/len(query.split())
                # tf = (1 + np.log(query.split().count(term)/len(query.split()))) #normalized
                query_vector[list(inverted_index.keys()).index(term)] = self.inverseDocumentFrequency(term) * tf
            except:
                pass
        sparse_query_vector = sp.sparse.csr_matrix(query_vector)
        # print("--- %f seconds to create vector---" % (time.time() - start_time))
        return sparse_query_vector

    def documentVector(self,document) -> np.array:
        inverted_index = self.inverted_index
        inverted_index_length = len(inverted_index) 
        document_vector = np.zeros(inverted_index_length)
        with open(os.path.join(os.path.dirname(__file__), '../data/docs/processed', document), 'r') as f:
            doc = f.read()
            for term in set(doc.split()): #///TODO split once
                #if the term is in the document, add its score to the document vector
                document_vector[list(inverted_index.keys()).index(term)] = self.termScore(term,document)
        sparse_document_vector = sp.sparse.csr_matrix(document_vector)
        return sparse_document_vector

    def generateDocumentVectors(self,save_to_npz = False) -> np.array:
        document_vectors = []
        # sort files by name
        for file in sorted(os.listdir("data/docs/processed")):
            document_vectors = sp.sparse.vstack((document_vectors, self.documentVector(file)))
            print(f"Document vector for {file} generated")
        if(save_to_npz == True):
            print("Saving document vectors to tmp/document_vectors.npz")
            sp.sparse.save_npz(os.path.join(os.path.dirname(__file__), 'tmp/document_vectors.npz'), document_vectors)
    
    def getCosSimilarities(self, docs, query) -> np.array:
        # start_time = time.time()
        dot_products = np.dot(docs, query.T)
        norms = np.linalg.norm(docs, axis=1) * np.linalg.norm(query)
        dot_products = dot_products.flatten()
        cos_similarities = np.abs(np.divide(dot_products, norms))
        # print("--- %f seconds to calculate cos similarities---" % (time.time() - start_time))
        return cos_similarities
    
    def getTopKDocs(self, cos_similarities, k) -> np.array:
        return np.argsort(cos_similarities.flatten())[-k:][::-1]
    
    def test(self):
        start_time = time.time()
        self.generateDocumentVectors()
        print("--- %f seconds ---" % (time.time() - start_time))
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
vsm.generateDocumentVectors(True)
# print(len(mapper))
list_of_files = sorted(os.listdir("data/docs/processed"))
sparse_matrix = sp.sparse.load_npz(os.path.join(os.path.dirname(__file__), 'tmp/document_vectors.npz'))
# print(sparse_matrix.shape)
with open(os.path.join(os.path.dirname(__file__), 'tmp/results.txt'), 'w+') as f:
    f.write("")
for file in (os.listdir("data/Queries_Processed")):
    print(f"query file: {file}")
    with open(os.path.join(os.path.dirname(__file__), '../data/Queries_Processed', file), 'r') as f:
        query = f.read()
        with open(os.path.join(os.path.dirname(__file__), 'tmp/results.txt'), 'a') as f:
            f.write(f"\nquery file: {file}\n")
        query_vector = vsm.queryVector(query)
        cos_similarities = vsm.getCosSimilarities(sparse_matrix.tocsr().toarray()[1:], query_vector.tocsr().toarray())
        # print(cos_similarities.shape)
        for order, i in enumerate(vsm.getTopKDocs(cos_similarities, 20)):
            print(f"{order+1}. {list_of_files[i]}: {cos_similarities[i]}")
            with open(os.path.join(os.path.dirname(__file__), 'tmp/results.txt'), 'a') as f:
                f.write(f"{order+1}.  {list_of_files[i]}: {cos_similarities[i]}\n")
        print()