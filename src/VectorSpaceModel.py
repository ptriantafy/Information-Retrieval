import numpy as np
import scipy as sp
import os
import InvertedIndex 

class VectorSpaceModel:

    def __init__(self) -> None:
        ii_obj = InvertedIndex.InvertedIndex()
        self.inverted_index = ii_obj.generateInvertedIndex()

    #///////////// Private Methods ///////////////

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
        

    #///TODO use sparse matrix
    def documentVector(self,document) -> np.array:
        inverted_index = self.inverted_index
        document_vector = []
        for term in inverted_index:
            document_vector.append(self.termScore(term,document))
        return np.array(document_vector)
    


    def generateDocumentVectors(self) -> np.array:
        document_vectors = []
        for file in os.listdir("data/docs/processed"):
            document_vectors.append(self.documentVector(file))
        return np.array(document_vectors)
    


vsm = VectorSpaceModel()

print(vsm.termScore('aeruginosa','00001.txt'))

        