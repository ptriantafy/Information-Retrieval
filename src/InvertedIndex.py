import os


class InvertedIndex:

    def __init__(self, data_path = "data/docs/processed") -> None:
        self.documents_path = data_path
        self.inverted_index = {}


    def generateInvertedIndex(self):
        documents_path = self.documents_path
        inverted_index = self.inverted_index


        for file in os.listdir(documents_path):
            with open(os.path.join(documents_path, file), 'r') as f:
                text = f.readline().split(' ')
                # add to inverted index saving frequency of each word and position
                for i, word in enumerate(text):
                    if word not in inverted_index:
                        # create file dict for the new word 
                        file_dict = {}
                        # add file to file dict and save position of word to a list
                        file_dict[file] = [i]
                        inverted_index[word] = file_dict
                    else:
                        # if word already in inverted index, add file to file dict and save position of word to a list
                        if file not in inverted_index[word]:
                            inverted_index[word][file] = [i]
                        else:
                            inverted_index[word][file].append(i)

    
    def exportInvertedIndex(self):
        inverted_index = self.inverted_index
        #this is for debugging perposes
        with open('data/inverted_index.txt', 'w+') as f:
            for i, word in enumerate(inverted_index):
                f.write(f"{word}: ")
                for file in inverted_index[word]:
                    for position in inverted_index[word][file]:
                        f.write(f"({file}, {position}) ")
                f.write('\n')
            