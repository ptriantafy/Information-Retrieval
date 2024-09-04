import nltk
import os

LEMMATIZE = nltk.stem.WordNetLemmatizer().lemmatize
STEM = nltk.stem.PorterStemmer().stem


class InvertedIndex:
    '''
    Inverted Index class. Generates an inverted index from a set of documents. The inverted index can either be used directly as dictionay or exported to a file. 
    Then it can be imported back to be used as a dictionary.
    '''

    def __init__(self) -> None:
        pass


    def generate_inverted_index(self,file_path:str, export:bool = False, save_path:str = None)->dict:
        
        inverted_index = {}

        for file in os.listdir(file_path):
            with open(os.path.join(file_path, file), 'r') as f:
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
        if export:
            if not save_path:
                save_path = os.path.join(file_path, '../..')
            os.makedirs(save_path, exist_ok=True)

            self.__export_inverted_index(inverted_index, os.path.join(save_path, 'inverted_index.txt'))

        return inverted_index

    
    def import_inverted_index(self, load_path: str) -> dict:
        inverted_index = {}
        
        with open(load_path, 'r') as f:
            for line in f:
                term, postings = line.split(': ')
                postings = postings.strip().split(') ')
                
                for posting in postings:
                    if posting:
                        file, position = posting.strip('()').split(', ')
                        if term not in inverted_index:
                            inverted_index[term] = {}
                        if file not in inverted_index[term]:
                            inverted_index[term][file] = []
                        inverted_index[term][file].append(int(position))
        
        return inverted_index


    
    def __export_inverted_index(self,inverted_index:dict ,save_path:str)->None:
        
        with open(save_path, 'w') as f:
            for i, word in enumerate(inverted_index):
                f.write(f"{word}: ")
                for file in inverted_index[word]:
                    for position in inverted_index[word][file]:
                        f.write(f"({file}, {position}) ")
                f.write('\n')


class Preprocessing:


    def __init__(self):
        # Check if NLTK stopwords are downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        
    def save_queries_separately(self, file_path:str, save_path:str = None)->None:
        """
        Save each query in a separate file
        """

        if not save_path:
            base_name = os.path.basename(file_path)
            save_path = os.path.join(file_path[:-len(base_name)], 'raw')

        os.makedirs(save_path, exist_ok=True)

        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                with open(os.path.join(save_path, 'query_' + str(i+1) + '.txt'), 'w+') as f:
                    f.write(line)

        

    def flatten_text(self, file_path:str, save_path:str = None)->None:
        """
        Read a text file and flatten it into a single line
        """
        

        if not save_path:
            save_path = os.path.join(file_path, '../flattened')

        os.makedirs(save_path, exist_ok=True)

        # read all txt files in specified folder and flatten them
        for file in os.listdir(file_path):
            if file == '.gitignore':
                continue
            with open(os.path.join(file_path, file), 'r') as f:
                text = f.read().replace('\n', ' ')
            with open(os.path.join(save_path, file), 'w') as f:
                f.write(text)


    def remove_stop_words(self, file_path:str, save_path:str = None)->None:
        """
        Remove stop words from text
        """

        stop_words = nltk.corpus.stopwords.words('english')

        if not save_path:
            save_path = os.path.join(file_path, '../no_stop_words')

        os.makedirs(save_path, exist_ok=True)

        # remove stop words from specified files
        for file in os.listdir(file_path):
            if file == '.gitignore':
                continue
            with open(os.path.join(file_path, file), 'r') as f:
                text = f.read().lower()
            text = ' '.join([word for word in text.split() if word not in stop_words])
            with open(os.path.join(save_path, file), 'w') as f:
                f.write(text)


    def normalise_text(self,normalisation_function:callable, file_path:str, save_path:str = None)->None:
        """
        Stem or lemmatize text
        """

        if not save_path:
            save_path = os.path.join(file_path, '../normalized')

        os.makedirs(save_path, exist_ok=True)

        stem_lem = normalisation_function
        for file in os.listdir(file_path):
            if file == '.gitignore':
                continue
            with open(os.path.join(file_path, file), 'r') as f:
                text = f.read()
            text = ' '.join([stem_lem(word) for word in text.split()])
            with open(os.path.join(save_path, file), 'w+') as f:
                f.write(text)
                

