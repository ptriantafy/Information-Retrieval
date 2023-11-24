import nltk
import os

# nltk.download('stopwords')    # uncomment if you don't have stopwords downloaded
stop_words = nltk.corpus.stopwords.words('english')
# remove stop words from each line in data/queries.txt
text = ''
with open(os.path.join('data/queries.txt'), 'r') as f:
    for line in f.readlines():
        text += ' '.join([word for word in line.split() if word not in stop_words])
        text += '\n'
with open(os.path.join('data/queries_processed.txt'), 'w+') as f:
    f.write(text)

# stem and save each query in a separate file
stemmer = nltk.stem.PorterStemmer()
with open(os.path.join('data/queries_processed.txt'), 'r') as f:
    for i, line in enumerate(f.readlines()):
        text = ' '.join([stemmer.stem(word) for word in line.split()])
        with open(os.path.join('data/docs/processed', 'query_' + str(i+1) + '.txt'), 'w+') as f:
            f.write(text)




