import nltk
import os

# read all txt files in data/docs/raw and make them single line
for file in os.listdir('data/docs/raw'):
    with open(os.path.join('data/docs/raw', file), 'r') as f:
        text = f.read().replace('\n', ' ')
    with open(os.path.join('data/docs/raw', file), 'w') as f:
        f.write(text)

#remove stop words from data/docs/processed
# nltk.download('stopwords')    # uncomment if you don't have stopwords downloaded
stop_words = nltk.corpus.stopwords.words('english')
for file in os.listdir('data/docs/processed'):
    with open(os.path.join('data/docs/processed', file), 'r') as f:
        text = f.read()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    with open(os.path.join('data/docs/processed', file), 'w+') as f:
        f.write(text)

# stem all txt files in data/docs/raw and save them in data/docs/processed
stemmer = nltk.stem.PorterStemmer()
for file in os.listdir('data/docs/raw'):
    with open(os.path.join('data/docs/raw', file), 'r') as f:
        text = f.read()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    with open(os.path.join('data/docs/processed', file), 'w+') as f:
        f.write(text)




