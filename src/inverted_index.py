import os

documents_path = "data/docs/processed"
inverted_index = {}

for file in os.listdir(documents_path):
    with open(os.path.join(documents_path, file), 'r') as f:
        text = f.readline().split(' ')
        # add to inverted index saving frequency of each word and position
        for i, word in enumerate(text):
            if word not in inverted_index:
                inverted_index[word] = ['('+ file.replace('.txt', '') + ',' + str(i)+')']
            else:
                inverted_index[word].append('('+ file.replace('.txt', '') + ',' + str(i)+')')
with open('data/inverted_index.txt', 'w+') as f:
    for word in inverted_index:
        f.write(word + ' ' + ' '.join(inverted_index[word]) + '\n')
            