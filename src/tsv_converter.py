import os

if __name__ == '__main__':
    os.makedirs('data/docs/tsv/', exist_ok=True)
    os.makedirs('data/queries/tsv/', exist_ok=True)
    
    i = 0
    for file in sorted(os.listdir('data/docs/flattened')):
        with open('data/docs/flattened/' + file, 'r') as f:
            lines = f.readlines()
            with open('data/docs/tsv/docs.tsv', 'a') as tsv:
                for line in lines:
                    # write file name \t line
                    tsv.write(str(i) + '\t' + line)
                i = i + 1
                tsv.write('\n')
                    
    for file in os.listdir('data/queries/flattened'):
        if file.endswith('.txt'):
            with open('data/queries/flattened/' + file, 'r') as f:
                lines = f.readlines()
                with open('data/queries/tsv/queries.tsv', 'a') as tsv:
                    for line in lines:
                        tsv.write(file[6:-4] + '\t' + line)
                    tsv.write('\n')

