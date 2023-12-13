import os

with open (os.path.join('data/tsv/docs.tsv'), 'w+') as doc_tsv_out:
    for i, file in enumerate(sorted(os.listdir("data/docs/processed"))):
        with open('data/docs/processed/'+file, 'r') as doc:
            doc_tsv_out.write(str(i)+"\t"+doc.read()+"\n")

with open (os.path.join('data/tsv/queries.tsv'), 'w+') as quer_tsv_out:
    for file in sorted(os.listdir("data/Queries_Processed")):
        with open('data/Queries_Processed/'+file, 'r') as quer:
            quer_tsv_out.write(file[6:-4]+"\t"+quer.read()+"\n")