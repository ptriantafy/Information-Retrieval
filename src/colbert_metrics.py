import re
import os
import ast

list_of_docs = sorted(os.listdir("data/docs/normalized"))  
def colbert_metrics(file_path:str = '../data/results/colbert_results.txt'):
    '''
    Calculates the metrics for the colbert results
    '''
    average_p_at_5 = 0
    average_p_at_10 = 0
    average_p_at_20 = 0
    m_r_r = 0
    average_f_score = 0
    with open(os.path.join(os.path.dirname(__file__), file_path), 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            query_number = int(parts[0])
            pred_docs = ast.literal_eval(parts[1])
            docs = extract_doc_id(index_to_doc(pred_docs))
            relevant_docs = extract_relevant(query_number)
            p_at_20 = precision_at_k(docs, relevant_docs, 20)
            p_at_10 = precision_at_k(docs, relevant_docs, 10)
            p_at_5 = precision_at_k(docs, relevant_docs, 5)
            print("Precision at 5: ", p_at_5)
            print("Precision at 10: ", p_at_10)
            print("Precision at 20: ", p_at_20)
            average_p_at_20 = average_p_at_20 + p_at_20
            average_p_at_10 = average_p_at_10 + p_at_10
            average_p_at_5 = average_p_at_5 + p_at_5
            r_r = reciproral_ranking(docs, relevant_docs)
            m_r_r = m_r_r + r_r
            rec = recall(docs, relevant_docs)
            prec = precision(docs, relevant_docs)
            f_score = harmonic_mean(prec, rec)
            average_f_score = average_f_score + f_score
    average_p_at_5 = average_p_at_5/19
    average_p_at_10 = average_p_at_10/19
    average_p_at_20 = average_p_at_20/19
    m_r_r = m_r_r/19
    average_f_score = average_f_score/19
    print("Average Precision at 5: ", average_p_at_5)
    print("Average Precision at 10: ", average_p_at_10)
    print("Average Precision at 20: ", average_p_at_20)
    print("Mean Reciprocal Rank: ", m_r_r)
    print("Average F1 Score: ", average_f_score)

def index_to_doc(index:int|list) -> str|list:
    '''
    Indices and doc names do not have an 1:1 mapping, this function returns the doc name for a given index
    '''
    if isinstance(index,int):
        return list_of_docs[index]
    elif isinstance(index,list):
        return [list_of_docs[i] for i in index]
    
def extract_doc_id(file: str|list) -> int:
    '''
    Extracts the numeric part of a file name and returns it as an integer.
    Example: "00153.txt" -> 153
    '''
    if isinstance(file, str):
        match = re.search(r'\d+', file)
        if match:
            return int(match.group())
        else:
            raise ValueError("No numeric part found in the file name")
    elif isinstance(file, list):
        return [extract_doc_id(f) for f in file]
 
def extract_relevant(file_index:int):
    '''
    Returns the relevant documents for a specific query as a list of integers
    '''
    with open(os.path.join(os.path.dirname(__file__), '../data/Relevant_20.txt'), 'r') as f:
        lines = f.readlines()
        return [int(x) for x in lines[file_index - 1].split()[0:]]
    
def precision_at_k(pred_rel:list, true_rel:list, k:int)-> float:
    '''
    Calculates the precision at k
    '''
    relevant = 0
    for prediction in pred_rel[:k]:
        if prediction in true_rel:
            relevant = relevant + 1
    return relevant/k

def reciproral_ranking(pred_relev:list, true_relev:list)-> float:
    for i, prediction in enumerate(pred_relev):
        if prediction in true_relev:
            return 1/(i+1)
    return 0

def recall(pred_rel:list, true_rel:list)-> float:
    relevant = 0
    for prediction in pred_rel:
        if prediction in true_rel:
            relevant = relevant + 1
    return relevant/len(true_rel)

def precision(pred_rel:list, true_rel:list)-> float:
    relevant = 0
    for prediction in pred_rel:
        if prediction in true_rel:
            relevant = relevant + 1
    return relevant/len(pred_rel)

def harmonic_mean(precision:float, recall:float)-> float:
    if precision + recall == 0:
        return 0
    return 2*(precision*recall)/(precision + recall)

if __name__ == '__main__':
    colbert_metrics()