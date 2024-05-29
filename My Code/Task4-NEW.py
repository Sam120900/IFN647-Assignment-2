import os
import math
import re
from collections import defaultdict, Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text, stop_words):
    """
    Process the text by tokenizing, converting to lower case, removing stopwords, and stemming.
    """
    tokens = word_tokenize(text.lower())
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]

def load_stop_words(file_path):
    """
    Load stop words from the NLTK library and a custom file.
    """
    stop_words = set(stopwords.words('english'))
    with open(file_path, 'r', encoding='utf-8') as file:
        custom_stop_words = file.readline().strip().split(',')
    stop_words.update(word.strip() for word in custom_stop_words)
    stop_words.update({'xml', 'newsitem', 'root', 'en', 'titl'})
    return stop_words

def load_queries(query_file_path, stop_words):
    """
    Load queries from a file and process them into a dictionary of tokenized texts.
    """
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            description_search = re.search(r'<desc> Description:\s*(.*?)(?=\n<narr>|</Query>)', raw_query, re.DOTALL)
            narrative_search = re.search(r'<narr> Narrative:\s*(.*?)\n\n', raw_query, re.DOTALL)

            description = description_search.group(1).strip() if description_search else ""
            narrative = narrative_search.group(1).strip() if narrative_search else ""

            full_query = f"{title} {description} {narrative}"
            queries[number] = process_text(full_query, stop_words)
    return queries

def load_documents_bm25(directory_path, stop_words):
    """
    Load and process all documents from a specified directory.
    """
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            documents[filename] = process_text(file.read().strip(), stop_words)
    return documents
def load_documents_jmlm(directory_path, stop_words):
    #save documents in a dictionary
    documents = {}
    corpus_length = 0
    document_lengths = {}  # Dictionary to store individual document lengths
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read().strip()
            tokens = process_text(text, stop_words)
            documents[filename] = tokens
            #print("document", documents[filename])
            doc_length = len(tokens)  # Length of this particular document
            #print("doc length", doc_length)
            document_lengths[filename] = doc_length  # Store individual document length
            corpus_length += doc_length
            #print("corpus length", corpus_length)
    return documents, document_lengths, corpus_length

def build_corpus_frequency(documents):
    """
    Build frequency distribution of terms across all documents.
    """
    return Counter(token for tokens in documents.values() for token in tokens)

def calculate_bm25(N, avgdl, documents, queries, df):
    """
    Calculate BM25 scores for each document given a set of queries.
    """
    k1 = 1.2
    k2 = 500
    b = 0.75
    scores = {}
    for query_id, query in queries.items():
        scores[query_id] = {}
        for doc_id, doc in documents.items():
            score = 0
            dl = len(doc)
            for word in set(query):
                if word in doc:
                    n = df.get(word, 0)
                    f = doc.count(word)
                    qf = query.count(word)
                    K = k1 * ((1 - b) + b * dl / avgdl)
                    idf = math.log((N - n + 0.5) / (n + 0.5), 10)
                    term_score = idf * ((f * (k1 + 1)) / (f + K)) * ((qf * (k2 + 1)) / (qf + k2))
                    score += term_score
            scores[query_id][doc_id] = score
    return scores

def calculate_jm_scores(queries, documents, corpus_frequency, corpus_length, lambda_param=0.4):
    """
    Calculate JM smoothing scores for each document against each query.
    """
    scores = defaultdict(dict)
    for query_id, query_tokens in queries.items():
        for doc_id, doc_tokens in documents.items():
            doc_length = len(doc_tokens)
            score = 0
            for term in query_tokens:
                doc_term_freq = doc_tokens.count(term)
                corpus_term_freq = corpus_frequency[term]
                p_td = (1 - lambda_param) * (doc_term_freq / doc_length) if doc_length > 0 else 0
                p_tc = lambda_param * (corpus_term_freq / corpus_length) if corpus_length > 0 else 0
                score += p_td + p_tc
            scores[query_id][doc_id] = score
    return scores

def save_bm25_scores(scores, output_folder, query_id):
    """
    Save the BM25 scores to files, each corresponding to a query.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file_path = os.path.join(output_folder, f"BM25_{query_id}Ranking.dat")
    with open(output_file_path, 'w') as file:
        for doc_id, score in sorted(scores[query_id].items(), key=lambda x: x[1], reverse=True):
            file.write(f"{doc_id} {score}\n")

def save_jmlm_scores(scores, output_folder):
    """
    Save scores to files, one for each query.
    """
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"JM_LM_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id}\t{score}\n")


def main():
    query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
    base_data_directory = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection'
    output_folder = ('RankingOutputs-New')
    file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'

    stop_words = load_stop_words(file_path)
    queries = load_queries(query_file_path, stop_words)

    for query_id, query_tokens in queries.items():
        data_directory = os.path.join(base_data_directory, f"Data_C{query_id[1:]}")
        documents_bm25 = load_documents_bm25(data_directory, stop_words)
        N = len(documents_bm25)
        avgdl = sum(len(doc) for doc in documents_bm25.values()) / N
        df = {word: sum(1 for doc in documents_bm25.values() if word in doc) for doc in documents_bm25.values() for word in set(doc)}
        scores_bm25 = calculate_bm25(N, avgdl, documents_bm25, {query_id: query_tokens}, df)
        save_bm25_scores(scores_bm25, output_folder, query_id)

        documents_jmlm, document_lengths, corpus_length = load_documents_jmlm(data_directory, stop_words)
        freq = build_corpus_frequency(documents_jmlm)
        scores_jmlm = calculate_jm_scores({query_id: queries[query_id]}, documents_jmlm, freq, corpus_length)
        save_jmlm_scores(scores_jmlm, output_folder)

if __name__ == "__main__":
    main()