import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Constants
lambda_param = 0.7

# Paths
document_path = 'C:/Users/samin/Desktop/IFN647/Assignment 2/Data_Collection-1/Data_Collection/'
query_file_path = 'C:/Users/samin/Desktop/IFN647/Assignment 2/the50Queries.txt'
output_path = 'C:/Users/samin/Desktop/IFN647/Assignment 2/My Code/Outputs-Task2/'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# def process_text(text):
#     tokens = word_tokenize(text.lower())
#     filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
#     stemmer = PorterStemmer()
#     stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
#     return stemmed_tokens

stop_words = set(stopwords.words('english'))
def process_text(text, stop_words):
    """
    Returns:
    list: A list of stemmed tokens.
    """
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lower case.
    stemmer = PorterStemmer()  # Initialize the PorterStemmer.
    # Filter out stopwords and stem the remaining words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]
    return stemmed_tokens

def load_documents(directory_path):
    #save documents in a dictionary
    documents = {}
    corpus_length = 0
    document_lengths = {}  # Dictionary to store individual document lengths
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            text = file.read().strip()
            tokens = process_text(text)
            documents[filename] = tokens
            doc_length = len(tokens)  # Length of this particular document
            document_lengths[filename] = doc_length  # Store individual document length
            corpus_length += doc_length
    return documents, document_lengths, corpus_length


def build_corpus_frequency(documents):
    return Counter(token for tokens in documents.values() for token in tokens)

# def calculate_jm_scores(queries, documents, corpus_frequency, corpus_length):
#     scores = defaultdict(dict)
#     for query_id, query in queries.items():
#         for doc_id, doc_tokens in documents.items():
#             doc_length = len(doc_tokens)
#             score = 0
#             for term in query:
#                 doc_term_freq = doc_tokens.count(term)
#                 corpus_term_freq = corpus_frequency[term]
#                 p_td = (1 - lambda_param) * (doc_term_freq / doc_length) if doc_length > 0 else 0
#                 p_tc = lambda_param * (corpus_term_freq / corpus_length) if corpus_length > 0 else 0
#                 term_score = p_td + p_tc
#                 if term_score > 0:
#                     score += math.log(term_score)
#             scores[query_id][doc_id] = score
#     return scores

def calculate_jm_scores(queries, documents, document_lengths, corpus_frequency, corpus_length):
    scores = defaultdict(dict)
    for query_id, query in queries.items():
        for doc_id, doc_tokens in documents.items():
            doc_length = document_lengths[doc_id]  # Use the stored document length
            score = 0
            for term in query:
                doc_term_freq = doc_tokens.count(term)
                corpus_term_freq = corpus_frequency[term]
                p_td = lambda_param * (doc_term_freq / doc_length if doc_length > 0 else 0)
                p_tc = (1 - lambda_param) * (corpus_term_freq / corpus_length if corpus_length > 0 else 0)
                term_score = p_td + p_tc
                if term_score > 0:
                    score += math.log(term_score)
                else:
                    score += math.log(1e-10)  # Avoid taking log of zero
            scores[query_id][doc_id] = score
    return scores



def load_queries(query_file_path):
    """
        Load and process queries from a structured text file, including title, description, and narrative.

        Args:
        query_file_path (str): Path to the query file.
        stop_words (set): A set of stopwords to remove.

        Returns:
        dict: A dictionary containing query ids and their processed tokens.
        """
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        query_blocks = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for block in query_blocks:
            number = re.search(r'<num> Number: (R\d+)', block).group(1)
            title = re.search(r'<title>(.*?)\n', block).group(1).strip()
            desc = re.search(r'<desc> Description:(.*?)\n', block, re.DOTALL).group(1).strip()
            narr = re.search(r'<narr> Narrative:(.*?)\n', block, re.DOTALL).group(1).strip()
            full_query = f"{title} {desc} {narr}"  # Concatenate title, description, and narrative.
            queries[number] = process_text(full_query, stop_words)
    return queries

def save_scores(scores, output_folder):
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"JM_LM_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id}\t{score}\n")

# Main execution flow
queries = load_queries(query_file_path)
all_scores = {}

for i in range(101, 151):  # Assumes the data collections are named as 'Data_C101' to 'Data_C150'
    data_directory = os.path.join(document_path, f"Data_C{i}")
    documents, corpus_len = load_documents(data_directory)
    freq = build_corpus_frequency(documents)
    scores = calculate_jm_scores(queries, documents, freq, corpus_len)
    all_scores.update(scores)

save_scores(all_scores, output_path)
