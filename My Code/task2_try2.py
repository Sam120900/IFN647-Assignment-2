import os
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')


def process_text(text):
    """ Tokenize text, remove stopwords, and apply stemming. """
    tokens = word_tokenize(text.lower())
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return [stemmer.stem(word) for word in tokens if word not in stop_words]


def load_queries(query_file_path):
    """ Load and parse queries from a text file. """
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        matches = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for match in matches:
            num = re.search(r'<num> Number: (R\d+)', match).group(1)
            title = re.search(r'<title>(.*?)\n', match).group(1).strip()
            desc = re.search(r'<desc> Description:(.*?)\n', match, re.DOTALL).group(1).strip()
            narr = re.search(r'<narr> Narrative:(.*?)\n', match, re.DOTALL).group(1).strip()
            full_query = ' '.join([title, desc, narr])
            queries[num] = process_text(full_query)
    return queries


def load_documents(folder_path):
    """ Load and process all XML documents within a folder. """
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            path = os.path.join(folder_path, filename)
            tree = ET.parse(path)
            root = tree.getroot()
            # Assuming text is directly under <root><text></text></root>
            text = "".join(root.itertext())
            documents[filename[:-4]] = process_text(text)
    return documents


def compute_jm_lm(queries, data_folder_path, output_folder):
    """ Compute Jelinek-Mercer Language Model scores and output rankings. """
    lambda_val = 0.4  # Smoothing parameter
    for query_id, query_terms in queries.items():
        documents = load_documents(os.path.join(data_folder_path, f"Data_C{query_id[1:]}"))
        total_words_in_collection = sum(len(doc) for doc in documents.values())
        collection_freq = Counter(word for doc in documents.values() for word in doc)

        results = {}
        for doc_id, doc_terms in documents.items():
            doc_length = len(doc_terms)
            doc_freq = Counter(doc_terms)
            score = 0
            for term in query_terms:
                if term in collection_freq:
                    f_qi_D = doc_freq[term]
                    f_qi_C = collection_freq[term]
                    prob_d = lambda_val * (f_qi_D / doc_length if doc_length > 0 else 0)
                    prob_c = (1 - lambda_val) * (
                        f_qi_C / total_words_in_collection if total_words_in_collection > 0 else 0)
                    score += math.log(prob_d + prob_c) if prob_d + prob_c > 0 else float('-inf')
            results[doc_id] = score

        # Save the results
        output_path = os.path.join(output_folder, f"{query_id}_JM_LM_ranking.dat")
        with open(output_path, 'w') as file:
            for doc_id, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
                file.write(f"{doc_id} {score}\n")


# Example usage
query_file_path = 'path_to_queries.txt'
base_data_folder = 'base_folder_for_data_collections'
output_folder = 'RankingOutputs'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

queries = load_queries(query_file_path)
compute_jm_lm(queries, base_data_folder, output_folder)
