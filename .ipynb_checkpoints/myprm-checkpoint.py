import os
import math
import re
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# Path setup
data_directory = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/Data_Collection-1/Data_Collection/'
query_file_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/the50Queries.txt'
output_directory = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/00111Outputs-Task3/'

# Parameters
N_top_docs = 5  # Number of top documents to use for query expansion

# Stopwords and stemmer setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def process_text(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]

def load_queries(query_file_path):
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        pattern = re.compile(r'<num> Number: (R\d+).*?<title>(.*?)\n', re.DOTALL)
        for match in pattern.finditer(content):
            query_id, title = match.groups()
            queries[query_id] = process_text(title)
    return queries

def load_documents(data_directory, query_id):
    document_path = os.path.join(data_directory, f"Data_C{query_id[1:]}")  # Adjust query_id if needed
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Directory not found: {document_path}")
    documents = {}
    for filename in os.listdir(document_path):
        with open(os.path.join(document_path, filename), 'r', encoding='utf-8') as file:
            documents[filename] = process_text(file.read())
    return documents

def calculate_tfidf(documents, query):
    tfidf_scores = defaultdict(float)
    doc_count = len(documents)
    doc_freq = Counter(word for doc in documents.values() for word in set(doc))

    for doc_id, content in documents.items():
        term_freq = Counter(content)
        doc_length = len(content)
        for term in query:
            tf = term_freq[term] / doc_length
            idf = math.log(doc_count / (1 + doc_freq[term]) + 1)
            tfidf_scores[doc_id] += tf * idf

    return tfidf_scores

def expand_query(top_documents, documents):
    term_frequency = Counter(word for doc_id in top_documents for word in documents[doc_id])
    most_common_terms = [word for word, freq in term_frequency.most_common(10)]
    return most_common_terms

def main():
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    queries = load_queries(query_file_path)

    for query_id, query_tokens in queries.items():
        documents = load_documents(data_directory, query_id)
        initial_scores = calculate_tfidf(documents, query_tokens)
        top_documents = sorted(initial_scores, key=initial_scores.get, reverse=True)[:N_top_docs]
        expanded_query = expand_query(top_documents, documents)
        final_scores = calculate_tfidf(documents, query_tokens + expanded_query)

        output_path = os.path.join(output_directory, f"My_PRM_{query_id}Ranking.dat")
        with open(output_path, 'w') as file:
            for doc_id in sorted(final_scores, key=final_scores.get, reverse=True):
                file.write(f"{doc_id}\t{final_scores[doc_id]}\n")

if __name__ == "__main__":
    main()
