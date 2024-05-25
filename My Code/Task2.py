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

# stop_words = set(stopwords.words('english'))
# #had to add this because these are useless to the processing
# custom_stopwords = {'xml', 'newsitem', 'root', 'en', 'titl'}  # Add more as needed
# stop_words.update(custom_stopwords)
#

def load_stop_words(file_path):
    # Start with the default English stop words from NLTK
    stop_words = set(stopwords.words('english'))

    # Open the file and read stop words from it
    with open(file_path, 'r', encoding='utf-8') as file:
        # Since all words are on one line, read the single line
        line = file.readline()
        # Split the line into words based on commas and strip whitespace
        custom_stop_words = line.strip().split(',')
        # Add each word from the file to the stop words set
        stop_words.update(word.strip() for word in custom_stop_words)
    # #had to add this because these are useless to the processing

    custom_stopwords = {'xml', 'newsitem', 'root', 'en', 'titl'}  # Add more as needed
    stop_words.update(custom_stopwords)

    return stop_words

# Example usage
file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt'  # Replace with the actual file path
stop_words = load_stop_words(file_path)


def process_text(text, stop_words):
    """
    Returns:
    list: A list of stemmed tokens.
    """
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lower case.
    stemmer = PorterStemmer()  # Initialize the PorterStemmer.
    # Filter out stopwords and stem the remaining words
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalnum()]
    #print(stemmed_tokens)

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

# def calculate_jm_scores(queries, documents, document_lengths, corpus_frequency, corpus_length):
#     scores = defaultdict(dict)
#     for query_id, query in queries.items():
#         for doc_id, doc_tokens in documents.items():
#             doc_length = document_lengths[doc_id]  # Use the stored document length
#             score = 0
#             for term in query:
#                 doc_term_freq = doc_tokens.count(term)
#                 corpus_term_freq = corpus_frequency[term]
#                 p_td = (1- lambda_param) * (doc_term_freq / doc_length if doc_length > 0 else 0)
#                 p_tc = lambda_param * (corpus_term_freq / corpus_length if corpus_length > 0 else 0)
#                 term_score = p_td + p_tc
#                 if term_score > 0:
#                     score += math.log(term_score)
#                 else:
#                     score += math.log(1e-10)  # Avoid taking log of zero
#             scores[query_id][doc_id] = score
#     return scores

def calculate_jm_scores(queries, documents, corpus_frequency, corpus_length):
    lambda_param = 0.4
    scores = defaultdict(dict)
    for query_id, query in queries.items():
        for doc_id, doc_tokens in documents.items():
            doc_length = len(doc_tokens)
            score = 0
            for term in query:
                doc_term_freq = doc_tokens.count(term)
                corpus_term_freq = corpus_frequency[term]
                p_td = (1 - lambda_param) * (doc_term_freq / doc_length) if doc_length > 0 else 0
                p_tc = lambda_param * (corpus_term_freq / corpus_length) if corpus_length > 0 else 0
                term_score = p_td + p_tc
                if term_score > 0:
                    score += math.log(term_score)
            scores[query_id][doc_id] = score
    return scores



# def load_queries(query_file_path):
#     """
#         Load and process queries from a structured text file, including title, description, and narrative.
#
#         Args:
#         query_file_path (str): Path to the query file.
#         stop_words (set): A set of stopwords to remove.
#
#         Returns:
#         dict: A dictionary containing query ids and their processed tokens.
#         """
#     queries = {}
#     with open(query_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#         raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
#         for raw_query in raw_queries:
#             number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
#             title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
#             narrative = re.search(r'<narr>(.*?)\n', raw_query).group(1).strip()
#             description = re.search(r'<desc>(.*?)\n', raw_query).group(1).strip()
#             # queries[number] = process_text(title + narrative + description)
#             full_query = f"{title} {description} {narrative}"  # Concatenate title, description, and narrative.
#             print(full_query)
#             queries[number] = process_text(full_query, stop_words)
#     return queries

# def load_queries(query_file_path):
#     """
#     Load and process queries from a structured text file, including title, description, and narrative.
#
#     Args:
#     query_file_path (str): Path to the query file.
#     stop_words (set): A set of stopwords to remove.
#
#     Returns:
#     dict: A dictionary containing query ids and their processed tokens.
#     """
#     queries = {}
#     with open(query_file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#         # Updated regex to capture multiline text until the next '<Query>' or end of the content
#         raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
#         for raw_query in raw_queries:
#             number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
#             title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
#             print("TITLE", title)
#             description = re.search(r'<desc>(.*?)\n', raw_query).group(1).strip()
#             print("DESC", description)
#             narrative = re.search(r'<narr>(.*?)\n', raw_query).group(1).strip()
#             print("NARR", narrative)
#             full_query = f"{title} {description} {narrative}"  # Concatenate title, description, and narrative.
#             stop_words = load_stop_words("C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\common-english-words.txt")
#             queries[number] = process_text(full_query, stop_words)
#             print(full_query)
#     return queries

def load_queries(query_file_path, stop_words):
    # queries = {}
    # with open(query_file_path, 'r', encoding='utf-8') as file:
    #     content = file.read()
    #     raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
    #     for raw_query in raw_queries:
    #         number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
    #         title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
    #         print("TITLE: ", title)
    #         description = re.search(r'<desc> Description:\s*(.*?)(?=\n<narr>|</Query>)', raw_query, re.DOTALL)
    #         narrative = re.search(r'<narr> Narrative:\s*(.*?)(?=\n</Query>)', raw_query, re.DOTALL)
    #
    #         # Check if description and narrative are found, and handle the case where they might not be present
    #         if description:
    #             description = description.group(1).strip()
    #         else:
    #             description = ""
    #         print("DESC: ", description)
    #
    #         if narrative:
    #             narrative = narrative.group(1).strip()
    #         else:
    #             narrative = ""
    #         print("NARR: ", narrative)
    #
    #         full_query = f"{title} {description} {narrative}"  # Concatenate title, description, and narrative.
    #         print("full query: ", full_query)
    #         queries[number] = process_text(full_query, stop_words)
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        #print(raw_queries)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            description_search = re.search(r'<desc> Description:\s*(.*?)(?=\n<narr>|</Query>)', raw_query, re.DOTALL)
            narrative_search = re.search(r'<narr> Narrative:\s*(.*?)\n\n', raw_query, re.DOTALL)
            #narrative_search = re.search(r'<narr>\s*Narrative:\s*(.*?)\s(?=</Query>)', raw_query, re.DOTALL)


            description = description_search.group(1).strip() if description_search else ""
            narrative = narrative_search.group(1).strip() if narrative_search else ""

            full_query = f"{title} {description} {narrative}"
            queries[number] = process_text(full_query, stop_words)

            # Print statements for debugging
            print(f"NUMBER: {number}")
            print(f"TITLE: {title}")
            print(f"DESCRIPTION: {description}")
            print(f"NARRATIVE: {narrative}")
            print(f"FULL QUERY: {full_query}")
        return queries


def save_scores(scores, output_folder):
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"JM_LM_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id}\t{score}\n")

# Main execution flow
queries = load_queries(query_file_path, stop_words)
all_scores = {}

for i in range(101, 151):  # Assumes the data collections are named as 'Data_C101' to 'Data_C150'
    data_directory = os.path.join(document_path, f"Data_C{i}")
    documents, document_lengths, corpus_length = load_documents(data_directory)
    freq = build_corpus_frequency(documents)
    #scores = calculate_jm_scores(queries, documents, document_lengths, freq, corpus_length)
    scores = calculate_jm_scores(queries, documents, freq, corpus_length)
    all_scores.update(scores)

save_scores(all_scores, output_path)