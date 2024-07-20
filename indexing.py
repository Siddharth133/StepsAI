"""
This Code create the index of the content from the json files and save the index 
in the pickle file.

This Code need to run only once to create the index and save it in the pickle file.

When the user add any other new document in the json file we will list that file down 
in the json_files list and run the code to generate the index.
"""


import json
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import torch
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('punkt')

class TreeNode:
    def __init__(self, title, page=None, content=None):
        self.title = title
        self.page = page
        self.content = content
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode(title={self.title}, page={self.page})"

def deserialize_tree(data):
    if data is None:
        return None
    node = TreeNode(
        title=data.get('title'),
        page=data.get('page'),
        content=data.get('content')
    )
    for child_data in data.get('sections', []):
        child_node = deserialize_tree(child_data)
        if child_node:
            node.add_child(child_node)
    for child_data in data.get('parts', []):
        child_node = deserialize_tree(child_data)
        if child_node:
            node.add_child(child_node)
    return node

def load_tree_from_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return deserialize_tree(data)

def expand_query(query):
    expanded_query = query.split()
    for word in nltk.word_tokenize(query):
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() not in expanded_query:
                    expanded_query.append(lemma.name())
    return ' '.join(expanded_query)

class Index:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dpr_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
        self.vectorizer = TfidfVectorizer()
        self.corpus = []
        self.bm25 = None
        self.index = {}
        self.embeddings = {}
        self.dpr_embeddings = {}

    def add_to_index(self, node, source_name):
        if node is None:
            return
        key = f"{source_name}/{node.title}"
        content = node.content or ""

        # Compute embeddings
        embedding = self.model.encode(content, convert_to_tensor=True).tolist()
        dpr_embedding = self.dpr_encoder.encode(content, convert_to_tensor=True).tolist()

        # Store in the index
        self.index[key] = {
            'title': node.title,
            'page': node.page,
            'content': content,
            'source': source_name
        }
        self.embeddings[key] = embedding
        self.dpr_embeddings[key] = dpr_embedding
        self.corpus.append(content)

        for child in node.children:
            self.add_to_index(child, source_name)

    def finalize_index(self):
        self.vectorizer.fit(self.corpus)
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query):
        expanded_query = expand_query(query)
        query_embedding = self.model.encode(expanded_query, convert_to_tensor=True).tolist()
        query_dpr_embedding = self.dpr_encoder.encode(expanded_query, convert_to_tensor=True).tolist()

        results = []
        for key, content_embedding in self.embeddings.items():
            similarity = util.pytorch_cos_sim(query_embedding, content_embedding).item()
            dpr_similarity = util.pytorch_cos_sim(query_dpr_embedding, self.dpr_embeddings[key]).item()
            if similarity > 0.3 or dpr_similarity > 0.3:
                results.append((key, similarity, dpr_similarity))

        reranked_results = self.rerank(expanded_query, results)
        return [self.index[key] for key, _ in reranked_results]

    def rerank(self, expanded_query, results):
        query_vec = self.vectorizer.transform([expanded_query])
        query_tokens = expanded_query.split()

        reranked = []
        for key, semantic_score, dpr_score in results:
            content = self.index[key]['content']
            tfidf_score = query_vec.dot(self.vectorizer.transform([content]).T)[0, 0]
            bm25_score = self.bm25.get_scores(query_tokens)[self.corpus.index(content)]

            combined_score = 0.3 * semantic_score + 0.3 * dpr_score + 0.2 * tfidf_score + 0.2 * bm25_score
            reranked.append((key, combined_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def save_index(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.index, f)
            pickle.dump(self.embeddings, f)
            pickle.dump(self.dpr_embeddings, f)
            pickle.dump(self.corpus, f)

    def load_index(self, filename):
        with open(filename, 'rb') as f:
            self.index = pickle.load(f)
            self.embeddings = pickle.load(f)
            self.dpr_embeddings = pickle.load(f)
            self.corpus = pickle.load(f)
            self.finalize_index()

# Example usage
json_files = ['final_json/Final_ML.json', 'final_json/Final_NLTK.json']  
index = Index()
for json_file in json_files:
    print(json_file)
    tree = load_tree_from_file(json_file)
    index.add_to_index(tree, source_name=json_file)

index.finalize_index()
index.save_index('index.pkl')

# Query can be asked as below example to get the relative content from all the files

query_results = index.search("Explain N-Gram Tagging")
for result in query_results:
    print(f"Title: {result['title']}")
    print(f"Page: {result['page']}")
    print(f"Source: {result['source']}")
    print(f"Content: {result['content'][:200]}...")
    print()
