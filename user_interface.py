import streamlit as st
import json
import pickle
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import torch
import os
import nltk
from nltk.corpus import wordnet
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFaceHub
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Download necessary NLTK resources if not already downloaded
nltk.download('wordnet')
nltk.download('punkt')

class Index:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dpr_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
        self.vectorizer = None
        self.corpus = []
        self.bm25 = None
        self.index = {}
        self.embeddings = {}
        self.dpr_embeddings = {}

    def load_index(self, filename):
        with open(filename, 'rb') as f:
            self.index = pickle.load(f)
            self.embeddings = pickle.load(f)
            self.dpr_embeddings = pickle.load(f)
            self.corpus = pickle.load(f)
            self.finalize_index()

    def finalize_index(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.corpus)
        tokenized_corpus = [doc.split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query):
        expanded_query = self.expand_query(query)
        query_embedding = self.model.encode(expanded_query, convert_to_tensor=True).tolist()
        query_dpr_embedding = self.dpr_encoder.encode(expanded_query, convert_to_tensor=True).tolist()

        results = []
        for key, content_embedding in self.embeddings.items():
            similarity = util.pytorch_cos_sim(query_embedding, content_embedding).item()
            dpr_similarity = util.pytorch_cos_sim(query_dpr_embedding, self.dpr_embeddings[key]).item()
            if similarity > 0.5 or dpr_similarity > 0.5:
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

    def expand_query(self, query):
        expanded_query = query.split()
        for word in nltk.word_tokenize(query):
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() not in expanded_query:
                        expanded_query.append(lemma.name())
        return ' '.join(expanded_query)

# Load the index from the Pickle file
index = Index()
index.load_index('index.pkl')

# Initialize language model from Hugging Face Hub for query expansion
repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
llm = HuggingFaceHub(
    repo_id=repo_id,
    model_kwargs={"temperature": 0.7, "top_k": 50, "max_length": 1000, "max_new_tokens": 300},
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY")
)

# Template for query expansion
template = """
Use the context provided to generate an accurate answer for the query.
You are a knowledgeable assistant. Given the context below, answer the question concisely and accurately.
The answer should be in the form of complete sentences and provide a clear explanation.
Use External Resources if necessary.
Answer Provide should involve the main word query defination, use and example.
Original Query: {question}
Context : {context}
Answer :
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question","context"]
)

pipeline = (
    RunnablePassthrough()
    | prompt
    | llm
    | StrOutputParser()
)

class Generator:
    def generate(self, question, context):
        result = pipeline.invoke({"question": question, "context": context})
        return result.strip().split('Answer :', 1)[-1].strip()

# Streamlit interface
st.title("Document Search and QA System")

query = st.text_input("Enter your query:")

if query:
    with st.spinner("Searching and generating answer..."):
        query_results = index.search(query)
        
        if query_results:
            top_results = query_results[:3]
            context = " ".join([result['content'] for result in top_results])
            generator = Generator()
            answer = generator.generate(query, context)
            
            st.subheader("Generated Answer")
            st.write(answer)
            
            top_results = query_results[:10]
            st.subheader("Top Retrieved Documents")
            for result in top_results:
                with st.expander(f"**Title:** {result['title']} - **Page:** {result['page']}"):
                    st.markdown(f"**Source:** {result['source']}")
                    st.markdown(result['content'])
                st.markdown("---")
        else:
            st.error("Sorry, cannot find any information about this query.")
