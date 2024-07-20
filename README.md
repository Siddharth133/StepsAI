## Steps AI assessment

This repository contains the solution to the StepsAI assessment which has objective of implementating a robust Document Search and Question Answering (QA) system that utilizes Retrieval-Augmented Generation (RAG) with Tree-Based Hierarchical Indexing Techniques. The system incorporates multiple advanced retrieval methods, including BM25 and Dense Passage Retrieval (DPR), to ensure highly relevant and accurate document retrieval.

## Approach Highlights
- Utilized Tree-Based Hierarchical Indexing and Retrieval-Augmented Generation (RAG).
- Integrated BM25 and Dense Passage Retrieval (DPR) for improved results.
- Focused on efficient document retrieval and context preservation.
- Retained document structure and context.
- Planned for query expansion and re-ranking.

## Features
- **Tree-Based Hierarchical Indexing**: Efficiently captures the structure and organization of large textbooks, facilitating precise and context-aware document retrieval.
- **Advanced Retrieval Techniques**: Implements hybrid retrieval methods, including BM25, Dense Passage Retrieval (DPR), and TF-IDF, to ensure high-quality search results.
- **Query Expansion**: Enhances the search process by expanding queries using synonyms and related terms from WordNet, improving retrieval accuracy.
- **Re-ranking Algorithms**: Combines semantic similarity, DPR scores, TF-IDF, and BM25 scores to prioritize and rerank retrieved documents for optimal relevance.
- **Streamlit Interface**: Provides an intuitive and user-friendly interface for inputting queries, viewing search results, and accessing detailed document content.
- **Expandable Context Sections**: Allows users to expand and view full sections of retrieved documents, enhancing the user experience and providing comprehensive context.

## Initial Approach
- Explored direct chunking of text extracted from the textbook, which proved ineffective for retrieving relevant content.
- Considered using regex to extract the Table of Contents (ToC) and store each section in a JSON file, maintaining the tree structure.
- Developed a two-step process: first extracting the ToC, then filling it with content.
- Created a JSON format ToC maintaining parent-child relationships.
- Matched section titles and page numbers from the textbook text to accurately fill the sections and maintain the hierarchical structure.
- Investigated real-time indexing, which proved to be time-consuming.

## Final Implementation
- Implemented a local storage solution using pickle files for efficient index storage and retrieval.
- Utilized sentence transformers ('all-MiniLM-L6-v2' and 'facebook-dpr-question_encoder-single-nq-base') for semantic search and DPR.
- Incorporated BM25 and TF-IDF for additional relevance scoring.
- Developed a query expansion technique using WordNet for improved search accuracy.
- Implemented a re-ranking algorithm combining semantic similarity, DPR scores, TF-IDF, and BM25 scores.
- Integrated a language model (Mistral-7B-Instruct-v0.1) for advanced query understanding and answer generation.
- Created a Streamlit interface for user-friendly interaction with the system.
- Optimized the search and retrieval process for quick inference time.

## Solution Development
- **Initial Thoughts:** Focused on direct text chunking and regex-based ToC extraction.
- **Intermediate Steps:**
  - Developed a two-step ToC extraction and content filling process.
  - Explored real-time indexing before opting for pre-built index storage.
- **Final Approach:**
  - Incorporated a multi-faceted retrieval system using BM25, DPR, TF-IDF, and semantic search.
  - Implemented dynamic query expansion using WordNet and a language model.
  - Developed a sophisticated re-ranking algorithm to improve result relevance.
  - Created a hierarchical indexing structure to maintain document context and organization.
  - Integrated a Streamlit interface for enhanced user experience and result presentation.

## Results
The final implementation achieved efficient and accurate document retrieval within a short timeframe, with enhanced context preservation and user experience.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#Results)
- [Solution/Approach](#Solution/Approach)
- [Future Improvements](#future-improvements)
- [License](#license)

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Siddharth133/StepsAI.git
    cd StepsAI
    ```

2. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download NLTK Resources**:

    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')
    ```

4. **Set Up Environment Variables**:

    Create a `.env` file in the root directory and add your Hugging Face API key:

    ```makefile
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    ```

5. **Load the Index**:

    Ensure the `index.pkl` file is in the project directory.

## Usage

To start the Streamlit application, run the following command:

```bash
streamlit run user_interface.py

- Enter your query in the input box.
- View the top retrieved documents and their contexts.
- Expand sections to view full document content.
```
# Project Structure

```bash
StepsAI/
├── user_interface.py     # Main Streamlit application
├── index.py              # Indexing and search functionalities
├── generator.py          # Query expansion and answer generation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── index.pkl             # Pre-built index file

```
# Results


![initial](https://github.com/user-attachments/assets/289ac113-ea1d-4f95-891b-89411a166cca)


![Retrievd Doc](https://github.com/user-attachments/assets/e13e50ba-06ae-4ab4-accf-ad3e51122bcd)


![Expander](https://github.com/user-attachments/assets/6d81055e-f43c-4f11-9e23-ad1feddce02b)



# Solution/Approach

I will be discussing the chain of thought I had on the problem statement when I first read it, what I was able to implement, and how much I achieved in this short span of time.

### Initial Thoughts (You may skip the initial thoughts to see the final approach)

1. **Text Extraction**: The first step was to extract text from the textbook. I initially considered direct chunking of the extracted text, but this proved ineffective for retrieving relevant content.

2. **Table of Contents Extraction**: I thought about using regex to extract the Table of Contents (ToC) and store each section in a JSON file, maintaining the tree structure. This approach needed refinement.

3. **Hierarchical Structure**: To preserve the book's structure, I developed a two-step process: first extracting the ToC, then filling it with content. This maintained the parent-child relationships in a JSON format.

4. **Content Matching**: I implemented a method to match section titles and page numbers from the textbook text to accurately fill the sections and maintain the hierarchical structure.

5. **Indexing Techniques**: I explored various indexing techniques, initially considering real-time indexing, which proved to be time-consuming.

6. **Retrieval Models**: Researched different retrieval models including BM25, Dense Passage Retrieval (DPR), and TF-IDF for optimal performance.

7. **Query Expansion**: Considered techniques for query expansion to improve search accuracy.

### Final Approach Before Submission

1. **Efficient Storage**: Implemented a local storage solution using pickle files for efficient index storage and retrieval, avoiding the time-consuming real-time indexing.

2. **Advanced Embedding Models**: Utilized sentence transformers ('all-MiniLM-L6-v2' and 'facebook-dpr-question_encoder-single-nq-base') for semantic search and DPR, enhancing the understanding of both documents and queries.

3. **Multi-faceted Retrieval**: Incorporated BM25 and TF-IDF alongside the embedding models for a more comprehensive relevance scoring system.

4. **Query Enhancement**: Developed a query expansion technique using WordNet to improve search accuracy by including synonyms and related terms.

5. **Sophisticated Re-ranking**: Implemented a re-ranking algorithm that combines semantic similarity, DPR scores, TF-IDF, and BM25 scores, significantly improving the relevance of retrieved documents.

6. **Advanced Language Model Integration**: Integrated the Mistral-7B-Instruct-v0.1 language model for advanced query understanding and high-quality answer generation.

7. **User Interface**: Created a Streamlit interface for user-friendly interaction with the system, allowing easy input of queries and clear presentation of results.

8. **Performance Optimization**: Focused on optimizing the search and retrieval process for quick inference time with the help of Tree-Strutured Index, ensuring responses within a reasonable timeframe.

9. **Expandable Results**: Implemented an expandable view for retrieved documents, allowing users to see full context when needed.

10. **Testing and Refinement**: Conducted extensive testing with various queries to refine the system's performance and accuracy.

The final implementation achieved efficient and accurate document retrieval and question answering, with enhanced context preservation and user experience. The system demonstrates the ability to understand complex queries, retrieve relevant information from a large textbook, and generate coherent answers, all while maintaining the structural integrity of the source material.


# Future Improvements

- **Enhance Answer Generation:** Improve the LLM's accuracy and relevance by fine-tuning with domain-specific data and better prompt engineering.
- **Optimize Performance:** Further optimize indexing and retrieval algorithms for increased speed and scalability.
- **Expand Query Techniques:** Implement additional query expansion techniques to improve retrieval accuracy.
- **User Interface Enhancements:** Add more interactive features to the Streamlit interface for a better user experience.
-  **Mixture of Experts(MoE) Model:** Implement a mixture of experts model to dynamically select the best retrieval and answering strategies, enhancing flexibility, accuracy, and performance.
-  **Real-Time Document Processing:** Implement functionality for users to upload their own documents in real-time. This feature would dynamically process, index, and integrate new documents into the existing knowledge base, allowing the system to answer queries from newly uploaded content without pre-processing or system redeployment. This enhancement would significantly increase the platform's flexibility and usefulness across various domains and use cases.

# License

This project is licensed under the MIT License. See the LICENSE file for more details.
