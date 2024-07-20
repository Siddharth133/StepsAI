# Document Search and QA System

This repository contains the implementation of a robust Document Search and Question Answering (QA) system that utilizes Retrieval-Augmented Generation (RAG) with Tree-Based Hierarchical Indexing Techniques. The system incorporates multiple advanced retrieval methods, including BM25 and Dense Passage Retrieval (DPR), to ensure highly relevant and accurate document retrieval.

## Features

- **Tree-Based Hierarchical Indexing**: Efficiently captures the structure and organization of large textbooks, facilitating precise and context-aware document retrieval.
- **Advanced Retrieval Techniques**: Implements hybrid retrieval methods, including BM25, Dense Passage Retrieval (DPR), and TF-IDF, to ensure high-quality search results.
- **Query Expansion**: Enhances the search process by expanding queries using synonyms and related terms from WordNet, improving retrieval accuracy.
- **Re-ranking Algorithms**: Combines semantic similarity, DPR scores, TF-IDF, and BM25 scores to prioritize and rerank retrieved documents for optimal relevance.
- **Streamlit Interface**: Provides an intuitive and user-friendly interface for inputting queries, viewing search results, and accessing detailed document content.
- **Expandable Context Sections**: Allows users to expand and view full sections of retrieved documents, enhancing the user experience and providing comprehensive context.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/document-search-qa-system.git
    cd document-search-qa-system
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
streamlit run app.py

- Enter your query in the input box.
- View the top retrieved documents and their contexts.
- Expand sections to view full document content.
```
# Project Structure

```bash
document-search-qa-system/
├── app.py                # Main Streamlit application
├── index.py              # Indexing and search functionalities
├── generator.py          # Query expansion and answer generation
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── index.pkl             # Pre-built index file

```

## Future Improvements

- **Enhance Answer Generation:** Improve the LLM's accuracy and relevance by fine-tuning with domain-specific data and better prompt engineering.
- **Optimize Performance:** Further optimize indexing and retrieval algorithms for increased speed and scalability.
- **Expand Query Techniques:** Implement additional query expansion techniques to improve retrieval accuracy.
- **User Interface Enhancements:** Add more interactive features to the Streamlit interface for a better user experience.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
