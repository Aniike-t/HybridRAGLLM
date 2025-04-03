# Hybrid RAG LLM - System Architecture

This repository implements a **Hybrid Retrieval-Augmented Generation (RAG) System** for processing and retrieving information from PDF documents. Below is the architecture of the system:

## Integrating Traditional IR with Neural Retrieval for Enhanced Question Answering in Large Documents

1. Install all the required libraries from requirements.txt
2. Relocate pdf in data/
3. Gemini key in `config.py`


## System Overview

```mermaid
flowchart TD
    %% External Inputs
    PDF["Raw PDF Documents"]:::external
    Query["User Query"]:::external

    %% Data Processing Layer
    subgraph "Data Processing"
        DP1["pdf_extractor"]:::processing
        DP2["Chunking"]:::processing
    end

    %% Indexing Layer
    subgraph "Indexing"
        IDX1["Inverted Index"]:::indexing
        IDX2["Embedding Index"]:::indexing
        IDX3["Signature Index"]:::indexing
    end

    %% Retrieval Layer
    subgraph "Retrieval"
        RET1["Query Processor"]:::retrieval
        RET2["Hybrid Retriever"]:::retrieval
        RET3["Prompt Builder"]:::retrieval
    end

    %% Configuration & Utilities
    subgraph "Configuration & Utilities"
        CFG["Configuration Settings"]:::config
        UT["Helper Functions"]:::utilities
    end

    %% Main Application Flow
    MAIN["Main Application Flow"]:::main

    %% Data Flow Connections
    PDF -->|"ExtractAndChunk"| DP1
    DP1 -->|"ProcessText"| DP2
    DP2 -->|"FeedData"| IDX1
    DP2 -->|"FeedData"| IDX2
    DP2 -->|"FeedData"| IDX3

    Query -->|"RouteQuery"| RET1
    RET1 -->|"DelegateToHybrid"| RET2
    IDX1 -->|"IndexResults"| RET2
    IDX2 -->|"IndexResults"| RET2
    IDX3 -->|"IndexResults"| RET2
    RET2 -->|"BuildFinalPrompt"| RET3
    RET3 -->|"OutputPrompt"| MAIN

    MAIN -->|"InitializeWith"| CFG
    MAIN -->|"Utilize"| UT
    MAIN -->|"StartProcessing"| DP1

    %% Click Events
    click DP1 "https://github.com/aniike-t/hybridragllm/blob/master/data_processing/pdf_extractor.py"
    click DP2 "https://github.com/aniike-t/hybridragllm/blob/master/data_processing/chunking.py"
    click IDX1 "https://github.com/aniike-t/hybridragllm/blob/master/indexing/inverted_index.py"
    click IDX2 "https://github.com/aniike-t/hybridragllm/blob/master/indexing/embedding_index.py"
    click IDX3 "https://github.com/aniike-t/hybridragllm/blob/master/indexing/signature_index.py"
    click RET1 "https://github.com/aniike-t/hybridragllm/blob/master/retrieval/query_processor.py"
    click RET2 "https://github.com/aniike-t/hybridragllm/blob/master/retrieval/hybrid_retriever.py"
    click RET3 "https://github.com/aniike-t/hybridragllm/blob/master/retrieval/prompt_builder.py"
    click CFG "https://github.com/aniike-t/hybridragllm/blob/master/config.py"
    click UT "https://github.com/aniike-t/hybridragllm/blob/master/utils/helpers.py"
    click MAIN "https://github.com/aniike-t/hybridragllm/blob/master/main.py"

    %% Style Definitions
    classDef external fill:#fef3bd,stroke:#e3a21a,stroke-width:2px;
    classDef processing fill:#cce5ff,stroke:#004085,stroke-width:2px;
    classDef indexing fill:#d4edda,stroke:#155724,stroke-width:2px;
    classDef retrieval fill:#f8d7da,stroke:#721c24,stroke-width:2px;
    classDef config fill:#fff3cd,stroke:#856404,stroke-width:2px;
    classDef utilities fill:#d1ecf1,stroke:#0c5460,stroke-width:2px;
    classDef main fill:#e2e3e5,stroke:#6c757d,stroke-width:2px;
```

## Features
- **PDF Extraction**: Parses raw PDFs into extractable text data.
- **Text Chunking**: Splits extracted text into manageable chunks.
- **Multiple Indexing Strategies**: Uses **Inverted Index, Embedding Index, and Signature Index** for efficient retrieval.
- **Hybrid Retrieval**: Combines different search strategies for enhanced accuracy.
- **Query Processing**: Routes and processes user queries efficiently.
- **Prompt Generation**: Builds well-structured prompts for LLM interaction.
- **Configurable & Extensible**: Modular design with configurable settings and utilities.

## Installation
```sh
# Clone the repository
git clone https://github.com/aniike-t/hybridragllm.git
cd hybridragllm

# Install dependencies
pip install -r requirements.txt
```

## Usage
```sh
python main.py
```

## Directory Structure
```
├── data_processing/
│   ├── pdf_extractor.py
│   ├── chunking.py
├── indexing/
│   ├── inverted_index.py
│   ├── embedding_index.py
│   ├── signature_index.py
├── retrieval/
│   ├── query_processor.py
│   ├── hybrid_retriever.py
│   ├── prompt_builder.py
├── utils/
│   ├── helpers.py
├── config.py
├── main.py
└── README.md
```

## Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License.

