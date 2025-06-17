# AI Research Assistant Usage Guide

## Overview

The AI Research Assistant is a powerful tool that combines knowledge graphs, semantic search, and AI agents to help researchers analyze academic papers, identify research gaps, and generate hypotheses. The system supports both local and cloud-based AI models, with a focus on privacy and flexibility.

## Prerequisites

- Python 3.8+
- Neo4j Database
- Ollama (for local models)
- OpenAI API key (optional, for cloud models)
- 8GB+ RAM for local models

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Required
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# Optional
export MODEL_PROVIDER="local"  # or "openai"
export OPENAI_API_KEY="your_key"  # if using OpenAI
```

## Starting the Application

Run the application:
```bash
python main.py
```

The web interface will be available at `http://localhost:8000`

## Using the Interface

The application provides four main modes of operation:

### 1. Research Analysis

The Research Analysis tab provides a complete research workflow:

1. Enter a research topic (e.g., "transformer attention mechanisms")
2. Select your preferred model provider (Local or OpenAI)
3. Click "Research"
4. The system will:
   - Fetch relevant papers from arXiv
   - Build a knowledge graph
   - Analyze research trends
   - Identify gaps
   - Generate hypotheses

### 2. Semantic Search

The Semantic Search tab provides high-level semantic search capabilities:

1. Enter your search query
2. Adjust the similarity threshold (0.0 to 1.0)
   - Higher values (e.g., 0.8) return more precise matches
   - Lower values (e.g., 0.5) return more results but may be less relevant
3. Click "Search"
4. View results with similarity scores

### 3. Embedding Search

The Embedding Search tab provides direct access to embedding vectors:

1. Enter your search query
2. Adjust parameters:
   - Similarity threshold (0.0 to 1.0)
   - Result limit (1 to 100)
3. Click "Search Embeddings"
4. View results including:
   - Paper details
   - Similarity scores
   - Raw embedding vectors (expandable)

### 4. Knowledge Graph Visualization

The Knowledge Graph tab provides an interactive visualization of the research network:

1. **Search Functionality:**
   - Enter search terms in the search box
   - Filter by node type (Papers, Concepts, Authors, Methods)
   - Use the Reset button to clear search and restore view

2. **Navigation:**
   - Zoom: Use mouse wheel or pinch gesture
   - Pan: Click and drag
   - Focus: Click on a node to center it
   - Reset View: Click the Reset button

3. **Node Types and Colors:**
   - Papers: Blue
   - Concepts: Red
   - Authors: Green
   - Methods: Orange

4. **Search Results:**
   - Matching nodes are highlighted in gold
   - Results count is displayed
   - First match is automatically focused

## Understanding the Results

### Research Analysis Results

- **Research Overview**: Summary of papers found and analysis time
- **Key Concepts**: Important terms and concepts identified
- **Research Trend Analysis**: Insights into dominant areas and emerging topics
- **Research Gap Analysis**: Identified gaps in current research
- **Generated Hypotheses**: Suggested research directions

### Semantic Search Results

- **Similar Papers**: Papers ranked by semantic similarity
- **Similarity Scores**: Percentage indicating how closely each paper matches your query
- **Abstracts**: Paper summaries for quick review

### Embedding Search Results

- **Query Embedding**: The vector representation of your search query
- **Paper Embeddings**: Vector representations of matching papers
- **Similarity Scores**: Cosine similarity between query and paper embeddings
- **Raw Vectors**: Expandable sections showing the actual embedding vectors

### Knowledge Graph Results

- **Node Information**: Hover over nodes to see details
- **Relationship Types**: Different types of connections between nodes
- **Search Matches**: Highlighted nodes showing search results
- **Node Types**: Color-coded nodes for different entity types

## Advanced Usage

### Local vs. Cloud Models

The system supports two model providers:

1. **Local (Ollama)**:
   - Models: llama3.1:8b, codellama:7b, nomic-embed-text
   - Advantages: Privacy, no API costs
   - Requirements: 8GB+ RAM

2. **Cloud (OpenAI)**:
   - Models: gpt-4, text-embedding-3-small
   - Advantages: Better performance, no local resources
   - Requirements: OpenAI API key

### Knowledge Graph Structure

The Neo4j database stores:

- **Paper Nodes**: Research papers with metadata and embeddings
- **Concept Nodes**: Key concepts and terms
- **Author Nodes**: Paper authors
- **Method Nodes**: Research methods and techniques
- **Relationships**: Connections between papers, concepts, authors, and methods

### API Endpoints

The system provides several REST API endpoints:

- `POST /research`: Full research analysis
- `POST /semantic-search`: Semantic search
- `POST /embedding-search`: Direct embedding search
- `GET /graph-data`: Knowledge graph data
- `GET /health`: Health check

## Troubleshooting

### Common Issues

1. **Neo4j Connection Errors**:
   - Verify Neo4j is running
   - Check connection credentials
   - Ensure Neo4j has enough memory

2. **Model Loading Issues**:
   - For local models: Ensure Ollama is running
   - For OpenAI: Verify API key
   - Check system memory

3. **Search Performance**:
   - Adjust similarity threshold
   - Reduce result limit
   - Use more specific queries

4. **Graph Visualization Issues**:
   - Clear browser cache
   - Check browser console for errors
   - Ensure sufficient memory for large graphs

### Performance Tips

1. **Local Models**:
   - Close other memory-intensive applications
   - Use smaller models if memory is limited
   - Consider using cloud models for large datasets

2. **Search Optimization**:
   - Start with higher similarity thresholds
   - Use specific, focused queries
   - Limit results when possible

3. **Graph Visualization**:
   - Use type filters to reduce visible nodes
   - Focus on specific subgraphs
   - Use search to find relevant nodes quickly

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Feature requests
- Documentation improvements
- Performance optimizations

## License

[Your License Here]

## Contact

[Your Contact Information] 