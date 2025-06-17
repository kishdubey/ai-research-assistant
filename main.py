# AI Research Assistant with Multi-Modal Knowledge Graph
# Complete working implementation with local models

import os
import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Core dependencies
import requests
import feedparser
import PyPDF2
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from neo4j import GraphDatabase
import ollama
import openai
from abc import ABC, abstractmethod

# Web framework
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    """Research paper data structure"""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published: datetime
    categories: List[str]
    content: str = ""
    embeddings: Optional[np.ndarray] = None

@dataclass
class KnowledgeNode:
    """Knowledge graph node"""
    id: str
    type: str  # paper, concept, author, method
    properties: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None

class AbstractModelManager(ABC):
    """Abstract base class for model managers"""
    @abstractmethod
    async def generate_response(self, prompt: str, model_type: str = 'reasoning') -> str:
        pass

    @abstractmethod
    async def get_embeddings(self, text: str) -> np.ndarray:
        pass

class LocalModelManager(AbstractModelManager):
    """Manages local Ollama models"""
    
    def __init__(self):
        self.models = {
            'reasoning': 'llama3.1:8b',
            'code': 'codellama:7b',
            'embedding': 'nomic-embed-text'  # This model produces 768-dimensional embeddings
        }
        self.client = ollama.Client()
        
    async def ensure_models_available(self):
        """Download models if not available"""
        for model_type, model_name in self.models.items():
            try:
                self.client.show(model_name)
                logger.info(f"Model {model_name} already available")
            except:
                logger.info(f"Downloading {model_name}...")
                self.client.pull(model_name)
    
    async def generate_response(self, prompt: str, model_type: str = 'reasoning') -> str:
        """Generate response using local model"""
        model_name = self.models.get(model_type, self.models['reasoning'])
        
        response = self.client.generate(
            model=model_name,
            prompt=prompt,
            stream=False
        )
        
        return response['response']
    
    async def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using local model"""
        response = self.client.embeddings(
            model=self.models['embedding'],
            prompt=text
        )
        return np.array(response['embedding'])

class OpenAIModelManager(AbstractModelManager):
    """Manages OpenAI API models"""
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.models = {
            'reasoning': 'gpt-4o',
            'code': 'gpt-4o',
            'embedding': 'text-embedding-3-small'  # This model produces 1536-dimensional embeddings
        }

    async def generate_response(self, prompt: str, model_type: str = 'reasoning') -> str:
        """Generate response using OpenAI model"""
        model_name = self.models.get(model_type, self.models['reasoning'])
        
        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Error generating response from OpenAI."

    async def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using OpenAI model"""
        try:
            response = await self.client.embeddings.create(
                model=self.models['embedding'],
                input=text,
                encoding_format="float"
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return np.array([])

class ArxivProcessor:
    """Processes arXiv papers"""
    
    def __init__(self, model_manager: AbstractModelManager):
        self.model_manager = model_manager
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search_papers(self, query: str, max_results: int = 50) -> List[ResearchPaper]:
        """Search for papers on arXiv"""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        response = requests.get(self.base_url, params=params)
        feed = feedparser.parse(response.content)
        
        papers = []
        for entry in feed.entries:
            paper = ResearchPaper(
                title=entry.title,
                authors=[author.name for author in entry.authors],
                abstract=entry.summary,
                arxiv_id=entry.id.split('/')[-1],
                published=datetime(*entry.published_parsed[:6]),
                categories=[tag.term for tag in entry.tags]
            )
            
            # Get embeddings for the abstract
            paper.embeddings = await self.model_manager.get_embeddings(
                f"{paper.title} {paper.abstract}"
            )
            
            papers.append(paper)
        
        return papers
    
    async def extract_key_concepts(self, paper: ResearchPaper) -> List[str]:
        """Extract key concepts from paper using local model"""
        prompt = f"""
        Analyze this research paper and extract 10-15 key technical concepts, methods, and terms.
        Focus on specific techniques, algorithms, datasets, and domain-specific terminology.
        
        Title: {paper.title}
        Abstract: {paper.abstract}
        Categories: {', '.join(paper.categories)}
        
        Return only a comma-separated list of key concepts:
        """
        
        response = await self.model_manager.generate_response(prompt)
        concepts = [concept.strip() for concept in response.split(',')]
        return concepts[:15]  # Limit to 15 concepts

class HuggingFaceProcessor:
    """Processes Hugging Face models and datasets"""
    
    def __init__(self, model_manager: AbstractModelManager):
        self.model_manager = model_manager
        self.base_url = "https://huggingface.co/api"
    
    async def get_trending_models(self, limit: int = 50) -> List[Dict]:
        """Get trending models from Hugging Face"""
        response = requests.get(f"{self.base_url}/models", params={
            'sort': 'trending',
            'limit': limit,
            'filter': 'pytorch'
        })
        
        if response.status_code == 200:
            return response.json()
        return []
    
    async def get_datasets(self, limit: int = 50) -> List[Dict]:
        """Get popular datasets"""
        response = requests.get(f"{self.base_url}/datasets", params={
            'sort': 'trending',
            'limit': limit
        })
        
        if response.status_code == 200:
            return response.json()
        return []

class KnowledgeGraphBuilder:
    """Builds and manages the knowledge graph"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graph = nx.DiGraph()
    
    def close(self):
        self.driver.close()
    
    async def add_paper_node(self, paper: ResearchPaper) -> str:
        """Add paper node to knowledge graph with embeddings"""
        with self.driver.session() as session:
            query = """
            CREATE (p:Paper {
                id: $id,
                title: $title,
                authors: $authors,
                abstract: $abstract,
                arxiv_id: $arxiv_id,
                published: $published,
                categories: $categories,
                embeddings: $embeddings,
                embedding_provider: $provider
            })
            RETURN p.id as id
            """
            
            result = session.run(query, {
                'id': f"paper_{paper.arxiv_id}",
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'arxiv_id': paper.arxiv_id,
                'published': paper.published.isoformat(),
                'categories': paper.categories,
                'embeddings': paper.embeddings.tolist() if paper.embeddings is not None else None,
                'provider': os.getenv("MODEL_PROVIDER", "local")
            })
            
            return result.single()['id']
    
    async def add_concept_nodes(self, concepts: List[str], paper_id: str):
        """Add concept nodes and link to paper"""
        with self.driver.session() as session:
            for concept in concepts:
                # Create or merge concept node with proper ID generation
                concept_id = f"concept_{concept.lower().replace(' ', '_')}"
                concept_query = """
                MERGE (c:Concept {id: $id})
                ON CREATE SET 
                    c.name = $name,
                    c.created = datetime()
                ON MATCH SET 
                    c.last_updated = datetime()
                RETURN c.id as id
                """
                
                session.run(concept_query, {
                    'id': concept_id,
                    'name': concept
                })
                
                # Link paper to concept using MERGE to avoid duplicate relationships
                link_query = """
                MATCH (p:Paper {id: $paper_id})
                MATCH (c:Concept {id: $concept_id})
                MERGE (p)-[r:USES_CONCEPT]->(c)
                ON CREATE SET r.created = datetime()
                ON MATCH SET r.last_updated = datetime()
                """
                
                session.run(link_query, {
                    'paper_id': paper_id,
                    'concept_id': concept_id
                })
    
    async def find_similar_papers(self, paper_embeddings: np.ndarray, threshold: float = 0.7) -> List[Dict]:
        """Find similar papers based on embeddings using cosine similarity"""
        with self.driver.session() as session:
            # First get all papers with their embeddings
            query = """
            MATCH (p:Paper)
            RETURN p.id as id, p.title as title, p.abstract as abstract, p.embeddings as embeddings
            """
            
            results = session.run(query)
            papers = []
            
            for record in results:
                if record['embeddings']:
                    # Convert stored embeddings back to numpy array
                    stored_embeddings = np.array(record['embeddings'])
                    # Calculate cosine similarity
                    similarity = np.dot(paper_embeddings, stored_embeddings) / (
                        np.linalg.norm(paper_embeddings) * np.linalg.norm(stored_embeddings)
                    )
                    
                    if similarity >= threshold:
                        papers.append({
                            'id': record['id'],
                            'title': record['title'],
                            'abstract': record['abstract'],
                            'similarity': float(similarity)
                        })
            
            # Sort by similarity score
            return sorted(papers, key=lambda x: x['similarity'], reverse=True)
    
    async def identify_research_gaps(self) -> List[Dict]:
        """Identify potential research gaps"""
        with self.driver.session() as session:
            # Find concepts that appear in few papers
            gap_query = """
            MATCH (c:Concept)<-[:USES_CONCEPT]-(p:Paper)
            WITH c, count(p) as paper_count
            WHERE paper_count < 3
            RETURN c.name as concept, paper_count
            ORDER BY paper_count ASC
            LIMIT 20
            """
            
            results = session.run(gap_query)
            gaps = []
            
            for record in results:
                gaps.append({
                    'concept': record['concept'],
                    'paper_count': record['paper_count'],
                    'gap_type': 'underexplored_concept'
                })
            
            return gaps

class ResearchAgent:
    """Individual research agent with specific capabilities"""
    
    def __init__(self, name: str, specialty: str, model_manager: AbstractModelManager):
        self.name = name
        self.specialty = specialty
        self.model_manager = model_manager
    
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze data based on agent specialty"""
        if self.specialty == "paper_ingestion":
            return await self._analyze_papers(data)
        elif self.specialty == "gap_analysis":
            return await self._analyze_gaps(data)
        elif self.specialty == "hypothesis_generation":
            return await self._generate_hypotheses(data)
        
        return {}
    
    async def _analyze_papers(self, papers: List[ResearchPaper]) -> Dict[str, Any]:
        """Analyze papers for trends and patterns"""
        if not papers:
            return {}
        
        # Analyze trends
        categories = {}
        for paper in papers:
            for cat in paper.categories:
                categories[cat] = categories.get(cat, 0) + 1
        
        # Get AI analysis of trends
        trend_text = f"Research trends in {len(papers)} papers: " + \
                    ", ".join([f"{cat}: {count}" for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]])
        
        prompt = f"""
        Analyze these research trends and provide insights:
        {trend_text}
        
        Provide a brief analysis of:
        1. Dominant research areas
        2. Emerging topics
        3. Potential future directions
        """
        
        analysis = await self.model_manager.generate_response(prompt)
        
        return {
            'trend_analysis': analysis,
            'category_distribution': categories,
            'total_papers': len(papers)
        }
    
    async def _analyze_gaps(self, gaps: List[Dict]) -> Dict[str, Any]:
        """Analyze research gaps"""
        if not gaps:
            return {}
        
        gap_text = "Research gaps identified: " + \
                  ", ".join([f"{gap['concept']} ({gap['paper_count']} papers)" for gap in gaps[:10]])
        
        prompt = f"""
        Analyze these research gaps and provide insights:
        {gap_text}
        
        Identify:
        1. Most promising research opportunities
        2. Potential reasons for these gaps
        3. Suggested research directions
        """
        
        analysis = await self.model_manager.generate_response(prompt)
        
        return {
            'gap_analysis': analysis,
            'identified_gaps': gaps,
            'recommendations': []
        }
    
    async def _generate_hypotheses(self, context: Dict) -> Dict[str, Any]:
        """Generate research hypotheses"""
        prompt = f"""
        Based on the research analysis, generate 5 specific, testable research hypotheses.
        
        Context: {json.dumps(context, indent=2)}
        
        For each hypothesis provide:
        1. Clear statement
        2. Rationale
        3. Potential methodology
        4. Expected impact
        """
        
        hypotheses = await self.model_manager.generate_response(prompt)
        
        return {
            'generated_hypotheses': hypotheses,
            'generation_time': datetime.now().isoformat()
        }

class MultiAgentOrchestrator:
    """Coordinates multiple research agents"""
    
    def __init__(self, model_manager: AbstractModelManager):
        self.model_manager = model_manager
        self.agents = {
            'paper_agent': ResearchAgent('PaperAnalyst', 'paper_ingestion', model_manager),
            'gap_agent': ResearchAgent('GapAnalyst', 'gap_analysis', model_manager),
            'hypothesis_agent': ResearchAgent('HypothesisGenerator', 'hypothesis_generation', model_manager)
        }
        # Initialize Neo4j integration (update credentials as needed)
        self.kg_builder = KnowledgeGraphBuilder(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "research123")
        )
        self.hf_processor = HuggingFaceProcessor(model_manager)
    
    async def orchestrate_research(self, query: str) -> Dict[str, Any]:
        """Orchestrate complete research analysis"""
        logger.info(f"Starting research orchestration for query: {query}")
        
        # Initialize processors
        arxiv_processor = ArxivProcessor(self.model_manager)
        
        # Step 1: Gather papers
        logger.info("Gathering research papers...")
        papers = await arxiv_processor.search_papers(query, max_results=20)
        
        # Persist papers and concepts to Neo4j
        for paper in papers[:5]:  # Limit for demo
            paper_id = await self.kg_builder.add_paper_node(paper)
            concepts = await arxiv_processor.extract_key_concepts(paper)
            await self.kg_builder.add_concept_nodes(concepts, paper_id)
        
        # Step 2: Analyze papers
        logger.info("Analyzing papers...")
        paper_analysis = await self.agents['paper_agent'].analyze(papers)
        
        # Step 3: Build knowledge graph (already done above)
        logger.info("Building knowledge graph...")
        concepts = []
        for paper in papers[:5]:  # Limit for demo
            paper_concepts = await arxiv_processor.extract_key_concepts(paper)
            concepts.extend(paper_concepts)
        
        # Step 4: Identify gaps using Neo4j
        logger.info("Identifying research gaps...")
        gaps = await self.kg_builder.identify_research_gaps()
        gap_analysis = await self.agents['gap_agent'].analyze(gaps)
        
        # Step 5: Generate hypotheses
        logger.info("Generating hypotheses...")
        context = {
            'query': query,
            'paper_analysis': paper_analysis,
            'gap_analysis': gap_analysis
        }
        hypotheses = await self.agents['hypothesis_agent'].analyze(context)
        
        # Step 6: Optionally, fetch trending Hugging Face models/datasets
        trending_models = await self.hf_processor.get_trending_models(limit=5)
        trending_datasets = await self.hf_processor.get_datasets(limit=5)
        
        return {
            'query': query,
            'papers_found': len(papers),
            'paper_analysis': paper_analysis,
            'identified_gaps': gap_analysis,
            'generated_hypotheses': hypotheses,
            'key_concepts': list(set(concepts)),
            'trending_models': trending_models,
            'trending_datasets': trending_datasets,
            'timestamp': datetime.now().isoformat()
        }

# Factory function to get the correct model manager
def get_model_manager(provider: Optional[str] = None) -> AbstractModelManager:
    """Factory function to get the configured model manager."""
    if provider is None:
        provider = os.getenv("MODEL_PROVIDER", "local").lower()
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenAI provider")
        logger.info("Using OpenAI model provider.")
        return OpenAIModelManager(api_key=api_key)
    
    logger.info("Using local (Ollama) model provider.")
    return LocalModelManager()

# FastAPI Application
app = FastAPI(title="AI Research Assistant", description="Multi-Modal Knowledge Graph Research Assistant")

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting AI Research Assistant...")
    # Pre-download local models if that's the default provider
    if os.getenv("MODEL_PROVIDER", "local").lower() == "local":
        logger.info("Local provider set, ensuring models are available...")
        local_manager = LocalModelManager()
        await local_manager.ensure_models_available()
    logger.info("Startup complete.")

@app.get("/")
async def root():
    """Serve the main interface"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Research Assistant</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/axios/0.21.1/axios.min.js"></script>
        <style>
            :root {
                --primary: #2563eb;
                --primary-light: #3b82f6;
                --text: #1f2937;
                --text-light: #6b7280;
                --bg: #ffffff;
                --bg-light: #f9fafb;
                --border: #e5e7eb;
                --shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.5;
                color: var(--text);
                background: var(--bg-light);
                min-height: 100vh;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }

            .header {
                text-align: center;
                margin-bottom: 3rem;
            }

            .header h1 {
                font-size: 2rem;
                font-weight: 600;
                color: var(--text);
                margin-bottom: 0.5rem;
            }

            .header p {
                color: var(--text-light);
                font-size: 1.1rem;
            }

            .search-tabs {
                display: flex;
                gap: 1rem;
                margin-bottom: 2rem;
                justify-content: center;
            }

            .search-tab {
                padding: 0.75rem 1.5rem;
                border: none;
                background: var(--bg);
                color: var(--text-light);
                border-radius: 0.5rem;
                cursor: pointer;
                transition: all 0.2s;
                font-size: 0.95rem;
                box-shadow: var(--shadow);
            }

            .search-tab.active {
                background: var(--primary);
                color: white;
            }

            .search-container {
                background: var(--bg);
                padding: 2rem;
                border-radius: 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 2rem;
            }

            .input-group {
                margin-bottom: 1rem;
            }

            .input-group label {
                display: block;
                margin-bottom: 0.5rem;
                color: var(--text-light);
                font-size: 0.9rem;
            }

            input[type="text"] {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                font-size: 1rem;
                transition: border-color 0.2s;
            }

            input[type="text"]:focus {
                outline: none;
                border-color: var(--primary);
            }

            .settings-container {
                display: flex;
                gap: 1rem;
                align-items: flex-end;
            }

            select {
                padding: 0.75rem;
                border: 1px solid var(--border);
                border-radius: 0.5rem;
                font-size: 1rem;
                background: var(--bg);
                color: var(--text);
                cursor: pointer;
            }

            button {
                padding: 0.75rem 1.5rem;
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 0.5rem;
                font-size: 1rem;
                cursor: pointer;
                transition: background 0.2s;
            }

            button:hover {
                background: var(--primary-light);
            }

            button:disabled {
                opacity: 0.7;
                cursor: not-allowed;
            }

            .results {
                margin-top: 2rem;
            }

            .result-section {
                background: var(--bg);
                padding: 1.5rem;
                border-radius: 1rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .result-section h3 {
                color: var(--text);
                margin-bottom: 1rem;
                font-size: 1.2rem;
            }

            .loading {
                text-align: center;
                padding: 2rem;
                color: var(--text-light);
            }

            .error {
                background: #fee2e2;
                color: #dc2626;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 1rem 0;
            }

            .similarity-score {
                color: var(--primary);
                font-weight: 500;
            }

            .threshold-slider {
                width: 100%;
                margin: 0.5rem 0;
            }

            details {
                margin: 1rem 0;
            }

            details summary {
                cursor: pointer;
                color: var(--text-light);
                padding: 0.5rem 0;
            }

            details pre {
                background: var(--bg-light);
                padding: 1rem;
                border-radius: 0.5rem;
                overflow-x: auto;
                margin-top: 0.5rem;
                font-size: 0.9rem;
            }

            .paper-card {
                background: var(--bg-light);
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
            }

            .paper-card h4 {
                color: var(--text);
                margin-bottom: 0.5rem;
            }

            .paper-card p {
                color: var(--text-light);
                margin-bottom: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>AI Research Assistant</h1>
                <p>Multi-Modal Knowledge Graph Research</p>
            </div>
            
            <div class="search-tabs">
                <button class="search-tab active" onclick="switchTab('research')">Research</button>
                <button class="search-tab" onclick="switchTab('semantic')">Semantic Search</button>
                <button class="search-tab" onclick="switchTab('embedding')">Embeddings</button>
            </div>
            
            <div id="researchTab" class="search-container">
                <div class="input-group">
                    <input type="text" id="queryInput" placeholder="Enter research topic (e.g., 'transformer attention mechanisms')" />
                </div>
            </div>

            <div id="semanticTab" class="search-container" style="display: none;">
                <div class="input-group">
                    <input type="text" id="semanticQueryInput" placeholder="Enter semantic search query" />
                </div>
                <div class="input-group">
                    <label for="threshold">Similarity Threshold</label>
                    <input type="range" id="threshold" class="threshold-slider" min="0" max="1" step="0.1" value="0.7">
                    <span id="thresholdValue">0.7</span>
                </div>
            </div>

            <div id="embeddingTab" class="search-container" style="display: none;">
                <div class="input-group">
                    <input type="text" id="embeddingQueryInput" placeholder="Enter search query" />
                </div>
                <div class="input-group">
                    <label for="embeddingThreshold">Similarity Threshold</label>
                    <input type="range" id="embeddingThreshold" class="threshold-slider" min="0" max="1" step="0.1" value="0.7">
                    <span id="embeddingThresholdValue">0.7</span>
                </div>
                <div class="input-group">
                    <label for="embeddingLimit">Result Limit</label>
                    <input type="number" id="embeddingLimit" min="1" max="100" value="10" style="width: 100px;">
                </div>
            </div>

            <div class="settings-container">
                <div class="input-group" style="flex: 1;">
                    <label for="providerSelect" id="providerLabel">Model Provider</label>
                    <select id="providerSelect">
                        <option value="local">Ollama (Local)</option>
                        <option value="openai">OpenAI (Cloud)</option>
                    </select>
                </div>
                <button onclick="startResearch()" id="searchBtn">Research</button>
                <button onclick="startSemanticSearch()" id="semanticSearchBtn" style="display: none;">Search</button>
                <button onclick="startEmbeddingSearch()" id="embeddingSearchBtn" style="display: none;">Search Embeddings</button>
            </div>
            
            <div id="results" class="results"></div>
        </div>

        <script>
            function switchTab(tab) {
                document.querySelectorAll('.search-tab').forEach(t => t.classList.remove('active'));
                event.target.classList.add('active');
                
                // Hide all tabs and buttons
                document.getElementById('researchTab').style.display = 'none';
                document.getElementById('semanticTab').style.display = 'none';
                document.getElementById('embeddingTab').style.display = 'none';
                document.getElementById('searchBtn').style.display = 'none';
                document.getElementById('semanticSearchBtn').style.display = 'none';
                document.getElementById('embeddingSearchBtn').style.display = 'none';
                
                // Show/hide provider select
                const providerSelect = document.getElementById('providerSelect');
                const providerLabel = document.getElementById('providerLabel');
                if (tab === 'research') {
                    providerSelect.style.display = 'block';
                    providerLabel.style.display = 'block';
                } else {
                    providerSelect.style.display = 'none';
                    providerLabel.style.display = 'none';
                }
                
                // Show selected tab and button
                if (tab === 'research') {
                    document.getElementById('researchTab').style.display = 'flex';
                    document.getElementById('searchBtn').style.display = 'block';
                } else if (tab === 'semantic') {
                    document.getElementById('semanticTab').style.display = 'flex';
                    document.getElementById('semanticSearchBtn').style.display = 'block';
                } else if (tab === 'embedding') {
                    document.getElementById('embeddingTab').style.display = 'flex';
                    document.getElementById('embeddingSearchBtn').style.display = 'block';
                }
            }

            document.getElementById('threshold').addEventListener('input', function(e) {
                document.getElementById('thresholdValue').textContent = e.target.value;
            });

            document.getElementById('embeddingThreshold').addEventListener('input', function(e) {
                document.getElementById('embeddingThresholdValue').textContent = e.target.value;
            });

            async function startEmbeddingSearch() {
                const query = document.getElementById('embeddingQueryInput').value.trim();
                const threshold = parseFloat(document.getElementById('embeddingThreshold').value);
                const limit = parseInt(document.getElementById('embeddingLimit').value);
                const provider = document.getElementById('providerSelect').value;
                
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                const searchBtn = document.getElementById('embeddingSearchBtn');
                
                searchBtn.disabled = true;
                searchBtn.textContent = '🔍 Searching...';
                
                resultsDiv.innerHTML = '<div class="loading">🔍 Performing embedding search...</div>';
                
                try {
                    const response = await axios.post('/embedding-search', { 
                        query: query,
                        threshold: threshold,
                        limit: limit,
                        provider: provider
                    });
                    displayEmbeddingResults(response.data);
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.response?.data?.detail || error.message}</div>`;
                } finally {
                    searchBtn.disabled = false;
                    searchBtn.textContent = '🔍 Search Embeddings';
                }
            }
            
            function displayEmbeddingResults(data) {
                const resultsDiv = document.getElementById('results');
                
                let html = `
                    <div class="result-section">
                        <h3>🔍 Embedding Search Results</h3>
                        <p><strong>Query:</strong> ${data.query}</p>
                        <p><strong>Found:</strong> ${data.similar_papers.length} similar papers</p>
                        <details>
                            <summary>Query Embedding (${data.query_embedding.length} dimensions)</summary>
                            <pre>${JSON.stringify(data.query_embedding, null, 2)}</pre>
                        </details>
                    </div>
                `;
                
                if (data.similar_papers.length > 0) {
                    html += `
                        <div class="result-section">
                            <h3>📚 Similar Papers</h3>
                            ${data.similar_papers.map(paper => `
                                <div style="margin-bottom: 20px; padding: 15px; background: #f8fafc; border-radius: 10px;">
                                    <h4>${paper.title}</h4>
                                    <p><span class="similarity-score">Similarity: ${(paper.similarity * 100).toFixed(1)}%</span></p>
                                    <p>${paper.abstract}</p>
                                    <details>
                                        <summary>Paper Embedding (${paper.embedding.length} dimensions)</summary>
                                        <pre>${JSON.stringify(paper.embedding, null, 2)}</pre>
                                    </details>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    html += `
                        <div class="result-section">
                            <p>No similar papers found. Try adjusting the similarity threshold or using different search terms.</p>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = html;
            }
            
            async function startResearch() {
                const query = document.getElementById('queryInput').value.trim();
                const provider = document.getElementById('providerSelect').value;
                
                if (!query) {
                    alert('Please enter a research topic');
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                const searchBtn = document.getElementById('searchBtn');
                
                searchBtn.disabled = true;
                searchBtn.textContent = '🔄 Researching...';
                
                resultsDiv.innerHTML = '<div class="loading">🤖 AI agents are analyzing research papers and building knowledge graph...</div>';
                
                try {
                    const response = await axios.post('/research', { 
                        query: query,
                        provider: provider 
                    });
                    displayResults(response.data);
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.response?.data?.detail || error.message}</div>`;
                } finally {
                    searchBtn.disabled = false;
                    searchBtn.textContent = '🚀 Research';
                }
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                
                let html = `
                    <div class="result-section">
                        <h3>📊 Research Overview</h3>
                        <p><strong>Query:</strong> ${data.query}</p>
                        <p><strong>Papers Analyzed:</strong> ${data.papers_found}</p>
                        <p><strong>Analysis Time:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                    </div>
                `;
                
                if (data.key_concepts && data.key_concepts.length > 0) {
                    html += `
                        <div class="result-section">
                            <h3>🔑 Key Concepts Identified</h3>
                            <div>
                                ${data.key_concepts.map(concept => `<span class="concept-tag">${concept}</span>`).join('')}
                            </div>
                        </div>
                    `;
                }
                
                if (data.paper_analysis && data.paper_analysis.trend_analysis) {
                    html += `
                        <div class="result-section">
                            <h3>📈 Research Trend Analysis</h3>
                            <div style="white-space: pre-line;">${data.paper_analysis.trend_analysis}</div>
                        </div>
                    `;
                }
                
                if (data.identified_gaps && data.identified_gaps.gap_analysis) {
                    html += `
                        <div class="result-section">
                            <h3>🔍 Research Gap Analysis</h3>
                            <div style="white-space: pre-line;">${data.identified_gaps.gap_analysis}</div>
                        </div>
                    `;
                }
                
                if (data.generated_hypotheses && data.generated_hypotheses.generated_hypotheses) {
                    html += `
                        <div class="result-section">
                            <h3>💡 Generated Research Hypotheses</h3>
                            <div style="white-space: pre-line;">${data.generated_hypotheses.generated_hypotheses}</div>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = html;
            }
            
            async function startSemanticSearch() {
                const query = document.getElementById('semanticQueryInput').value.trim();
                const threshold = parseFloat(document.getElementById('threshold').value);
                const provider = document.getElementById('providerSelect').value;
                
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                const resultsDiv = document.getElementById('results');
                const searchBtn = document.getElementById('semanticSearchBtn');
                
                searchBtn.disabled = true;
                searchBtn.textContent = '🔍 Searching...';
                
                resultsDiv.innerHTML = '<div class="loading">🔍 Performing semantic search...</div>';
                
                try {
                    const response = await axios.post('/semantic-search', { 
                        query: query,
                        threshold: threshold,
                        provider: provider
                    });
                    displaySemanticResults(response.data);
                } catch (error) {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.response?.data?.detail || error.message}</div>`;
                } finally {
                    searchBtn.disabled = false;
                    searchBtn.textContent = '🔍 Search';
                }
            }

            function displaySemanticResults(data) {
                const resultsDiv = document.getElementById('results');
                
                let html = `
                    <div class="result-section">
                        <h3>🔍 Semantic Search Results</h3>
                        <p><strong>Query:</strong> ${data.query}</p>
                        <p><strong>Found:</strong> ${data.similar_papers.length} similar papers</p>
                    </div>
                `;
                
                if (data.similar_papers.length > 0) {
                    html += `
                        <div class="result-section">
                            <h3>📚 Similar Papers</h3>
                            ${data.similar_papers.map(paper => `
                                <div class="paper-card">
                                    <h4>${paper.title}</h4>
                                    <p><span class="similarity-score">Similarity: ${(paper.similarity * 100).toFixed(1)}%</span></p>
                                    <p>${paper.abstract}</p>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    html += `
                        <div class="result-section">
                            <p>No similar papers found. Try adjusting the similarity threshold or using different search terms.</p>
                        </div>
                    `;
                }
                
                resultsDiv.innerHTML = html;
            }

            // Add Neo4j graph visualization
            async function fetchAndDisplayGraph() {
                try {
                    // Remove any existing graph section
                    const existingGraph = document.querySelector('.graph-section');
                    if (existingGraph) {
                        existingGraph.remove();
                    }

                    const response = await axios.get('/graph-data');
                    const graphData = response.data;
                    
                    console.log('Graph data received:', graphData); // Debug log
                    
                    if (!graphData.nodes || !graphData.edges) {
                        console.error('Invalid graph data structure:', graphData);
                        return;
                    }
                    
                    if (graphData.nodes.length === 0) {
                        const graphSection = document.createElement('div');
                        graphSection.className = 'result-section graph-section';
                        graphSection.innerHTML = `
                            <h3>📊 Knowledge Graph Visualization</h3>
                            <p>No graph data available. Start by adding some papers through the research tab.</p>
                        `;
                        document.getElementById('results').appendChild(graphSection);
                        return;
                    }
                    
                    const graphSection = document.createElement('div');
                    graphSection.className = 'result-section graph-section';
                    graphSection.innerHTML = `
                        <h3>📊 Knowledge Graph Visualization</h3>
                        <div class="graph-controls" style="margin-bottom: 1rem;">
                            <div class="input-group" style="display: flex; gap: 1rem; align-items: center;">
                                <input type="text" id="graphSearchInput" placeholder="Search nodes..." style="flex: 1;">
                                <select id="graphSearchType" style="width: 150px;">
                                    <option value="all">All Types</option>
                                    <option value="paper">Papers</option>
                                    <option value="concept">Concepts</option>
                                    <option value="author">Authors</option>
                                    <option value="method">Methods</option>
                                </select>
                                <button onclick="searchGraph()" style="white-space: nowrap;">🔍 Search</button>
                                <button onclick="resetGraphView()" style="white-space: nowrap;">🔄 Reset</button>
                            </div>
                            <div id="searchResults" style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-light);"></div>
                        </div>
                        <div id="graph-container" style="height: 500px; border: 1px solid var(--border); border-radius: 0.5rem;"></div>
                    `;
                    
                    document.getElementById('results').appendChild(graphSection);
                    
                    // Initialize vis.js network
                    const container = document.getElementById('graph-container');
                    
                    // Prepare nodes with colors based on type and ensure unique IDs
                    const nodeMap = new Map();
                    graphData.nodes.forEach(node => {
                        if (!nodeMap.has(node.id)) {
                            nodeMap.set(node.id, {
                                id: node.id,
                                label: node.title,
                                group: node.label.toLowerCase(),
                                title: `${node.label}: ${node.title}`, // Tooltip
                                color: getNodeColor(node.label),
                                originalColor: getNodeColor(node.label),
                                type: node.label.toLowerCase(),
                                searchableText: `${node.title} ${node.abstract || ''}`.toLowerCase()
                            });
                        }
                    });
                    
                    // Prepare edges with arrows and ensure unique connections
                    const edgeMap = new Map();
                    graphData.edges.forEach(edge => {
                        const edgeKey = `${edge.from}-${edge.to}-${edge.label}`;
                        if (!edgeMap.has(edgeKey)) {
                            edgeMap.set(edgeKey, {
                                from: edge.from,
                                to: edge.to,
                                label: edge.label,
                                arrows: 'to',
                                smooth: { type: 'curvedCW', roundness: 0.2 }
                            });
                        }
                    });
                    
                    const data = {
                        nodes: new vis.DataSet(Array.from(nodeMap.values())),
                        edges: new vis.DataSet(Array.from(edgeMap.values()))
                    };
                    
                    const options = {
                        nodes: {
                            shape: 'dot',
                            size: 16,
                            font: {
                                size: 12,
                                face: 'Tahoma'
                            },
                            borderWidth: 2,
                            shadow: true
                        },
                        edges: {
                            width: 1,
                            font: {
                                size: 10,
                                align: 'middle'
                            },
                            color: {
                                color: '#848484',
                                highlight: '#848484',
                                hover: '#848484',
                                inherit: 'from',
                                opacity: 0.8
                            },
                            smooth: {
                                type: 'curvedCW',
                                roundness: 0.2
                            }
                        },
                        physics: {
                            stabilization: {
                                iterations: 100
                            },
                            barnesHut: {
                                gravitationalConstant: -2000,
                                springConstant: 0.04,
                                springLength: 200
                            }
                        },
                        interaction: {
                            hover: true,
                            tooltipDelay: 200,
                            zoomView: true,
                            dragView: true
                        }
                    };
                    
                    window.network = new vis.Network(container, data, options);
                    
                    // Add keyboard shortcut for search
                    document.getElementById('graphSearchInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                            searchGraph();
                        }
                    });
                    
                } catch (error) {
                    console.error('Error fetching or displaying graph:', error);
                    const graphSection = document.createElement('div');
                    graphSection.className = 'result-section graph-section';
                    graphSection.innerHTML = `
                        <h3>📊 Knowledge Graph Visualization</h3>
                        <div class="error">Error loading graph: ${error.message}</div>
                    `;
                    document.getElementById('results').appendChild(graphSection);
                }
            }

            // Search function for the graph
            function searchGraph() {
                const searchTerm = document.getElementById('graphSearchInput').value.toLowerCase();
                const searchType = document.getElementById('graphSearchType').value;
                const resultsDiv = document.getElementById('searchResults');
                
                if (!window.network) {
                    resultsDiv.innerHTML = 'Graph not initialized';
                    return;
                }
                
                // Reset all nodes to original color
                const nodes = window.network.body.data.nodes;
                nodes.forEach(node => {
                    node.color = node.originalColor;
                });
                window.network.body.data.nodes.update(nodes.get());
                
                if (!searchTerm) {
                    resultsDiv.innerHTML = '';
                    return;
                }
                
                // Find matching nodes
                const matches = nodes.get().filter(node => {
                    const typeMatch = searchType === 'all' || node.type === searchType;
                    const textMatch = node.searchableText.includes(searchTerm);
                    return typeMatch && textMatch;
                });
                
                if (matches.length === 0) {
                    resultsDiv.innerHTML = 'No matches found';
                    return;
                }
                
                // Highlight matching nodes
                matches.forEach(node => {
                    node.color = {
                        background: '#FFD700',
                        border: '#FFA500',
                        highlight: {
                            background: '#FFD700',
                            border: '#FFA500'
                        }
                    };
                });
                window.network.body.data.nodes.update(matches);
                
                // Focus on the first match
                if (matches.length > 0) {
                    window.network.focus(matches[0].id, {
                        scale: 1.5,
                        animation: true
                    });
                }
                
                // Show results count
                resultsDiv.innerHTML = `Found ${matches.length} matching node${matches.length === 1 ? '' : 's'}`;
            }

            // Reset graph view
            function resetGraphView() {
                const nodes = window.network.body.data.nodes;
                nodes.forEach(node => {
                    node.color = node.originalColor;
                });
                window.network.body.data.nodes.update(nodes.get());
                window.network.fit({
                    animation: true
                });
                document.getElementById('graphSearchInput').value = '';
                document.getElementById('searchResults').innerHTML = '';
            }

            // Helper function to get node colors based on type
            function getNodeColor(label) {
                const colors = {
                    'paper': '#97C2FC',
                    'concept': '#FB7E81',
                    'author': '#7BE141',
                    'method': '#FFA807',
                    'default': '#E5E5E5'
                };
                return colors[label.toLowerCase()] || colors.default;
            }

            // Call fetchAndDisplayGraph when the page loads
            document.addEventListener('DOMContentLoaded', fetchAndDisplayGraph);
        </script>
        <!-- Add vis.js for graph visualization -->
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    </body>
    </html>
    """)

@app.post("/research")
async def conduct_research(request: dict):
    """Conduct research analysis"""
    query = request.get('query', '').strip()
    provider = request.get('provider', os.getenv("MODEL_PROVIDER", "local")).strip().lower()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        model_manager = get_model_manager(provider)
        orchestrator = MultiAgentOrchestrator(model_manager)
        result = await orchestrator.orchestrate_research(query)
        return result
    except Exception as e:
        logger.error(f"Research error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@app.post("/semantic-search")
async def semantic_search(request: dict):
    """Perform semantic search on papers"""
    search_query = request.get('query', '').strip()
    threshold = float(request.get('threshold', 0.7))
    provider = request.get('provider', os.getenv("MODEL_PROVIDER", "local")).strip().lower()

    if not search_query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        model_manager = get_model_manager(provider)
        # Get embeddings for the query
        query_embeddings = await model_manager.get_embeddings(search_query)
        
        # Initialize knowledge graph builder
        kg_builder = KnowledgeGraphBuilder(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "research123")
        )
        
        # Find similar papers with matching embedding dimensions
        with kg_builder.driver.session() as session:
            embedding_dim = len(query_embeddings)
            
            # Get papers with matching embedding dimensions
            cypher_query = """
            MATCH (p:Paper)
            WHERE p.embeddings IS NOT NULL 
            AND size(p.embeddings) = $dim
            AND p.embedding_provider = $provider
            RETURN p.id as id, p.title as title, p.abstract as abstract, p.embeddings as embeddings
            """
            
            results = session.run(cypher_query, {
                'dim': embedding_dim,
                'provider': provider
            })
            
            papers = []
            for record in results:
                stored_embeddings = np.array(record['embeddings'])
                similarity = np.dot(query_embeddings, stored_embeddings) / (
                    np.linalg.norm(query_embeddings) * np.linalg.norm(stored_embeddings)
                )
                
                if similarity >= threshold:
                    papers.append({
                        'id': record['id'],
                        'title': record['title'],
                        'abstract': record['abstract'],
                        'similarity': float(similarity)
                    })
            
            # Sort by similarity score
            papers = sorted(papers, key=lambda x: x['similarity'], reverse=True)
            
            return {
                'query': search_query,
                'similar_papers': papers,
                'timestamp': datetime.now().isoformat(),
                'provider': provider,
                'embedding_dimension': embedding_dim
            }
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/embedding-search")
async def embedding_search(request: dict):
    """Direct embedding search with raw vectors"""
    search_query = request.get('query', '').strip()
    threshold = float(request.get('threshold', 0.7))
    limit = int(request.get('limit', 10))
    provider = request.get('provider', os.getenv("MODEL_PROVIDER", "local")).strip().lower()

    if not search_query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    try:
        model_manager = get_model_manager(provider)
        query_embeddings = await model_manager.get_embeddings(search_query)
        
        # Initialize knowledge graph builder
        kg_builder = KnowledgeGraphBuilder(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "research123")
        )
        
        # Get all papers with their embeddings
        with kg_builder.driver.session() as session:
            # First, get the embedding dimension from the query
            embedding_dim = len(query_embeddings)
            
            # Then, get papers with matching embedding dimensions
            cypher_query = """
            MATCH (p:Paper)
            WHERE p.embeddings IS NOT NULL AND size(p.embeddings) = $dim
            RETURN p.id as id, p.title as title, p.abstract as abstract, p.embeddings as embeddings
            """
            
            results = session.run(cypher_query, {'dim': embedding_dim})
            papers = []
            
            for record in results:
                stored_embeddings = np.array(record['embeddings'])
                similarity = np.dot(query_embeddings, stored_embeddings) / (
                    np.linalg.norm(query_embeddings) * np.linalg.norm(stored_embeddings)
                )
                
                if similarity >= threshold:
                    papers.append({
                        'id': record['id'],
                        'title': record['title'],
                        'abstract': record['abstract'],
                        'similarity': float(similarity),
                        'embedding': stored_embeddings.tolist()
                    })
            
            # Sort and limit results
            papers = sorted(papers, key=lambda x: x['similarity'], reverse=True)[:limit]
            
            return {
                'query': search_query,
                'query_embedding': query_embeddings.tolist(),
                'similar_papers': papers,
                'timestamp': datetime.now().isoformat(),
                'embedding_dimension': embedding_dim,
                'provider': provider
            }
    except Exception as e:
        logger.error(f"Embedding search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/graph-data")
async def get_graph_data():
    """Get graph data for visualization"""
    try:
        kg_builder = KnowledgeGraphBuilder(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "research123")
        )
        
        with kg_builder.driver.session() as session:
            # Get nodes with their properties
            nodes_query = """
            MATCH (n)
            RETURN n.id as id, 
                   labels(n)[0] as label, 
                   n.title as title,
                   n.name as name,
                   n.abstract as abstract
            """
            nodes_result = session.run(nodes_query)
            nodes = []
            for record in nodes_result:
                node = {
                    'id': record['id'],
                    'label': record['label'],
                    'title': record['title'] or record['name'] or record['id'],
                    'abstract': record['abstract'] or ''
                }
                nodes.append(node)
            
            # Get edges with their properties
            edges_query = """
            MATCH (n)-[r]->(m)
            RETURN n.id as from, 
                   m.id as to, 
                   type(r) as label,
                   r.weight as weight
            """
            edges_result = session.run(edges_query)
            edges = []
            for record in edges_result:
                edge = {
                    'from': record['from'],
                    'to': record['to'],
                    'label': record['label'],
                    'weight': record['weight'] or 1
                }
                edges.append(edge)
            
            logger.info(f"Retrieved {len(nodes)} nodes and {len(edges)} edges from Neo4j")
            
            return {
                'nodes': nodes,
                'edges': edges,
                'timestamp': datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error fetching graph data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph data: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting AI Research Assistant...")
    print("📋 Make sure you have:")
    print("   - Ollama installed and running")
    print("   - Neo4j running (optional for full functionality)")
    print("   - At least 8GB RAM for models")
    print("\n🌐 Opening web interface at http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)