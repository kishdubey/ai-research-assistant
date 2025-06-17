#!/bin/bash

echo "🚀 Setting up AI Research Assistant..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create project directory
PROJECT_DIR="ai-research-assistant"
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir "$PROJECT_DIR"
    echo "📁 Created project directory: $PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Create necessary directories
mkdir -p data logs

echo "📦 Starting services with Docker Compose..."

# Start Neo4j and Ollama first
docker-compose up -d neo4j ollama

echo "⏳ Waiting for services to start..."
sleep 30

# Check if Ollama is ready
echo "🤖 Setting up Ollama models..."
docker-compose exec ollama ollama pull llama3.1:8b
docker-compose exec ollama ollama pull codellama:7b
docker-compose exec ollama ollama pull nomic-embed-text

echo "🏗️ Building and starting the Research Assistant..."
docker-compose up -d research-assistant

echo "✅ Setup complete!"
echo ""
echo "🌐 Services available at:"
echo "   - Research Assistant: http://localhost:8000"
echo "   - Neo4j Browser: http://localhost:7474 (user: neo4j, password: research123) (MATCH (n) RETURN n))"
echo "   - Ollama API: http://localhost:11434"
echo ""
echo "📊 To check status: docker-compose ps"
echo "📋 To view logs: docker-compose logs -f research-assistant"
echo "🛑 To stop: docker-compose down"
echo "To restart with modification, run docker-compose down && docker-compose up --build -d"