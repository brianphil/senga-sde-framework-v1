#!/bin/bash

# Senga SDE Demo Launcher
# Starts both API and Streamlit demo interface

set -e

echo "🚚 Senga Sequential Decision Engine - Demo Launcher"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
    source venv/bin/activate
fi

# Check if databases exist
if [ ! -f "data/senga_config.db" ] || [ ! -f "data/senga_state.db" ]; then
    echo -e "${YELLOW}🗄️  Initializing databases...${NC}"
    python scripts/initialize_system.py
    echo -e "${GREEN}✅ Databases initialized${NC}"
else
    echo -e "${GREEN}✅ Databases found${NC}"
fi

# Check if API is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  API already running on port 8000${NC}"
else
    echo -e "${YELLOW}🚀 Starting FastAPI backend...${NC}"
    # Start API in background
    python src/api/main.py &
    API_PID=$!
    echo -e "${GREEN}✅ API started (PID: $API_PID)${NC}"
    
    # Wait for API to be ready
    echo -e "${YELLOW}⏳ Waiting for API to be ready...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is ready${NC}"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ API failed to start${NC}"
            kill $API_PID 2>/dev/null || true
            exit 1
        fi
    done
fi

# Start Streamlit
echo -e "${YELLOW}🎨 Starting Streamlit demo interface...${NC}"
echo ""
echo -e "${GREEN}🌐 Demo will open at: http://localhost:8501${NC}"
echo -e "${GREEN}📡 API available at: http://localhost:8000${NC}"
echo -e "${GREEN}📚 API docs at: http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Trap Ctrl+C to cleanup
trap 'echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"; kill $API_PID 2>/dev/null || true; exit 0' INT

# Run Streamlit
streamlit run scripts/streamlit_demo.py

# Cleanup on exit
kill $API_PID 2>/dev/null || true
echo -e "${GREEN}✅ Services stopped${NC}"