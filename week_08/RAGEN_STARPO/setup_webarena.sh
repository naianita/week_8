#!/bin/bash
# WebArena Setup Script for Mac OS
# Sets up WebArena evaluation environment (baseline testing only, no training)

set -e  # Exit on error

echo "=================================================="
echo "WebArena Setup Script"
echo "=================================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Mac
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Mac OS${NC}"
fi

# Create WebArena directory
WEBARENA_DIR="WebArena"
if [ -d "$WEBARENA_DIR" ]; then
    echo -e "${YELLOW}WebArena directory already exists. Skipping clone...${NC}"
else
    echo "Step 1: Cloning WebArena repository..."
    git clone https://github.com/web-arena-x/webarena.git "$WEBARENA_DIR"
    echo -e "${GREEN}✓ Repository cloned${NC}"
fi

cd "$WEBARENA_DIR"

# Check Docker installation
echo ""
echo "Step 2: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not found${NC}"
    echo ""
    echo "Please install Docker Desktop for Mac:"
    echo "  https://docs.docker.com/desktop/install/mac-install/"
    echo ""
    echo "After installation:"
    echo "  1. Open Docker Desktop"
    echo "  2. Wait for it to start"
    echo "  3. Run this script again"
    exit 1
else
    echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"
fi

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo -e "${RED}✗ Docker daemon not running${NC}"
    echo ""
    echo "Please start Docker Desktop and run this script again"
    exit 1
else
    echo -e "${GREEN}✓ Docker daemon running${NC}"
fi

# Check Docker Compose
echo ""
echo "Step 3: Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not found${NC}"
    echo "Docker Compose should come with Docker Desktop"
    exit 1
else
    echo -e "${GREEN}✓ Docker Compose available${NC}"
fi

# Setup Python virtual environment
echo ""
echo "Step 4: Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

source venv/bin/activate

# Install Python dependencies
echo ""
echo "Step 5: Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}No requirements.txt found. Installing minimal dependencies...${NC}"
    pip install --upgrade pip
    pip install openai python-dotenv requests beautifulsoup4 playwright
    pip install selenium webdriver-manager
    echo -e "${GREEN}✓ Minimal dependencies installed${NC}"
fi

# Install Playwright browsers
echo ""
echo "Step 6: Installing Playwright browsers..."
playwright install chromium
echo -e "${GREEN}✓ Playwright browsers installed${NC}"

# Download task benchmark
echo ""
echo "Step 7: Setting up task benchmark..."
if [ ! -d "config_files" ]; then
    echo -e "${YELLOW}Config files directory not found${NC}"
    echo "The task benchmark should be in the repository"
else
    echo -e "${GREEN}✓ Task config files found${NC}"
fi

# Create .env file template if it doesn't exist
echo ""
echo "Step 8: Setting up environment variables..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOL'
# WebArena Environment Configuration
# Add your API keys here

# OpenAI API Key (required for GPT-4 baseline)
OPENAI_API_KEY=your_openai_api_key_here

# WebArena URLs (default local Docker setup)
SHOPPING=http://localhost:7770
SHOPPING_ADMIN=http://localhost:7780/admin
REDDIT=http://localhost:9999
GITLAB=http://localhost:8023
WIKIPEDIA=http://localhost:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
MAP=http://localhost:3000
HOMEPAGE=http://localhost:4399

# Database credentials
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=webarena
EOL
    echo -e "${GREEN}✓ .env template created${NC}"
    echo -e "${YELLOW}IMPORTANT: Edit .env and add your OPENAI_API_KEY${NC}"
else
    echo -e "${YELLOW}.env file already exists${NC}"
fi

# Instructions for Docker setup
echo ""
echo "=================================================="
echo "WebArena Setup Complete!"
echo "=================================================="
echo ""
echo -e "${YELLOW}NEXT STEPS:${NC}"
echo ""
echo "1. Edit WebArena/.env and add your OpenAI API key:"
echo "   nano WebArena/.env"
echo ""
echo "2. Start Docker containers (OPTIONAL - for full WebArena):"
echo "   cd WebArena"
echo "   docker-compose up -d"
echo "   (Note: This may take 10-30 minutes and requires ~10GB disk space)"
echo ""
echo "3. For QUICK BASELINE EVALUATION (recommended):"
echo "   python eval/evaluate_webarena_baseline.py --quick"
echo "   (Uses cached baseline results from WebArena paper)"
echo ""
echo "4. For LIVE EVALUATION (requires Docker containers):"
echo "   python eval/evaluate_webarena_baseline.py --live --num_tasks 50"
echo ""
echo "=================================================="
echo ""
echo -e "${GREEN}Setup script completed successfully!${NC}"
echo ""
