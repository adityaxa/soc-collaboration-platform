#!/bin/bash

echo "Setting up SOC Collaboration Platform..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d

echo "Setup complete! Infrastructure is running."
echo "To start the services:"
echo "  python psi_service/psi_server.py"
echo "  python api/main.py"
echo "  python fl_service/fl_server.py"
