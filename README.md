# SOC Collaboration Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![Flower FL](https://img.shields.io/badge/Flower-1.5.0-green.svg)](https://flower.dev/)

A privacy-preserving platform for Security Operations Center (SOC) analysts to collaborate across organizations using **Private Set Intersection (PSI)** and **Federated Learning (FL)**. Share threat intelligence and train machine learning models collaboratively without exposing sensitive data.

## Features

- **Private Set Intersection (PSI)**: Find common threat indicators across organizations without revealing unique IOCs
-  **Federated Learning**: Collaboratively train ML models without centralizing sensitive security data
-  **Differential Privacy**: Protect model updates with calibrated noise to prevent information leakage
-  **Web Dashboard**: Intuitive interface for SOC analysts to query threats and manage training
-  **Docker Deployment**: Easy containerized deployment for development and production
-  **Real-time Monitoring**: Track query results and training progress in real-time

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)
- [License](#license)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Dashboard                           │
│            (React/HTML - User Interface)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
│         (FastAPI - Unified REST Interface)                  │
└────────┬────────────────────────────────────┬───────────────┘
         │                                    │
         ▼                                    ▼
┌──────────────────┐                 ┌─────────────────────┐
│   PSI Service    │                 │   FL Service        │
│  (OpenMined PSI) │                 │  (Flower Framework) │
├──────────────────┤                 ├─────────────────────┤
│ • Hash & Match   │                 │ • Server            │
│ • Privacy Layer  │                 │ • Client Nodes      │
│ • Query Engine   │                 │ • Model Aggregation │
└──────────────────┘                 └─────────────────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Infrastructure Services                        │
│         (Redis, PostgreSQL, Docker Network)                 │
└─────────────────────────────────────────────────────────────┘
```

### Component Descriptions

**PSI Service**: Implements private set intersection using OpenMined PSI library, allowing organizations to find common threat indicators without revealing their complete threat landscape.

**FL Service**: Uses the Flower federated learning framework to enable collaborative training of threat detection models with differential privacy protection.

**API Gateway**: FastAPI-based unified interface that integrates both PSI and FL services, providing a single entry point for all operations.

**Web Dashboard**: Responsive web interface for SOC analysts to submit queries, start training jobs, and monitor results in real-time.

## Start

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- 4GB RAM minimum (8GB recommended)

### Fast Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/soc-collaboration-platform.git
cd soc-collaboration-platform

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Start all services
python psi_service/psi_server.py &
python api/main.py &
python fl_service/fl_server.py &

# Open the dashboard
open http://localhost/frontend/index.html
```

##  Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/soc-collaboration-platform.git
cd soc-collaboration-platform
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start Infrastructure Services

```bash
docker-compose up -d
```

Verify services are running:
```bash
docker-compose ps
```

### Step 5: Initialize Services

**Terminal 1 - PSI Service:**
```bash
python psi_service/psi_server.py
```

**Terminal 2 - API Gateway:**
```bash
python api/main.py
```

**Terminal 3 - FL Server:**
```bash
python fl_service/fl_server.py
```

### Step 6: Access the Dashboard

Open your browser and navigate to:
```
http://localhost/frontend/index.html
```

## Usage

### Querying Threat Intelligence (PSI)

#### Using the Web Dashboard

1. Navigate to the "Threat Intelligence Query" section
2. Enter your organization ID
3. Select indicator type (IP, Domain, Hash, Email)
4. Enter indicators (one per line)
5. Click "Query Intersection"

#### Using Python Client

```python
from psi_service.psi_client import PSIClient

# Initialize client
client = PSIClient("http://localhost:8000", "org_alpha")

# Setup your organization's IOC database
server_iocs = [
    "192.168.1.1",
    "10.0.0.5",
    "malware_hash_abc123",
    "suspicious.domain.com"
]
client.setup_server_dataset(server_iocs)

# Query for common indicators
client_iocs = [
    "192.168.1.1",
    "different_ip",
    "malware_hash_abc123"
]
result = client.query_intersection(client_iocs, query_type="mixed")

print(f"Common indicators: {result['intersection']}")
print(f"Match count: {result['intersection_count']}")
```

#### Using cURL

```bash
# Query threat intersection
curl -X POST http://localhost:8001/api/v1/query/threat-intersection \
  -H "Content-Type: application/json" \
  -d '{
    "organization_id": "org_alpha",
    "indicators": ["192.168.1.1", "10.0.0.5"],
    "indicator_type": "ip"
  }'
```

### Federated Learning Training

#### Starting Training via Dashboard

1. Navigate to "Federated Learning Training" section
2. Enter organization ID
3. Select model type
4. Click "Start Training"
5. Monitor progress in real-time

#### Starting FL Client Programmatically

```python
from fl_service.fl_client import start_client

# Start FL client for your organization
start_client("localhost:8080", "org_alpha")
```

#### Multiple Organizations Training Together

```bash
# Terminal 1: FL Server (already running)
python fl_service/fl_server.py

# Terminal 2: Organization Alpha
python fl_service/fl_client.py org_alpha

# Terminal 3: Organization Beta
python fl_service/fl_client.py org_beta

# Terminal 4: Organization Gamma
python fl_service/fl_client.py org_gamma
```

##  API Documentation

### PSI Service Endpoints

#### Setup Organization Dataset
```
POST /psi/setup?organization_id={org_id}
Content-Type: application/json

Body: ["ioc1", "ioc2", "ioc3"]

Response: {
  "status": "success",
  "organization_id": "org_alpha"
}
```

#### Query Intersection
```
POST /psi/query
Content-Type: application/json

Body: {
  "organization_id": "org_alpha",
  "client_items": ["ioc1", "ioc2"],
  "query_type": "ip"
}

Response: {
  "intersection": ["ioc1"],
  "intersection_count": 1,
  "query_id": "uuid-here"
}
```

### API Gateway Endpoints

#### Threat Intelligence Query
```
POST /api/v1/query/threat-intersection
Content-Type: application/json

Body: {
  "organization_id": "org_alpha",
  "indicators": ["192.168.1.1"],
  "indicator_type": "ip"
}

Response: {
  "query_id": "uuid",
  "timestamp": "2026-01-18T10:30:00",
  "matches": ["192.168.1.1"],
  "match_count": 1,
  "participating_orgs": ["org_alpha", "org_beta"]
}
```

#### Start FL Training
```
POST /api/v1/training/start
Content-Type: application/json

Body: {
  "organization_id": "org_alpha",
  "model_type": "malware"
}

Response: {
  "training_id": "uuid",
  "status": "pending",
  "current_round": 0,
  "total_rounds": 5,
  "accuracy": null
}
```

#### Get Training Status
```
GET /api/v1/training/status/{training_id}

Response: {
  "training_id": "uuid",
  "status": "training",
  "current_round": 3,
  "total_rounds": 5,
  "accuracy": 0.94
}
```

#### Health Check
```
GET /health

Response: {
  "status": "healthy"
}
```

##  Development

### Project Structure

```
soc-collaboration-platform/
├── psi_service/          # Private Set Intersection service
│   ├── psi_server.py    # PSI server implementation
│   └── psi_client.py    # PSI client library
├── fl_service/           # Federated Learning service
│   ├── fl_server.py     # FL aggregation server
│   └── fl_client.py     # FL client nodes
├── api/                  # API Gateway
│   └── main.py          # FastAPI application
├── frontend/             # Web dashboard
│   └── index.html       # Single-page application
├── tests/                # Test suite
│   ├── test_psi.py      # PSI tests
│   ├── test_fl.py       # FL tests
│   └── test_integration.py  # Integration tests
├── deploy/               # Deployment configuration
│   ├── Dockerfile.*     # Service dockerfiles
│   └── docker-compose.prod.yml
└── scripts/              # Utility scripts
    ├── setup.sh         # Environment setup
    └── run_tests.sh     # Test runner
```

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Style

This project follows PEP 8 guidelines. Format code using:

```bash
black .
flake8 .
mypy .
```

### Adding New Features

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Write tests first (TDD approach)
3. Implement the feature
4. Ensure all tests pass: `./scripts/run_tests.sh`
5. Format code: `black .`
6. Commit: `git commit -m 'feat: Add amazing feature'`
7. Push and create Pull Request

##  Testing

### Running All Tests

```bash
# Using the test script
./scripts/run_tests.sh

# Using pytest directly
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=html
open htmlcov/index.html
```

### Running Specific Tests

```bash
# PSI tests only
pytest tests/test_psi.py -v

# FL tests only
pytest tests/test_fl.py -v

# Integration tests only
pytest tests/test_integration.py -v

# Run a specific test function
pytest tests/test_psi.py::test_psi_intersection -v
```

### Test Coverage

Current test coverage: **85%+**

To view detailed coverage:
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## Deployment

### Development Deployment

```bash
docker-compose up -d
```

### Production Deployment

```bash
cd deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes configurations (if you have k8s configs)
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

### Environment Variables

Create a `.env` file for production:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8001

# PSI Service
PSI_SERVICE_URL=http://psi-service:8000

# Database
DATABASE_URL=postgresql://user:pass@postgres:5432/soc_collab

# Redis
REDIS_URL=redis://redis:6379

# Security
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# FL Configuration
FL_SERVER_ADDRESS=fl-server:8080
FL_NUM_ROUNDS=10
FL_MIN_CLIENTS=3
```

### Scaling

**Horizontal Scaling:**
```bash
# Scale API gateway
docker-compose up -d --scale api-gateway=3

# Scale FL clients
docker-compose up -d --scale fl-client=5
```

**Load Balancing:**
Use NGINX or HAProxy for load balancing across API instances.


**PSI Privacy:**
- Data is hashed before comparison
- Only intersection is revealed
- Set cardinality can be protected
- No individual items are exposed

**FL Privacy:**
- Raw data never leaves organization
- Differential privacy on model updates
- Secure aggregation protocols
- Byzantine-robust aggregation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenMined](https://www.openmined.org/) for PSI library
- [Flower](https://flower.dev/) for federated learning framework
- [TensorFlow](https://www.tensorflow.org/) for ML capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for API framework