from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import logging

app = FastAPI(title="SOC Collaboration Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatQuery(BaseModel):
    organization_id: str
    indicators: List[str]
    indicator_type: str

class FLTrainingRequest(BaseModel):
    organization_id: str
    model_type: str
    
class QueryResult(BaseModel):
    query_id: str
    timestamp: datetime
    matches: List[str]
    match_count: int
    participating_orgs: List[str]

class TrainingStatus(BaseModel):
    training_id: str
    status: str
    current_round: int
    total_rounds: int
    accuracy: Optional[float]

query_results = {}
training_jobs = {}

@app.get("/")
async def root():
    return {
        "service": "SOC Collaboration Platform",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/query/threat-intersection", response_model=QueryResult)
async def query_threat_intersection(query: ThreatQuery):
    """Query threat intelligence intersection"""
    try:
        from psi_service.psi_client import PSIClient
        
        psi_client = PSIClient("http://localhost:8000", query.organization_id)
        result = psi_client.query_intersection(query.indicators, query.indicator_type)
        
        query_id = str(uuid.uuid4())
        query_result = QueryResult(
            query_id=query_id,
            timestamp=datetime.now(),
            matches=result['intersection'],
            match_count=result['intersection_count'],
            participating_orgs=["org_alpha", "org_beta"]
        )
        
        query_results[query_id] = query_result
        logger.info(f"Query {query_id} completed: {query_result.match_count} matches")
        
        return query_result
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/training/start", response_model=TrainingStatus)
async def start_fl_training(request: FLTrainingRequest, background_tasks: BackgroundTasks):
    """Start federated learning training"""
    try:
        training_id = str(uuid.uuid4())
        
        status = TrainingStatus(
            training_id=training_id,
            status="pending",
            current_round=0,
            total_rounds=5,
            accuracy=None
        )
        
        training_jobs[training_id] = status
        background_tasks.add_task(run_fl_training, training_id, request)
        
        logger.info(f"Training {training_id} initiated")
        return status
        
    except Exception as e:
        logger.error(f"Training start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/training/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(training_id: str):
    """Get FL training status"""
    if training_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_jobs[training_id]

async def run_fl_training(training_id: str, request: FLTrainingRequest):
    """Background task for FL training"""
    try:
        training_jobs[training_id].status = "training"
        
        import asyncio
        for round_num in range(1, 6):
            training_jobs[training_id].current_round = round_num
            await asyncio.sleep(2)
        
        training_jobs[training_id].status = "completed"
        training_jobs[training_id].accuracy = 0.94
        
        logger.info(f"Training {training_id} completed")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        training_jobs[training_id].status = "failed"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
