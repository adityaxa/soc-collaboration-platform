import openmined.psi as psi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import hashlib
import logging

app = FastAPI(title="PSI Service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PSIRequest(BaseModel):
    organization_id: str
    client_items: List[str]
    query_type: str

class PSIResponse(BaseModel):
    intersection: List[str]
    intersection_count: int
    query_id: str

ioc_databases = {}

def hash_item(item: str, salt: str) -> str:
    """Hash items with organization-specific salt"""
    return hashlib.sha256(f"{salt}{item}".encode()).hexdigest()

class PSIService:
    def __init__(self):
        self.client = psi.client.CreateWithNewKey(True)
        self.server = psi.server.CreateWithNewKey(True)
        
    def setup_server(self, server_items: List[str], organization_id: str):
        """Setup PSI server with organization's IOC set"""
        try:
            salt = organization_id
            hashed_items = [hash_item(item, salt) for item in server_items]
            
            server_setup = self.server.CreateSetupMessage(
                0.001,
                len(hashed_items),
                hashed_items
            )
            
            ioc_databases[organization_id] = {
                'setup': server_setup,
                'items': set(hashed_items),
                'original_items': set(server_items)
            }
            
            logger.info(f"PSI server setup complete for {organization_id}")
            return server_setup
            
        except Exception as e:
            logger.error(f"PSI setup error: {str(e)}")
            raise
    
    def process_request(self, client_items: List[str], 
                       server_org_id: str, 
                       client_org_id: str) -> List[str]:
        """Process PSI request and return intersection"""
        try:
            if server_org_id not in ioc_databases:
                raise ValueError(f"Organization {server_org_id} not found")
            
            salt = server_org_id
            hashed_client_items = [hash_item(item, salt) for item in client_items]
            
            client_request = self.client.CreateRequest(hashed_client_items)
            
            server_response = self.server.ProcessRequest(
                ioc_databases[server_org_id]['setup'],
                client_request
            )
            
            intersection_hashed = self.client.GetIntersection(
                ioc_databases[server_org_id]['setup'],
                server_response
            )
            
            intersection = []
            for item in client_items:
                if hash_item(item, salt) in intersection_hashed:
                    intersection.append(item)
            
            logger.info(f"PSI query: {client_org_id} -> {server_org_id}, "
                       f"found {len(intersection)} matches")
            
            return intersection
            
        except Exception as e:
            logger.error(f"PSI processing error: {str(e)}")
            raise

psi_service = PSIService()

@app.post("/psi/setup")
async def setup_psi(organization_id: str, ioc_list: List[str]):
    """Setup PSI server for an organization"""
    try:
        psi_service.setup_server(ioc_list, organization_id)
        return {"status": "success", "organization_id": organization_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/psi/query", response_model=PSIResponse)
async def query_psi(request: PSIRequest):
    """Query PSI intersection"""
    try:
        intersection = psi_service.process_request(
            request.client_items,
            "server_org",
            request.organization_id
        )
        
        import uuid
        return PSIResponse(
            intersection=intersection,
            intersection_count=len(intersection),
            query_id=str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "psi"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
