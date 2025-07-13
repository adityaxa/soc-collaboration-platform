import requests
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PSIClient:
    def __init__(self, server_url: str, organization_id: str):
        self.server_url = server_url
        self.organization_id = organization_id
    
    def query_intersection(self, ioc_list: List[str], query_type: str = 'ioc'):
        """Query PSI server for intersection"""
        try:
            response = requests.post(
                f"{self.server_url}/psi/query",
                json={
                    "organization_id": self.organization_id,
                    "client_items": ioc_list,
                    "query_type": query_type
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"PSI query failed: {str(e)}")
            raise
    
    def setup_server_dataset(self, ioc_list: List[str]):
        """Setup server with organization's IOC dataset"""
        try:
            response = requests.post(
                f"{self.server_url}/psi/setup",
                params={"organization_id": self.organization_id},
                json=ioc_list,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"PSI setup failed: {str(e)}")
            raise
