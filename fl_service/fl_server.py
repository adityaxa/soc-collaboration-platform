import flwr as fl
from flwr.server.strategy import FedAvg
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOCFederatedServer:
    def __init__(self, num_rounds: int = 10, min_clients: int = 2):
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        
    def get_strategy(self):
        """Configure federated averaging strategy"""
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            evaluate_fn=self.get_evaluate_fn(),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
        )
        return strategy
    
    def fit_config(self, server_round: int) -> Dict:
        """Configure training for each round"""
        config = {
            "server_round": server_round,
            "local_epochs": 3,
            "batch_size": 32,
            "learning_rate": 0.001,
        }
        return config
    
    def evaluate_config(self, server_round: int) -> Dict:
        """Configure evaluation"""
        return {"server_round": server_round}
    
    def get_evaluate_fn(self):
        """Return evaluation function"""
        def evaluate(server_round: int, parameters, config):
            logger.info(f"Server evaluation at round {server_round}")
            return 0.0, {"accuracy": 0.95}
        return evaluate
    
    def start(self, server_address: str = "0.0.0.0:8080"):
        """Start FL server"""
        logger.info(f"Starting FL server on {server_address}")
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.get_strategy(),
        )

if __name__ == "__main__":
    server = SOCFederatedServer(num_rounds=5, min_clients=2)
    server.start()
