import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SOCFLClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, organization_id: str):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.organization_id = organization_id
    
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return self.model.get_weights()
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple:
        logger.info(f"[{self.organization_id}] Starting local training")
        
        self.model.set_weights(parameters)
        
        epochs = config.get("local_epochs", 3)
        batch_size = config.get("batch_size", 32)
        
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        parameters = self.model.get_weights()
        parameters = self.add_differential_privacy(parameters, epsilon=1.0)
        
        logger.info(f"[{self.organization_id}] Training complete")
        
        return parameters, len(self.x_train), {}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        logger.info(f"[{self.organization_id}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return loss, len(self.x_test), {"accuracy": accuracy}
    
    def add_differential_privacy(self, parameters: List[np.ndarray], epsilon: float = 1.0):
        """Add Gaussian noise for differential privacy"""
        sensitivity = 0.1
        sigma = sensitivity / epsilon
        
        noisy_parameters = []
        for param in parameters:
            noise = np.random.normal(0, sigma, param.shape)
            noisy_parameters.append(param + noise)
        
        return noisy_parameters

def create_threat_detection_model(input_dim: int) -> keras.Model:
    """Create neural network for threat detection"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_organization_data(organization_id: str) -> Tuple:
    """Load organization's security data"""
    np.random.seed(hash(organization_id) % 2**32)
    
    n_samples = 1000
    n_features = 20
    
    x_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, n_samples)
    
    x_test = np.random.randn(200, n_features)
    y_test = np.random.randint(0, 2, 200)
    
    logger.info(f"Loaded data for {organization_id}")
    
    return x_train, y_train, x_test, y_test

def start_client(server_address: str, organization_id: str):
    """Start FL client"""
    logger.info(f"Starting FL client for {organization_id}")
    
    x_train, y_train, x_test, y_test = load_organization_data(organization_id)
    model = create_threat_detection_model(input_dim=x_train.shape[1])
    client = SOCFLClient(model, x_train, y_train, x_test, y_test, organization_id)
    
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fl_client.py <organization_id>")
        sys.exit(1)
    
    org_id = sys.argv[1]
    start_client("localhost:8080", org_id)
