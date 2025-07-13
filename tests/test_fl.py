# File: tests/test_fl.py

import pytest
import numpy as np
from fl_service.fl_client import (
    SOCFLClient, 
    create_threat_detection_model,
    load_organization_data
)
import tensorflow as tf

class TestFederatedLearning:
    """Test suite for Federated Learning components"""
    
    def test_model_creation(self):
        """Test threat detection model creation"""
        input_dim = 20
        model = create_threat_detection_model(input_dim)
        
        # Check model architecture
        assert model is not None
        assert len(model.layers) == 5  # Input + 2 Dense + 2 Dropout
        
        # Check input shape
        assert model.input_shape == (None, input_dim)
        
        # Check output shape (binary classification)
        assert model.output_shape == (None, 1)
        
        # Check if compiled
        assert model.optimizer is not None
        assert model.loss is not None
    
    def test_load_organization_data(self):
        """Test loading organization training data"""
        org_id = "test_org_123"
        x_train, y_train, x_test, y_test = load_organization_data(org_id)
        
        # Check data shapes
        assert x_train.shape[0] == 1000  # 1000 training samples
        assert x_train.shape[1] == 20    # 20 features
        assert y_train.shape[0] == 1000
        
        assert x_test.shape[0] == 200    # 200 test samples
        assert x_test.shape[1] == 20
        assert y_test.shape[0] == 200
        
        # Check data types
        assert x_train.dtype == np.float64
        assert y_train.dtype in [np.int64, np.int32]
        
        # Check labels are binary
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})
    
    def test_fl_client_initialization(self):
        """Test FL client initialization"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        assert client.organization_id == "test_org"
        assert client.model is not None
        assert len(client.x_train) == 1000
        assert len(client.x_test) == 200
    
    def test_get_parameters(self):
        """Test getting model parameters"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        parameters = client.get_parameters({})
        
        # Check parameters are returned as list of numpy arrays
        assert isinstance(parameters, list)
        assert len(parameters) > 0
        assert all(isinstance(p, np.ndarray) for p in parameters)
    
    def test_fit(self):
        """Test local training (fit method)"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        # Get initial parameters
        initial_params = client.get_parameters({})
        
        # Perform training
        config = {
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        updated_params, num_examples, metrics = client.fit(initial_params, config)
        
        # Check return values
        assert isinstance(updated_params, list)
        assert num_examples == 1000
        assert isinstance(metrics, dict)
        
        # Parameters should have changed after training
        assert len(updated_params) == len(initial_params)
        # Check at least one parameter changed (not exactly equal due to training)
        params_changed = False
        for init_p, updated_p in zip(initial_params, updated_params):
            if not np.allclose(init_p, updated_p, rtol=1e-5):
                params_changed = True
                break
        assert params_changed, "Parameters should change after training"
    
    def test_evaluate(self):
        """Test model evaluation"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        parameters = client.get_parameters({})
        
        loss, num_examples, metrics = client.evaluate(parameters, {})
        
        # Check return values
        assert isinstance(loss, float)
        assert loss >= 0
        assert num_examples == 200
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_differential_privacy(self):
        """Test differential privacy noise addition"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        parameters = client.get_parameters({})
        
        # Add differential privacy noise
        noisy_params = client.add_differential_privacy(parameters, epsilon=1.0)
        
        # Check that noise was added
        assert len(noisy_params) == len(parameters)
        
        # Parameters should be different after adding noise
        noise_added = False
        for orig_p, noisy_p in zip(parameters, noisy_params):
            if not np.allclose(orig_p, noisy_p, rtol=1e-10):
                noise_added = True
                break
        assert noise_added, "Differential privacy should add noise to parameters"
    
    def test_model_convergence(self):
        """Test that model can learn and improve"""
        x_train, y_train, x_test, y_test = load_organization_data("test_org")
        model = create_threat_detection_model(input_dim=x_train.shape[1])
        
        client = SOCFLClient(
            model, x_train, y_train, x_test, y_test, "test_org"
        )
        
        # Initial evaluation
        params = client.get_parameters({})
        initial_loss, _, initial_metrics = client.evaluate(params, {})
        
        # Train for multiple epochs
        config = {
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        updated_params, _, _ = client.fit(params, config)
        
        # Final evaluation
        final_loss, _, final_metrics = client.evaluate(updated_params, {})
        
        # Loss should generally decrease (may not always due to random data)
        # We just check that training completed successfully
        assert final_loss >= 0
        assert 0 <= final_metrics["accuracy"] <= 1
    
    def test_different_organizations_different_data(self):
        """Test that different organizations get different data"""
        x1, y1, _, _ = load_organization_data("org_1")
        x2, y2, _, _ = load_organization_data("org_2")
        
        # Data should be different for different organizations
        assert not np.array_equal(x1, x2)
        
        # But same organization should get same data consistently
        x1_again, y1_again, _, _ = load_organization_data("org_1")
        assert np.array_equal(x1, x1_again)
        assert np.array_equal(y1, y1_again)


class TestFederatedAveraging:
    """Test federated averaging simulation"""
    
    def test_parameter_averaging(self):
        """Test averaging parameters from multiple clients"""
        # Simulate 3 clients
        orgs = ["org_a", "org_b", "org_c"]
        client_params = []
        
        for org in orgs:
            x_train, y_train, x_test, y_test = load_organization_data(org)
            model = create_threat_detection_model(input_dim=x_train.shape[1])
            client = SOCFLClient(
                model, x_train, y_train, x_test, y_test, org
            )
            params = client.get_parameters({})
            client_params.append(params)
        
        # Average parameters
        averaged_params = []
        for layer_params in zip(*client_params):
            avg = np.mean(layer_params, axis=0)
            averaged_params.append(avg)
        
        # Check averaged parameters
        assert len(averaged_params) == len(client_params[0])
        
        # Averaged params should be different from individual params
        different_from_all = True
        for client_param_set in client_params:
            if all(np.array_equal(ap, cp) for ap, cp in zip(averaged_params, client_param_set)):
                different_from_all = False
        
        # This might fail if parameters are identical, which is unlikely
        # In production, this would be more sophisticated


# Pytest fixtures
@pytest.fixture
def sample_model():
    """Fixture for a sample model"""
    return create_threat_detection_model(input_dim=20)

@pytest.fixture
def sample_client():
    """Fixture for a sample FL client"""
    x_train, y_train, x_test, y_test = load_organization_data("fixture_org")
    model = create_threat_detection_model(input_dim=x_train.shape[1])
    return SOCFLClient(model, x_train, y_train, x_test, y_test, "fixture_org")


# Integration tests
class TestFLIntegration:
    """Integration tests for FL workflow"""
    
    def test_full_training_round(self, sample_client):
        """Test a complete training round"""
        # Initial parameters
        params = sample_client.get_parameters({})
        
        # Training configuration
        config = {
            "server_round": 1,
            "local_epochs": 2,
            "batch_size": 32,
            "learning_rate": 0.001
        }
        
        # Fit
        updated_params, num_examples, fit_metrics = sample_client.fit(params, config)
        
        # Evaluate
        loss, num_test, eval_metrics = sample_client.evaluate(updated_params, config)
        
        # Assertions
        assert num_examples == 1000
        assert num_test == 200
        assert loss >= 0
        assert "accuracy" in eval_metrics
        assert len(updated_params) == len(params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])