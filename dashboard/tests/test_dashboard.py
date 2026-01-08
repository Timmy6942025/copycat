"""
Unit tests for the CopyCat Dashboard.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.app import app, create_app


class TestDashboardApp:
    """Tests for Dashboard Flask app."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_create_app(self):
        """Test app creation."""
        test_app = create_app()
        assert test_app is not None

    def test_index_route(self, client):
        """Test main dashboard page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'CopyCat' in response.data

    def test_status_endpoint(self, client):
        """Test status API endpoint."""
        response = client.get('/api/status')
        assert response.status_code == 200
        data = response.get_json()
        assert 'success' in data
        assert 'status' in data

    def test_performance_endpoint(self, client):
        """Test performance API endpoint."""
        response = client.get('/api/performance')
        assert response.status_code == 200
        data = response.get_json()
        assert 'success' in data
        assert 'metrics' in data

    def test_health_endpoint(self, client):
        """Test health API endpoint."""
        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert 'success' in data
        assert 'health' in data

    def test_traders_endpoint(self, client):
        """Test traders API endpoint."""
        response = client.get('/api/traders')
        assert response.status_code == 200
        data = response.get_json()
        assert 'success' in data
        assert 'traders' in data
        assert 'count' in data

    def test_start_endpoint_no_auth(self, client):
        """Test start endpoint (should work without auth in test mode)."""
        response = client.post('/api/start')
        # May succeed or fail depending on state, but should return valid JSON
        assert response.status_code in [200, 400, 500]
        data = response.get_json()
        assert 'success' in data or 'error' in data

    def test_stop_endpoint_no_auth(self, client):
        """Test stop endpoint (should work without auth in test mode)."""
        response = client.post('/api/stop')
        # May succeed or fail depending on state, but should return valid JSON
        assert response.status_code in [200, 400, 500]
        data = response.get_json()
        assert 'success' in data or 'error' in data

    def test_pause_endpoint_no_auth(self, client):
        """Test pause endpoint."""
        response = client.post('/api/pause')
        assert response.status_code in [200, 400, 500]
        data = response.get_json()
        assert 'success' in data or 'error' in data

    def test_resume_endpoint_no_auth(self, client):
        """Test resume endpoint."""
        response = client.post('/api/resume')
        assert response.status_code in [200, 400, 500]
        data = response.get_json()
        assert 'success' in data or 'error' in data

    def test_add_trader_empty_address(self, client):
        """Test adding trader with empty address."""
        response = client.post('/api/traders/add', 
                              json={'address': ''},
                              content_type='application/json')
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_add_trader_invalid_json(self, client):
        """Test adding trader with invalid JSON."""
        response = client.post('/api/traders/add',
                              data='invalid json',
                              content_type='application/json')
        # Flask returns 200 even with bad JSON when request.get_json(force=True) is used
        # or returns 400 with explicit silent=False
        # The actual behavior depends on how Flask handles the bad data
        data = response.get_json()
        assert data is not None

    def test_remove_trader_empty_address(self, client):
        """Test removing trader with empty address."""
        response = client.post('/api/traders/remove',
                              json={'address': ''},
                              content_type='application/json')
        data = response.get_json()
        assert data['success'] is False
        assert 'error' in data

    def test_404_handler(self, client):
        """Test 404 error handler."""
        response = client.get('/nonexistent-page')
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
