
import pytest
from app import app

@pytest.fixture
def client():
    # Creates a test client for the Flask app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page_get(client):
    """Test GET request to home page"""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment Analysis with Hugging Face" in response.data

def test_home_page_post(client):
    """Test POST request with some text"""
    response = client.post("/", data={"user_input": "I love Hugging Face!"})
    assert response.status_code == 200
    assert b"Result:" in response.data
    assert b"POSITIVE" in response.data
    print(response.data)