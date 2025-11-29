"""
Tests for Social Media Aggregator MCP
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from app import app, XAPIClient


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_x_client():
    """Create a mock X client"""
    mock_client = MagicMock(spec=XAPIClient)
    mock_client.search_tweets = AsyncMock(return_value=[])
    mock_client.fetch_tweet = AsyncMock(return_value={})
    mock_client.close = AsyncMock()
    return mock_client


class TestHealthCheck:
    """Tests for health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestMCPTools:
    """Tests for MCP tool endpoints"""
    
    def test_list_tools(self, client):
        """Test listing available tools"""
        response = client.get("/mcp/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        tools = data["tools"]
        assert len(tools) == 2
        
        tool_names = [t["name"] for t in tools]
        assert "search_tweets" in tool_names
        assert "fetch_tweet" in tool_names
    
    def test_search_tweets_tool_schema(self, client):
        """Test search_tweets tool has correct schema"""
        response = client.get("/mcp/tools")
        data = response.json()
        
        search_tool = next(t for t in data["tools"] if t["name"] == "search_tweets")
        assert "description" in search_tool
        assert "inputSchema" in search_tool
        assert "query" in search_tool["inputSchema"]["properties"]
        assert "query" in search_tool["inputSchema"]["required"]
    
    def test_fetch_tweet_tool_schema(self, client):
        """Test fetch_tweet tool has correct schema"""
        response = client.get("/mcp/tools")
        data = response.json()
        
        fetch_tool = next(t for t in data["tools"] if t["name"] == "fetch_tweet")
        assert "description" in fetch_tool
        assert "inputSchema" in fetch_tool
        assert "tweet_id" in fetch_tool["inputSchema"]["properties"]
        assert "tweet_id" in fetch_tool["inputSchema"]["required"]
    
    def test_call_unknown_tool(self, client):
        """Test calling unknown tool returns error"""
        response = client.post(
            "/mcp/tools/call",
            json={"name": "unknown_tool", "arguments": {}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["isError"] is True
        assert "Unknown tool" in data["content"][0]["text"]
    
    def test_search_tweets_missing_query(self, client):
        """Test search_tweets without query returns error"""
        response = client.post(
            "/mcp/tools/call",
            json={"name": "search_tweets", "arguments": {}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["isError"] is True
        assert "query is required" in data["content"][0]["text"]
    
    def test_fetch_tweet_missing_id(self, client):
        """Test fetch_tweet without tweet_id returns error"""
        response = client.post(
            "/mcp/tools/call",
            json={"name": "fetch_tweet", "arguments": {}}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["isError"] is True
        assert "tweet_id is required" in data["content"][0]["text"]

    def test_search_tweets_success(self, client, mock_x_client):
        """Test successful search_tweets call"""
        mock_x_client.search_tweets.return_value = [
            {"id": "1", "text": "Test tweet 1"},
            {"id": "2", "text": "Test tweet 2"}
        ]
        
        with patch("app.x_client", mock_x_client):
            response = client.post(
                "/mcp/tools/call",
                json={
                    "name": "search_tweets",
                    "arguments": {"query": "test", "max_results": 5}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["isError"] is False
            mock_x_client.search_tweets.assert_called_once()

    def test_fetch_tweet_success(self, client, mock_x_client):
        """Test successful fetch_tweet call"""
        mock_x_client.fetch_tweet.return_value = {"id": "123", "text": "Test tweet"}
        
        with patch("app.x_client", mock_x_client):
            response = client.post(
                "/mcp/tools/call",
                json={
                    "name": "fetch_tweet",
                    "arguments": {"tweet_id": "123"}
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["isError"] is False
            mock_x_client.fetch_tweet.assert_called_once_with(tweet_id="123")


class TestDirectAPI:
    """Tests for direct API endpoints"""
    
    def test_search_tweets_api(self, client, mock_x_client):
        """Test direct search tweets API"""
        mock_x_client.search_tweets.return_value = [{"id": "1", "text": "Test"}]
        
        with patch("app.x_client", mock_x_client):
            response = client.get("/api/tweets/search?query=test")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "count" in data

    def test_fetch_tweet_api(self, client, mock_x_client):
        """Test direct fetch tweet API"""
        mock_x_client.fetch_tweet.return_value = {"id": "123", "text": "Test"}
        
        with patch("app.x_client", mock_x_client):
            response = client.get("/api/tweets/123")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data


class TestOAuth:
    """Tests for OAuth endpoints"""
    
    def test_oauth_authorize_missing_client_id(self, client):
        """Test OAuth authorize fails without client ID"""
        with patch("app.X_CLIENT_ID", ""):
            response = client.get("/oauth/authorize?redirect_uri=http://example.com")
            assert response.status_code == 500
    
    def test_oauth_token_missing_credentials(self, client):
        """Test OAuth token fails without credentials"""
        with patch("app.X_CLIENT_ID", ""), patch("app.X_CLIENT_SECRET", ""):
            response = client.post(
                "/oauth/token?code=test&redirect_uri=http://example.com&code_verifier=test"
            )
            assert response.status_code == 500


class TestRateLimiting:
    """Tests for rate limiting"""
    
    def test_rate_limit_headers_not_exceeded(self, client):
        """Test normal requests work within rate limit"""
        # Reset rate limit store
        from app import rate_limit_store
        rate_limit_store.clear()
        
        response = client.get("/mcp/tools")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
