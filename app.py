"""
FastAPI Social Media Aggregator with OpenAI MCP Protocol
X (Twitter) API integration with search_tweets and fetch_tweet tools
"""

import os
import time
import hashlib
import hmac
import base64
import urllib.parse
import secrets
from typing import Optional, Any
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
X_API_KEY = os.getenv("X_API_KEY", "")
X_API_SECRET = os.getenv("X_API_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")
X_CLIENT_ID = os.getenv("X_CLIENT_ID", "")
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET", "")

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "900"))  # 15 minutes

# In-memory rate limit storage (use Redis in production)
rate_limit_store: dict[str, list[float]] = {}


# OAuth 1.0a Helper Functions
def generate_oauth_signature(
    method: str,
    url: str,
    params: dict,
    consumer_secret: str,
    token_secret: str = ""
) -> str:
    """Generate OAuth 1.0a signature"""
    # Sort and encode parameters
    sorted_params = sorted(params.items())
    param_string = "&".join(
        f"{urllib.parse.quote(str(k), safe='')}"
        f"={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted_params
    )
    
    # Create signature base string
    base_string = "&".join([
        method.upper(),
        urllib.parse.quote(url, safe=""),
        urllib.parse.quote(param_string, safe="")
    ])
    
    # Create signing key
    signing_key = f"{urllib.parse.quote(consumer_secret, safe='')}&{urllib.parse.quote(token_secret, safe='')}"
    
    # Generate signature
    signature = hmac.new(
        signing_key.encode(),
        base_string.encode(),
        hashlib.sha1
    ).digest()
    
    return base64.b64encode(signature).decode()


def get_oauth_header(method: str, url: str, params: dict = None) -> str:
    """Generate OAuth 1.0a Authorization header"""
    if params is None:
        params = {}
    
    oauth_params = {
        "oauth_consumer_key": X_API_KEY,
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": X_ACCESS_TOKEN,
        "oauth_version": "1.0"
    }
    
    # Combine OAuth params with request params for signature
    all_params = {**oauth_params, **params}
    
    # Generate signature
    signature = generate_oauth_signature(
        method, url, all_params,
        X_API_SECRET, X_ACCESS_TOKEN_SECRET
    )
    oauth_params["oauth_signature"] = signature
    
    # Build Authorization header
    auth_header = "OAuth " + ", ".join(
        f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(v, safe="")}"'
        for k, v in sorted(oauth_params.items())
    )
    
    return auth_header


# Pydantic Models for MCP Protocol
class MCPToolInput(BaseModel):
    """Input schema for MCP tool calls"""
    name: str = Field(..., description="Name of the tool to execute")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")


class MCPToolOutput(BaseModel):
    """Output schema for MCP tool responses"""
    content: list[dict] = Field(default_factory=list, description="Tool output content")
    isError: bool = Field(default=False, description="Whether an error occurred")


class SearchTweetsInput(BaseModel):
    """Input for search_tweets tool"""
    query: str = Field(..., description="Search query for tweets")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    start_time: Optional[str] = Field(default=None, description="Start time in ISO format")
    end_time: Optional[str] = Field(default=None, description="End time in ISO format")


class FetchTweetInput(BaseModel):
    """Input for fetch_tweet tool"""
    tweet_id: str = Field(..., description="ID of the tweet to fetch")


class TweetData(BaseModel):
    """Tweet data model"""
    id: str
    text: str
    author_id: Optional[str] = None
    created_at: Optional[str] = None
    public_metrics: Optional[dict] = None


# X API Client
class XAPIClient:
    """Client for X (Twitter) API v2"""
    
    BASE_URL = "https://api.twitter.com/2"
    
    def __init__(self):
        self.bearer_token = X_BEARER_TOKEN
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        use_oauth: bool = False
    ) -> dict:
        """Make authenticated request to X API"""
        url = f"{self.BASE_URL}/{endpoint}"
        
        if use_oauth and X_API_KEY and X_ACCESS_TOKEN:
            headers = {"Authorization": get_oauth_header(method, url, params or {})}
        elif self.bearer_token:
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
        else:
            raise HTTPException(
                status_code=500,
                detail="No valid X API credentials configured"
            )
        
        try:
            if method.upper() == "GET":
                response = await self.http_client.get(url, params=params, headers=headers)
            else:
                response = await self.http_client.post(url, json=params, headers=headers)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            error_detail = f"X API error: {e.response.status_code}"
            try:
                error_json = e.response.json()
                if "detail" in error_json:
                    error_detail = f"X API error: {error_json['detail']}"
                elif "errors" in error_json:
                    error_detail = f"X API error: {error_json['errors']}"
            except Exception:
                pass
            raise HTTPException(status_code=e.response.status_code, detail=error_detail)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")
    
    async def search_tweets(
        self,
        query: str,
        max_results: int = 10,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> list[dict]:
        """Search recent tweets"""
        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": "created_at,author_id,public_metrics",
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        result = await self._make_request("GET", "tweets/search/recent", params)
        return result.get("data", [])
    
    async def fetch_tweet(self, tweet_id: str) -> dict:
        """Fetch a single tweet by ID"""
        params = {
            "tweet.fields": "created_at,author_id,public_metrics,entities",
            "expansions": "author_id",
            "user.fields": "name,username,profile_image_url"
        }
        
        result = await self._make_request("GET", f"tweets/{tweet_id}", params)
        return result.get("data", {})
    
    async def close(self):
        """Close the HTTP client"""
        await self.http_client.aclose()


# Global X API client instance
x_client: Optional[XAPIClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    global x_client
    # Startup
    x_client = XAPIClient()
    print("Social Media Aggregator MCP started")
    yield
    # Shutdown
    if x_client:
        await x_client.close()
    print("Social Media Aggregator MCP shut down")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Social Media Aggregator MCP",
    description="AI-powered social media aggregator using OpenAI MCP protocol",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiter dependency
async def check_rate_limit(request: Request):
    """Rate limit check as a FastAPI dependency"""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    
    # Clean up old entries
    if client_ip in rate_limit_store:
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip]
            if current_time - t < RATE_LIMIT_WINDOW
        ]
    else:
        rate_limit_store[client_ip] = []
    
    # Check rate limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Record this request
    rate_limit_store[client_ip].append(current_time)


# MCP Tool Definitions
MCP_TOOLS = [
    {
        "name": "search_tweets",
        "description": "Search for recent tweets on X (Twitter) matching a query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for tweets (supports X search operators)"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (1-100)",
                    "default": 10
                },
                "start_time": {
                    "type": "string",
                    "description": "Start time in ISO 8601 format (optional)"
                },
                "end_time": {
                    "type": "string",
                    "description": "End time in ISO 8601 format (optional)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_tweet",
        "description": "Fetch a specific tweet by its ID from X (Twitter)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tweet_id": {
                    "type": "string",
                    "description": "The unique ID of the tweet to fetch"
                }
            },
            "required": ["tweet_id"]
        }
    }
]


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# MCP Protocol Endpoints
@app.get("/mcp/tools")
async def list_tools():
    """List available MCP tools"""
    return {"tools": MCP_TOOLS}


@app.post("/mcp/tools/call")
async def call_tool(tool_input: MCPToolInput, _: None = Depends(check_rate_limit)):
    """Execute an MCP tool"""
    tool_name = tool_input.name
    arguments = tool_input.arguments
    
    try:
        if tool_name == "search_tweets":
            query = arguments.get("query")
            if not query:
                raise ValueError("query is required")
            
            max_results = arguments.get("max_results", 10)
            start_time = arguments.get("start_time")
            end_time = arguments.get("end_time")
            
            tweets = await x_client.search_tweets(
                query=query,
                max_results=max_results,
                start_time=start_time,
                end_time=end_time
            )
            
            return MCPToolOutput(
                content=[{"type": "text", "text": str(tweets)}],
                isError=False
            )
        
        elif tool_name == "fetch_tweet":
            tweet_id = arguments.get("tweet_id")
            if not tweet_id:
                raise ValueError("tweet_id is required")
            
            tweet = await x_client.fetch_tweet(tweet_id=tweet_id)
            
            return MCPToolOutput(
                content=[{"type": "text", "text": str(tweet)}],
                isError=False
            )
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    except HTTPException:
        raise
    except Exception as e:
        return MCPToolOutput(
            content=[{"type": "text", "text": f"Error: {str(e)}"}],
            isError=True
        )


# Direct API endpoints (for non-MCP usage)
@app.get("/api/tweets/search")
async def search_tweets_api(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(default=10, ge=1, le=100),
    start_time: Optional[str] = Query(default=None),
    end_time: Optional[str] = Query(default=None),
    _: None = Depends(check_rate_limit)
):
    """Search tweets directly via REST API"""
    tweets = await x_client.search_tweets(
        query=query,
        max_results=max_results,
        start_time=start_time,
        end_time=end_time
    )
    return {"data": tweets, "count": len(tweets)}


@app.get("/api/tweets/{tweet_id}")
async def fetch_tweet_api(tweet_id: str, _: None = Depends(check_rate_limit)):
    """Fetch a single tweet by ID"""
    tweet = await x_client.fetch_tweet(tweet_id=tweet_id)
    return {"data": tweet}


# OAuth 2.0 Authorization Code Flow endpoints
@app.get("/oauth/authorize")
async def oauth_authorize(
    redirect_uri: str = Query(..., description="Callback URL"),
    state: Optional[str] = Query(default=None, description="State parameter")
):
    """Initialize OAuth 2.0 authorization flow"""
    if not X_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="OAuth client ID not configured"
        )
    
    # Generate state if not provided
    if not state:
        state = secrets.token_urlsafe(32)
    
    # Build authorization URL
    auth_url = "https://twitter.com/i/oauth2/authorize"
    params = {
        "response_type": "code",
        "client_id": X_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "tweet.read users.read offline.access",
        "state": state,
        "code_challenge": secrets.token_urlsafe(32),
        "code_challenge_method": "plain"
    }
    
    query_string = urllib.parse.urlencode(params)
    return {"authorization_url": f"{auth_url}?{query_string}", "state": state}


@app.post("/oauth/token")
async def oauth_token(
    code: str = Query(..., description="Authorization code"),
    redirect_uri: str = Query(..., description="Callback URL"),
    code_verifier: str = Query(..., description="PKCE code verifier")
):
    """Exchange authorization code for access token"""
    if not X_CLIENT_ID or not X_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="OAuth credentials not configured"
        )
    
    token_url = "https://api.twitter.com/2/oauth2/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
                "client_id": X_CLIENT_ID
            },
            auth=(X_CLIENT_ID, X_CLIENT_SECRET)
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Token exchange failed: {response.text}"
            )
        
        return response.json()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
