# Social Media Aggregator MCP

AI-powered social media aggregator using OpenAI MCP (Model Context Protocol) with FastAPI, X (Twitter) API integration, and Cloud Run deployment.

## Features

- **OpenAI MCP Protocol**: Implements the Model Context Protocol for AI tool integration
- **X (Twitter) API Integration**: Search tweets and fetch individual tweets
- **Rate Limiting**: Built-in rate limiting to prevent API abuse
- **OAuth Support**: OAuth 1.0a and OAuth 2.0 authorization flows
- **Cloud Run Ready**: Optimized for Google Cloud Run deployment
- **Firebase Hosting**: Firebase configuration for seamless hosting

## MCP Tools

### search_tweets
Search for recent tweets on X (Twitter) matching a query.

**Parameters:**
- `query` (required): Search query for tweets (supports X search operators)
- `max_results` (optional): Maximum number of results (1-100, default: 10)
- `start_time` (optional): Start time in ISO 8601 format
- `end_time` (optional): End time in ISO 8601 format

### fetch_tweet
Fetch a specific tweet by its ID from X (Twitter).

**Parameters:**
- `tweet_id` (required): The unique ID of the tweet to fetch

## API Endpoints

### MCP Protocol Endpoints
- `GET /mcp/tools` - List available MCP tools
- `POST /mcp/tools/call` - Execute an MCP tool

### Direct API Endpoints
- `GET /api/tweets/search?query=<query>` - Search tweets
- `GET /api/tweets/{tweet_id}` - Fetch a single tweet

### OAuth Endpoints
- `GET /oauth/authorize` - Initialize OAuth 2.0 authorization
- `POST /oauth/token` - Exchange authorization code for token

### Health Check
- `GET /health` - Health check endpoint

## Setup

### Prerequisites
- Python 3.11+
- X (Twitter) Developer Account with API credentials

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/optimize-avc/social-media-aggregator-mcp.git
cd social-media-aggregator-mcp
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create environment file:
```bash
cp .env.example .env
# Edit .env with your X API credentials
```

5. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:8080`.

### Docker

Build and run with Docker:
```bash
docker build -t social-media-aggregator-mcp .
docker run -p 8080:8080 --env-file .env social-media-aggregator-mcp
```

## Deployment

### Cloud Run

The GitHub Actions workflow automatically deploys to Cloud Run on push to `main`.

#### Required GitHub Secrets:
- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `WIF_PROVIDER`: Workload Identity Federation provider
- `WIF_SERVICE_ACCOUNT`: Service account email

#### Required Secret Manager Secrets:
- `X_API_KEY`
- `X_API_SECRET`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `X_BEARER_TOKEN`
- `X_CLIENT_ID`
- `X_CLIENT_SECRET`

### Firebase Hosting

To deploy with Firebase Hosting:

1. Install Firebase CLI:
```bash
npm install -g firebase-tools
```

2. Login to Firebase:
```bash
firebase login
```

3. Deploy:
```bash
firebase deploy --only hosting
```

## Usage Examples

### List Available Tools
```bash
curl http://localhost:8080/mcp/tools
```

### Search Tweets via MCP
```bash
curl -X POST http://localhost:8080/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_tweets",
    "arguments": {
      "query": "OpenAI MCP",
      "max_results": 10
    }
  }'
```

### Fetch Tweet via MCP
```bash
curl -X POST http://localhost:8080/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fetch_tweet",
    "arguments": {
      "tweet_id": "1234567890"
    }
  }'
```

### Direct API Search
```bash
curl "http://localhost:8080/api/tweets/search?query=AI&max_results=5"
```

## Rate Limiting

The API includes built-in rate limiting:
- Default: 100 requests per 15 minutes per IP
- Configurable via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW` environment variables

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
