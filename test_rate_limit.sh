#!/bin/bash

# Get the first API key from environment
API_KEY=$(echo "$API_KEYS" | cut -d',' -f1 | xargs)

echo "Testing rate limiting with API key: ${API_KEY:0:10}..."
echo "Sending 12 requests (limit is 10 per minute)..."
echo ""

for i in {1..12}; do
  response=$(curl -s -X POST http://localhost:5000/scan \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{"file": "invalid"}')
  
  success=$(echo "$response" | python -c "import sys, json; print(json.load(sys.stdin).get('success', 'N/A'))" 2>/dev/null || echo "error")
  error_code=$(echo "$response" | python -c "import sys, json; print(json.load(sys.stdin).get('error_code', ''))" 2>/dev/null || echo "")
  
  if [ "$error_code" == "RATE_LIMIT_EXCEEDED" ]; then
    retry_after=$(echo "$response" | python -c "import sys, json; print(json.load(sys.stdin).get('retry_after', 'N/A'))" 2>/dev/null)
    echo "Request $i: RATE LIMITED (retry after ${retry_after}s)"
  elif [ "$error_code" == "INVALID_BASE64" ]; then
    echo "Request $i: ✓ Accepted (error in data, but passed auth)"
  else
    echo "Request $i: $success (error_code: $error_code)"
  fi
  
  sleep 0.2
done

echo ""
echo "Rate limiting test complete!"
