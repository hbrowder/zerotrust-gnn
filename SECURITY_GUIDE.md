# Zero-Trust Security Guide for ZeroTrustGNN API

## Overview

The ZeroTrustGNN API implements **zero-trust security principles** with API key authentication and rate limiting to protect your network analysis service from unauthorized access and abuse.

## Security Features

### 🔐 API Key Authentication

**What it protects:**
- Prevents unauthorized access to your API
- Ensures only your Adalo/Glide apps can submit PCAP files
- Allows tracking and auditing of API usage per key

**How it works:**
- Every request to `/scan` requires an `X-API-Key` header
- Keys are validated against your configured API keys in Replit Secrets
- Invalid or missing keys receive immediate 401/403 responses
- Public endpoints (`/health`, `/`) remain open for monitoring

**Response codes:**
- `401 Unauthorized`: Missing API key
- `403 Forbidden`: Invalid API key
- `200 OK`: Valid key and successful request

### ⏱️ Rate Limiting

**What it protects:**
- Prevents API abuse and DoS attacks
- Ensures fair usage across multiple clients
- Protects server resources from overload

**Default limits:**
- **10 requests per minute** per API key
- Sliding window algorithm (tracks last 60 seconds)
- Each API key has independent rate limit tracking

**How it works:**
- Tracks timestamps of requests per API key
- Returns `429 Too Many Requests` when limit exceeded
- Includes `retry_after` seconds in error response
- Response headers show remaining quota:
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Requests remaining in current window

**Response when rate limited:**
```json
{
  "success": false,
  "error": "Rate limit exceeded. Try again in 45 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

### 🔒 Additional Security Measures

1. **Input Validation**
   - Base64 decoding validation
   - PCAP file format verification
   - Risk threshold coercion (0-100 range)

2. **Error Handling**
   - Structured error responses with error codes
   - No sensitive data in error messages
   - No stack traces exposed to clients

3. **Temporary File Cleanup**
   - PCAP files deleted immediately after processing
   - Cleanup guaranteed even on errors (try/finally blocks)

4. **Production Hardening**
   - Debug mode disabled (`debug=False`)
   - CORS configured for cross-origin requests
   - Secure secrets management via Replit

## Configuration

### Setting Up API Keys

API keys are stored in **Replit Secrets** (never hardcoded in code):

1. **Navigate to Secrets:**
   - Click "Secrets" in the left sidebar
   - Or use the Tools menu

2. **Add API_KEYS secret:**
   - Key: `API_KEYS`
   - Value: Comma-separated list of keys
   - Example: `ztgnn_adalo_key,ztgnn_glide_key,ztgnn_test_key`

3. **Generate Strong Keys:**
   ```bash
   # Generate random API key
   openssl rand -hex 32
   
   # Or use a memorable format
   ztgnn_production_$(openssl rand -hex 8)
   ```

### Rate Limit Configuration

Customize rate limits via environment variables:

```bash
# In Replit Secrets, add:
RATE_LIMIT_REQUESTS=10      # Max requests per window
RATE_LIMIT_WINDOW=1         # Window size in minutes
```

**Example configurations:**

| Use Case | Requests | Window | Setting |
|----------|----------|--------|---------|
| Production (default) | 10 | 1 min | `10/1` |
| Development/Testing | 100 | 1 min | `100/1` |
| High-volume API | 60 | 1 min | `60/1` |
| Strict rate limiting | 5 | 1 min | `5/1` |

## Usage Examples

### Successful Request (Adalo)

```javascript
// Adalo Custom Action
{
  "method": "POST",
  "url": "https://your-app.repl.co/scan",
  "headers": {
    "X-API-Key": "ztgnn_adalo_key",
    "Content-Type": "application/json"
  },
  "body": {
    "file": "[Magic Text: Uploaded File]",
    "risk_threshold": 50
  }
}
```

### Successful Request (curl)

```bash
# Test with valid API key
curl -X POST https://your-app.repl.co/scan \
  -H "X-API-Key: ztgnn_production_key" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "'"$(base64 -w0 capture.pcap)"'",
    "risk_threshold": 50
  }'
```

### Error: Missing API Key

```bash
curl -X POST https://your-app.repl.co/scan \
  -H "Content-Type: application/json" \
  -d '{"file": "..."}'

# Response (401):
{
  "success": false,
  "error": "Missing API key. Include X-API-Key header in your request.",
  "error_code": "MISSING_API_KEY"
}
```

### Error: Invalid API Key

```bash
curl -X POST https://your-app.repl.co/scan \
  -H "X-API-Key: wrong_key" \
  -H "Content-Type: application/json" \
  -d '{"file": "..."}'

# Response (403):
{
  "success": false,
  "error": "Invalid API key",
  "error_code": "INVALID_API_KEY"
}
```

### Error: Rate Limit Exceeded

```bash
# After 10 requests in 1 minute
curl -X POST https://your-app.repl.co/scan \
  -H "X-API-Key: ztgnn_production_key" \
  -H "Content-Type: application/json" \
  -d '{"file": "..."}'

# Response (429):
{
  "success": false,
  "error": "Rate limit exceeded. Try again in 45 seconds.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 45
}
```

## Integration with No-Code Platforms

### Adalo Configuration

1. **Add Custom Action:**
   - Action name: "Scan Network Traffic"
   - Method: POST
   - URL: `https://your-app.repl.co/scan`

2. **Configure Headers:**
   ```
   X-API-Key: ztgnn_adalo_key
   Content-Type: application/json
   ```

3. **Add Body Parameters:**
   - `file`: Logged In User > Uploaded File (Base64)
   - `risk_threshold`: 50 (or user input)

4. **Handle Rate Limiting:**
   - Show error message on 429 response
   - Display "retry_after" seconds to user
   - Implement exponential backoff if retrying

### Glide Configuration

**Option 1: Via Make/Zapier (Recommended)**

```
Glide → Webhook → Make/Zapier → Add X-API-Key header → ZeroTrustGNN API
```

**Option 2: Direct API Call (if supported)**

1. Use "Call API" action
2. Set URL: `https://your-app.repl.co/scan`
3. Add custom header: `X-API-Key: ztgnn_glide_key`
4. Send JSON body with Base64 file

## Monitoring and Auditing

### Check API Status

```bash
# Health check (no auth required)
curl https://your-app.repl.co/health

# Response:
{
  "status": "healthy",
  "model": "gnn_model_calibrated.onnx",
  "timestamp": "2025-11-19T18:30:00.000Z"
}
```

### View API Documentation

```bash
# API docs (no auth required)
curl https://your-app.repl.co/

# Returns full API specification including:
# - Authentication requirements
# - Endpoint descriptions
# - Input/output formats
# - Integration guides
```

### Monitor Rate Limit Headers

Every successful response includes rate limit information:

```bash
curl -I -X POST https://your-app.repl.co/scan \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"file": "..."}'

# Headers:
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
```

## Best Practices

### 1. Key Management

✅ **DO:**
- Store keys in Replit Secrets
- Use different keys for different apps (Adalo, Glide, testing)
- Rotate keys periodically
- Use long, random keys (32+ characters)

❌ **DON'T:**
- Hardcode keys in code
- Share keys publicly (GitHub, forums)
- Use simple/guessable keys
- Reuse keys across projects

### 2. Rate Limit Handling

✅ **DO:**
- Respect rate limits in client code
- Implement exponential backoff on 429 errors
- Show "retry_after" time to users
- Cache results when possible

❌ **DON'T:**
- Spam API with rapid requests
- Ignore 429 responses
- Create multiple keys to bypass limits
- Retry immediately on rate limit

### 3. Error Handling

✅ **DO:**
- Check response status codes
- Parse error_code field for specific errors
- Show user-friendly error messages
- Log errors for debugging

❌ **DON'T:**
- Assume all requests succeed
- Expose API keys in error logs
- Show raw error messages to end users
- Ignore authentication failures

## Security Checklist

Before deploying to production:

- [ ] API keys configured in Replit Secrets
- [ ] Keys are long and random (32+ characters)
- [ ] Different keys for different clients
- [ ] Debug mode disabled (`debug=False`)
- [ ] HTTPS enabled (automatic on Replit deployments)
- [ ] Rate limits configured appropriately
- [ ] Error messages don't expose sensitive data
- [ ] Temporary files cleaned up after processing
- [ ] Health endpoint accessible for monitoring
- [ ] Client apps configured with correct API keys
- [ ] Rate limit handling implemented in clients
- [ ] Tested authentication failures (401, 403, 429)

## Troubleshooting

### "Missing API key" errors

**Problem:** Requests return 401 with `MISSING_API_KEY`

**Solution:**
1. Verify `X-API-Key` header is included
2. Check header name spelling (case-sensitive)
3. Ensure header value is not empty

### "Invalid API key" errors

**Problem:** Requests return 403 with `INVALID_API_KEY`

**Solution:**
1. Verify API key matches one in `API_KEYS` secret
2. Check for extra spaces or newlines in key
3. Confirm API_KEYS secret is set in Replit
4. Restart workflow after adding/changing keys

### Rate limit issues

**Problem:** Getting 429 responses too frequently

**Solution:**
1. Check `retry_after` value in response
2. Implement request queuing in client
3. Increase rate limits via environment variables
4. Use separate API keys for different apps
5. Cache results to reduce redundant requests

### API keys not working after restart

**Problem:** Authentication fails after workflow restart

**Solution:**
1. Verify `API_KEYS` secret still exists
2. Check secret format (comma-separated)
3. Look for startup errors in workflow logs
4. Restart workflow manually if needed

## Future Enhancements

Potential security improvements:

1. **OAuth 2.0 Integration**
   - Replace API keys with OAuth tokens
   - Integrate with Adalo/Glide user authentication

2. **JWT Tokens**
   - Short-lived tokens with expiration
   - Refresh token mechanism

3. **Per-Endpoint Rate Limits**
   - Different limits for different operations
   - Burst allowances for occasional spikes

4. **IP Whitelisting**
   - Restrict access to known IP ranges
   - Additional layer beyond API keys

5. **Request Logging**
   - Track all API requests
   - Audit trail for compliance

6. **Anomaly Detection on API Usage**
   - Detect unusual request patterns
   - Alert on potential abuse

---

**Current Status:** ✅ Zero-trust security implemented with API key authentication and rate limiting. Production-ready for Adalo/Glide integration.
