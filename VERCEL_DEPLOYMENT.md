# Vercel Deployment Guide for ZeroTrustGNN

## Overview

ZeroTrustGNN can be deployed to Vercel for free hosting with the following architecture:
- **Frontend**: React app on Vercel (free tier)
- **Backend**: Flask API as Vercel serverless functions (free tier)
- **Analytics**: Mixpanel (free tier - 1M events/month)

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Mixpanel Account**: Sign up at [mixpanel.com](https://mixpanel.com)
3. **GitHub Account**: For automatic deployments

## Step 1: Prepare Environment Variables

### Backend Environment Variables (Vercel Project Settings)

```bash
# Required for GDPR Compliance
ANONYMIZE_DATA=true
ANONYMIZATION_SALT=<your-secure-32-char-random-string>

# Optional
DATA_RETENTION_DAYS=30

# API Keys (configure in Replit Secrets or Vercel Environment Variables)
API_KEYS=<your-api-keys-comma-separated>
SESSION_SECRET=<your-session-secret>
```

### Frontend Environment Variables (.env in frontend/)

```bash
# Production API URL (after deploying backend)
VITE_API_BASE_URL=https://your-backend.vercel.app

# API Key
VITE_API_KEY=<one-of-your-api-keys>

# Mixpanel Token (from Mixpanel project settings)
VITE_MIXPANEL_TOKEN=<your-mixpanel-project-token>
```

## Step 2: Set Up Mixpanel

1. Go to [mixpanel.com](https://mixpanel.com) and create a free account
2. Create a new project (e.g., "ZeroTrustGNN")
3. Copy your **Project Token** from Settings → Project Settings
4. Add the token to `frontend/.env`:
   ```bash
   VITE_MIXPANEL_TOKEN=your_token_here
   ```

**Mixpanel Free Tier Includes:**
- 1M events per month
- Unlimited reports (max 5 saved per user)
- 20K monthly session replays
- Unlimited data history

**Events Tracked:**
- PCAP File Uploaded
- Scan Completed
- Alert Clicked
- GDPR Consent Given/Declined
- Data Deletion Requested

## Step 3: Deploy to Vercel

### Option A: Deploy via Vercel Dashboard (Recommended)

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push
   ```

2. **Import to Vercel**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Click "Import Git Repository"
   - Select your GitHub repository
   - Framework Preset: Vercel will auto-detect

3. **Configure Build Settings**
   - **Root Directory**: Leave blank (or `.`)
   - **Build Command**: `cd frontend && npm run build`
   - **Output Directory**: `frontend/dist`
   - **Install Command**: `cd frontend && npm install`

4. **Add Environment Variables**
   - Go to Project Settings → Environment Variables
   - Add all variables from Step 1 (both backend and frontend)
   - Make sure to select "Production", "Preview", and "Development"

5. **Deploy**
   - Click "Deploy"
   - Wait 2-3 minutes for build to complete
   - Your app will be live at `https://your-project.vercel.app`

### Option B: Deploy via Vercel CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

4. **Add Environment Variables**
   ```bash
   vercel env add ANONYMIZE_DATA production
   vercel env add VITE_MIXPANEL_TOKEN production
   # ... add all other variables
   ```

5. **Redeploy**
   ```bash
   vercel --prod
   ```

## Step 4: Configure Custom Domain (Optional)

1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Wait for SSL certificate to be automatically provisioned

## Step 5: Set Up GDPR Data Cleanup (Free Tier)

Since Vercel Cron requires a Pro plan ($20/month), we'll use a free external cron service for GDPR data cleanup:

**Option A: cron-job.org (Recommended)**

1. Go to [cron-job.org](https://cron-job.org) and create a free account
2. Create a new cron job:
   - **URL**: `https://your-backend.vercel.app/api/cleanup`
   - **Schedule**: Daily at 02:00 (your timezone)
   - **HTTP Method**: GET or POST
   - **Title**: "ZeroTrustGNN GDPR Cleanup"
3. Save and enable the job
4. The cleanup endpoint (`/api/cleanup.py`) is already configured in your deployment

**Option B: EasyCron**

1. Go to [easycron.com](https://www.easycron.com)
2. Free tier: 1 cron job, runs once per day
3. Configure similar to cron-job.org above

**Option C: Manual Cleanup**

If you prefer manual control:
```bash
curl -X POST https://your-backend.vercel.app/api/cleanup
```

**Verification:**

Test the cleanup endpoint after deployment:
```bash
curl https://your-backend.vercel.app/api/cleanup
```

Expected response:
```json
{
  "success": true,
  "result": {...},
  "timestamp": "..."
}
```

## Step 6: Monitor Your Deployment

### Vercel Dashboard
- **Deployments**: View build logs and deployment history
- **Analytics**: Basic traffic metrics (Vercel Analytics)
- **Logs**: Real-time function logs

### Mixpanel Dashboard
- **Events**: Track user interactions
- **Funnels**: Analyze conversion rates
- **Retention**: Monitor user engagement

## Architecture Diagram

```
User Browser
    │
    ├─→ React Frontend (Vercel CDN)
    │       ├─→ Mixpanel Analytics
    │       └─→ API Calls
    │
    └─→ Flask Backend (Vercel Serverless)
            ├─→ ONNX Model Inference
            ├─→ GDPR Endpoints
            └─→ Security Middleware
```

## Vercel Free Tier Limits

- **Bandwidth**: 100 GB/month
- **Build Time**: 6,000 minutes/month
- **Serverless Function Execution**: 100 GB-Hrs
- **Function Duration**: 10 seconds max
- **Edge Functions**: 500,000 requests/month
- **Deployments**: Unlimited

## Troubleshooting

### Issue: Build Fails with Python Error
**Solution**: Vercel uses Python 3.9 by default. Check your `requirements.txt` for compatibility.

### Issue: CORS Errors
**Solution**: Ensure `flask-cors` is installed and `CORS(app)` is configured in `api_server.py`.

### Issue: Environment Variables Not Working
**Solution**: 
- Check they're added in Vercel Dashboard → Settings → Environment Variables
- Redeploy after adding variables
- Use `VITE_` prefix for frontend variables

### Issue: 404 on API Routes
**Solution**: Verify `vercel.json` routing configuration is correct.

### Issue: Cold Start Delays
**Solution**: This is normal for serverless functions. First request may be slower.

## Cost Estimate (All Free Tiers)

| Service | Free Tier | Cost After Free |
|---------|-----------|----------------|
| **Vercel** | 100 GB bandwidth, 6000 min build | $20/month Pro |
| **Mixpanel** | 1M events/month | $140/month Growth |
| **Total** | **$0/month** | $160/month |

## Production Checklist

- [ ] Mixpanel token configured
- [ ] All environment variables set in Vercel
- [ ] `ANONYMIZE_DATA=true` for GDPR compliance
- [ ] Strong `ANONYMIZATION_SALT` generated
- [ ] API keys configured and secure
- [ ] Custom domain configured (optional)
- [ ] SSL certificate auto-provisioned
- [ ] Cron job set up for data cleanup (or external cron)
- [ ] Test file upload and scanning
- [ ] Test GDPR consent flow
- [ ] Monitor Mixpanel events
- [ ] Check Vercel function logs

## Next Steps

1. **Monitor Usage**: Keep an eye on Vercel and Mixpanel dashboards
2. **Optimize Performance**: Use Vercel Analytics to identify bottlenecks
3. **Scale**: Upgrade to paid tiers as usage grows
4. **Backup**: Export Mixpanel data regularly

## Support

- **Vercel Docs**: https://vercel.com/docs
- **Mixpanel Docs**: https://docs.mixpanel.com
- **ZeroTrustGNN Issues**: Create an issue in your GitHub repository
