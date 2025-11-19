# 🚀 Vercel Deployment Checklist

Your ZeroTrustGNN app is ready for deployment! Follow these steps:

## ✅ Pre-Deployment (Already Complete)

- [x] Vercel serverless backend configured (`/api/index.py`, `/api/cleanup.py`)
- [x] `vercel.json` routing configured
- [x] Frontend build script ready (`npm run build`)
- [x] Requirements.txt created for Python dependencies
- [x] `.vercelignore` configured
- [x] GDPR-compliant Mixpanel analytics integrated
- [x] Zero TypeScript errors
- [x] Both workflows running successfully

## 📦 Step 1: Push to GitHub

```bash
# If you haven't already, initialize git and push to GitHub
git init
git add .
git commit -m "Production-ready ZeroTrustGNN with Vercel deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## 🌐 Step 2: Deploy to Vercel

### Option A: Vercel Dashboard (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign up/login
2. Click **"Add New Project"**
3. Import your GitHub repository
4. Vercel will auto-detect the configuration from `vercel.json`
5. Click **"Deploy"**

### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from project root
vercel

# Follow the prompts:
# - Link to existing project? No
# - What's your project name? zerotrust-gnn
# - In which directory is your code? ./
# - Deploy? Yes
```

## 🔑 Step 3: Configure Environment Variables

In your Vercel project dashboard, go to **Settings → Environment Variables** and add:

### Required Variables
```
API_KEYS=key1,key2,key3
SESSION_SECRET=your-random-secret-key-here
```

### Optional Analytics (Free: 1M events/month)
```
VITE_MIXPANEL_TOKEN=your-mixpanel-token
```

### Optional Privacy Settings
```
ANONYMIZE_DATA=true
ANONYMIZATION_SALT=random-salt-string
DATA_RETENTION_DAYS=30
```

**How to get Mixpanel token:**
1. Go to [mixpanel.com](https://mixpanel.com) and create a free account
2. Create a new project
3. Go to **Project Settings** → Copy your **Project Token**

## 🔄 Step 4: Set Up GDPR Cleanup (Optional)

Since Vercel Cron requires a Pro plan, use a free external service:

### Using cron-job.org (Recommended)

1. Go to [cron-job.org](https://cron-job.org) and create a free account
2. Create a new cron job:
   - **URL**: `https://your-app.vercel.app/api/cleanup`
   - **Schedule**: Daily at 02:00
   - **HTTP Method**: GET
   - **Title**: ZeroTrustGNN GDPR Cleanup

### Test the Cleanup Endpoint

```bash
curl https://your-app.vercel.app/api/cleanup
```

Expected response:
```json
{
  "success": true,
  "result": {...}
}
```

## 🧪 Step 5: Test Your Deployment

1. Visit your Vercel URL: `https://your-app.vercel.app`
2. Test GDPR consent banner (accept/decline)
3. Upload a PCAP file and verify scanning works
4. Check Mixpanel dashboard for events (if configured)

## 📊 Monitoring

### Vercel Dashboard
- **Deployments**: View build logs and deployment history
- **Functions**: Monitor API endpoint performance
- **Analytics**: Track traffic (free tier: 100k requests/month)

### Mixpanel Dashboard
- **Events**: View real-time user actions
- **Retention**: Track user engagement
- **Funnels**: Analyze conversion flows

## 🆘 Troubleshooting

### Frontend shows 404
- Check that `vercel.json` is in the root directory
- Verify `frontend/package.json` has `"build": "tsc -b && vite build"`
- Check build logs in Vercel dashboard

### API endpoints return 500
- Check Python function logs in Vercel dashboard
- Verify all environment variables are set
- Ensure `requirements.txt` includes all dependencies

### GDPR consent not working
- Check browser console for errors
- Verify `VITE_API_BASE_URL` points to your Vercel backend
- Test `/gdpr/config` endpoint manually

## 🎉 Success!

Your app is live at: `https://your-app.vercel.app`

**Next Steps:**
- Add a custom domain (Settings → Domains)
- Enable Web Analytics (Settings → Analytics)
- Monitor usage and stay within free tier limits

---

**Free Tier Limits:**
- ✅ Vercel: 100 GB bandwidth/month, 100k serverless executions
- ✅ Mixpanel: 1M events/month (no credit card required)
- ✅ cron-job.org: Unlimited cron jobs on free tier

**Need help?** See the full deployment guide in `VERCEL_DEPLOYMENT.md`
