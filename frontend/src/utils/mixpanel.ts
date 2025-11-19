import mixpanel from 'mixpanel-browser'

const MIXPANEL_TOKEN = import.meta.env.VITE_MIXPANEL_TOKEN || ''

class MixpanelService {
  private initialized = false

  init() {
    if (!MIXPANEL_TOKEN) {
      console.warn('Mixpanel token not configured - analytics disabled')
      return
    }

    if (!this.initialized) {
      mixpanel.init(MIXPANEL_TOKEN, {
        debug: import.meta.env.DEV,
        track_pageview: true,
        persistence: 'localStorage',
        ignore_dnt: true
      })
      this.initialized = true
      console.log('Mixpanel initialized')
    }
  }

  track(eventName: string, properties?: Record<string, any>) {
    if (!this.initialized) return
    mixpanel.track(eventName, properties)
  }

  identify(userId: string) {
    if (!this.initialized) return
    mixpanel.identify(userId)
  }

  setUserProperties(properties: Record<string, any>) {
    if (!this.initialized) return
    mixpanel.people.set(properties)
  }

  trackPageView(pageName: string) {
    this.track('Page View', {
      page: pageName,
      url: window.location.href
    })
  }

  trackFileUpload(fileSize: number) {
    this.track('PCAP File Uploaded', {
      file_size_kb: Math.round(fileSize / 1024),
      timestamp: new Date().toISOString()
    })
  }

  trackScanComplete(alertsCount: number, highRisk: number, anonymized: boolean) {
    this.track('Scan Completed', {
      total_alerts: alertsCount,
      high_risk_flows: highRisk,
      data_anonymized: anonymized,
      timestamp: new Date().toISOString()
    })
  }

  trackConsentGiven(consentTypes: string[]) {
    this.track('GDPR Consent Given', {
      consent_types: consentTypes,
      timestamp: new Date().toISOString()
    })
  }

  trackConsentDeclined() {
    this.track('GDPR Consent Declined', {
      timestamp: new Date().toISOString()
    })
  }

  trackDataDeletion() {
    this.track('Data Deletion Requested', {
      timestamp: new Date().toISOString()
    })
  }

  trackAlertClick(riskLevel: string, riskScore: number) {
    this.track('Alert Clicked', {
      risk_level: riskLevel,
      risk_score: riskScore,
      timestamp: new Date().toISOString()
    })
  }
}

export const analytics = new MixpanelService()
