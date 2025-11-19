import { useState, useEffect } from 'react'
import { Shield, X, Check } from 'lucide-react'

interface GDPRConsentProps {
  onConsentChange: (consented: boolean, sessionId: string) => void
}

export default function GDPRConsent({ onConsentChange }: GDPRConsentProps) {
  const [showBanner, setShowBanner] = useState(false)
  const [showDetails, setShowDetails] = useState(false)
  const [privacyConfig, setPrivacyConfig] = useState<{anonymization: boolean; gdpr_compliant: boolean} | null>(null)
  const [sessionId] = useState(() => {
    const existing = localStorage.getItem('gdpr_session_id')
    if (existing) return existing
    
    const newId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    localStorage.setItem('gdpr_session_id', newId)
    return newId
  })

  useEffect(() => {
    const consent = localStorage.getItem('gdpr_consent')
    const consentSessionId = localStorage.getItem('gdpr_session_id')
    
    fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/gdpr/config`)
      .then(res => res.json())
      .then(data => {
        setPrivacyConfig({
          anonymization: data.anonymization_enabled === true,
          gdpr_compliant: data.gdpr_compliant === true
        })
      })
      .catch(() => {
        setPrivacyConfig({ anonymization: false, gdpr_compliant: false })
      })
    
    if (!consent) {
      setShowBanner(true)
    } else if (consentSessionId) {
      onConsentChange(consent === 'true', consentSessionId)
    }
  }, [onConsentChange])

  const handleConsent = async (accepted: boolean) => {
    const consentTypes = accepted ? ['logging', 'analytics'] : []
    
    try {
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/gdpr/consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          consent_given: accepted,
          consent_types: consentTypes
        })
      })

      if (response.ok) {
        localStorage.setItem('gdpr_consent', accepted.toString())
        localStorage.setItem('gdpr_session_id', sessionId)
        localStorage.setItem('gdpr_consent_date', new Date().toISOString())
        
        setShowBanner(false)
        onConsentChange(accepted, sessionId)
      }
    } catch (error) {
      console.error('Failed to record consent:', error)
    }
  }

  const handleDeleteData = async () => {
    const storedSessionId = localStorage.getItem('gdpr_session_id')
    if (!storedSessionId) return

    if (confirm('Are you sure you want to delete all your data? This action cannot be undone.')) {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/gdpr/delete-data`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: storedSessionId
          })
        })

        if (response.ok) {
          localStorage.removeItem('gdpr_consent')
          localStorage.removeItem('gdpr_session_id')
          localStorage.removeItem('gdpr_consent_date')
          
          alert('All your data has been permanently deleted.')
          setShowBanner(true)
        }
      } catch (error) {
        console.error('Failed to delete data:', error)
        alert('Failed to delete data. Please try again.')
      }
    }
  }

  const viewPrivacyPolicy = () => {
    window.open(`${import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'}/gdpr/privacy-policy`, '_blank')
  }

  if (!showBanner) {
    return (
      <button
        onClick={() => setShowBanner(true)}
        className="fixed bottom-4 right-4 bg-gray-800/90 backdrop-blur-sm text-white px-4 py-2 rounded-lg border border-gray-700 hover:bg-gray-700 transition-colors text-sm flex items-center space-x-2"
      >
        <Shield className="h-4 w-4" />
        <span>Privacy Settings</span>
      </button>
    )
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-gray-900/95 backdrop-blur-md border-t border-gray-700 p-6 z-50">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Shield className="h-6 w-6 text-blue-400" />
            <h3 className="text-xl font-bold text-white">Privacy & GDPR Compliance</h3>
          </div>
          <button
            onClick={() => setShowBanner(false)}
            className="text-gray-400 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="space-y-4">
          <p className="text-gray-300 text-sm">
            We respect your privacy. 
            {privacyConfig ? (
              privacyConfig.anonymization 
                ? ' IP addresses are anonymized for GDPR compliance.' 
                : ' WARNING: IP anonymization is currently disabled.'
            ) : ' Loading privacy settings...'}
            {' '}Your data is automatically deleted after 30 days.
          </p>

          {showDetails && privacyConfig && (
            <div className="bg-gray-800/50 rounded-lg p-4 text-sm text-gray-300 space-y-2">
              <h4 className="font-semibold text-white mb-2">What we collect (only with your consent):</h4>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Audit logs of your API requests (for security monitoring)</li>
                <li>Session analytics (anonymized usage patterns)</li>
              </ul>
              <h4 className="font-semibold text-white mt-3 mb-2">Data Protection:</h4>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Personal identifying information (PII): Not collected</li>
                {privacyConfig.anonymization ? (
                  <li className="text-green-400">✓ IP addresses: Pseudonymized using SHA-256 hashing</li>
                ) : (
                  <li className="text-red-400">⚠ WARNING: IP addresses are NOT anonymized - raw IPs exposed</li>
                )}
                <li>Cookies or tracking data: Not used</li>
                <li>PCAP files: Deleted immediately after processing</li>
              </ul>
              <h4 className="font-semibold text-white mt-3 mb-2">Your rights:</h4>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Right to access, rectify, and delete your data</li>
                <li>Right to withdraw consent at any time</li>
                <li>Right to data portability</li>
                <li>Automatic data deletion after 30 days</li>
              </ul>
            </div>
          )}

          <div className="flex items-center justify-between flex-wrap gap-3">
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-blue-400 hover:text-blue-300 text-sm underline"
              >
                {showDetails ? 'Hide Details' : 'Show Details'}
              </button>
              <button
                onClick={viewPrivacyPolicy}
                className="text-blue-400 hover:text-blue-300 text-sm underline"
              >
                View Full Privacy Policy
              </button>
              {localStorage.getItem('gdpr_consent') && (
                <button
                  onClick={handleDeleteData}
                  className="text-red-400 hover:text-red-300 text-sm underline"
                >
                  Delete My Data
                </button>
              )}
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={() => handleConsent(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm flex items-center space-x-2"
              >
                <X className="h-4 w-4" />
                <span>Decline</span>
              </button>
              <button
                onClick={() => handleConsent(true)}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors text-sm flex items-center space-x-2"
              >
                <Check className="h-4 w-4" />
                <span>Accept & Continue</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
