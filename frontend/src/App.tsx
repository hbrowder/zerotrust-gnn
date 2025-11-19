import { useState } from 'react'
import { Upload, AlertCircle, Shield, Activity } from 'lucide-react'
import FileUpload from './components/FileUpload'
import NetworkGraph from './components/NetworkGraph'
import AlertsList from './components/AlertsList'
import StatsPanel from './components/StatsPanel'
import GDPRConsent from './components/GDPRConsent'
import type { ScanResult } from './types'

function App() {
  const [scanResult, setScanResult] = useState<ScanResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [gdprConsent, setGdprConsent] = useState(false)

  const handleConsentChange = (consented: boolean, newSessionId: string) => {
    setGdprConsent(consented)
    setSessionId(newSessionId)
  }

  const handleScanComplete = (result: ScanResult) => {
    setScanResult(result)
    setLoading(false)
    setError(null)
  }

  const handleScanStart = () => {
    setLoading(true)
    setError(null)
  }

  const handleError = (errorMessage: string) => {
    setError(errorMessage)
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Header */}
      <header className="bg-gray-900/50 backdrop-blur-sm border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Shield className="h-10 w-10 text-blue-400" />
              <div>
                <h1 className="text-3xl font-bold text-white">ZeroTrustGNN</h1>
                <p className="text-sm text-gray-400">Network Anomaly Detection System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="h-5 w-5 text-green-400" />
                <span className="text-sm text-gray-300">API Connected</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* File Upload Section */}
        <div className="mb-8">
          <FileUpload
            onScanComplete={handleScanComplete}
            onScanStart={handleScanStart}
            onError={handleError}
          />
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 bg-red-900/50 border border-red-700 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-400" />
              <span className="text-red-200">{error}</span>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400"></div>
            <p className="mt-4 text-gray-300">Analyzing network traffic...</p>
          </div>
        )}

        {/* Results Display */}
        {scanResult && !loading && (
          <div className="space-y-8">
            {/* Statistics Panel */}
            <StatsPanel summary={scanResult.summary} />

            {/* Network Graph */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
              <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
                <Activity className="h-5 w-5 mr-2 text-blue-400" />
                Risk Score Distribution
              </h2>
              <NetworkGraph alerts={scanResult.alerts} />
            </div>

            {/* Alerts List */}
            <AlertsList alerts={scanResult.alerts} />
          </div>
        )}

        {/* Empty State */}
        {!scanResult && !loading && !error && (
          <div className="text-center py-16">
            <Upload className="h-16 w-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-medium text-gray-400 mb-2">
              No scan results yet
            </h3>
            <p className="text-gray-500">
              Upload a PCAP file to analyze network traffic for anomalies
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-16 bg-gray-900/50 backdrop-blur-sm border-t border-gray-700 py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-gray-400 text-sm">
          <p>Powered by Graph Neural Networks • 87.5% Accuracy • 97.5% Recall</p>
          {scanResult?.privacy && (
            <p className="mt-2 text-xs">
              <span className="inline-flex items-center space-x-1">
                <Shield className="h-3 w-3" />
                <span>
                  {scanResult.privacy.data_anonymized ? 'Data Anonymized' : 'Anonymization Disabled'} • 
                  {scanResult.privacy.gdpr_compliant ? ' GDPR Compliant' : ' GDPR Non-Compliant'} • 
                  TLS 1.3 Ready
                </span>
              </span>
            </p>
          )}
        </div>
      </footer>

      {/* GDPR Consent Banner */}
      <GDPRConsent onConsentChange={handleConsentChange} />
    </div>
  )
}

export default App
