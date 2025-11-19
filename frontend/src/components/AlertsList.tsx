import { useState } from 'react'
import { AlertCircle, X, ArrowRight } from 'lucide-react'
import type { Alert } from '../types'

interface AlertsListProps {
  alerts: Alert[]
}

export default function AlertsList({ alerts }: AlertsListProps) {
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-900/50 border-red-700 text-red-200'
      case 'high':
        return 'bg-orange-900/50 border-orange-700 text-orange-200'
      case 'medium':
        return 'bg-yellow-900/50 border-yellow-700 text-yellow-200'
      default:
        return 'bg-green-900/50 border-green-700 text-green-200'
    }
  }

  const getRiskBadgeColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'critical':
        return 'bg-red-600 text-white'
      case 'high':
        return 'bg-orange-600 text-white'
      case 'medium':
        return 'bg-yellow-600 text-white'
      default:
        return 'bg-green-600 text-white'
    }
  }

  const sortedAlerts = [...alerts].sort((a, b) => b.risk_score - a.risk_score)

  return (
    <>
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
          <AlertCircle className="h-5 w-5 mr-2 text-red-400" />
          Network Flow Alerts ({alerts.length})
        </h2>
        
        <div className="space-y-3">
          {sortedAlerts.slice(0, 10).map((alert) => (
            <div
              key={alert.flow_id}
              className={`border rounded-lg p-4 cursor-pointer transition-all hover:scale-[1.01] ${getRiskColor(alert.risk_level)}`}
              onClick={() => setSelectedAlert(alert)}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-3">
                  <span className={`px-3 py-1 rounded-full text-xs font-bold ${getRiskBadgeColor(alert.risk_level)}`}>
                    {alert.risk_level.toUpperCase()}
                  </span>
                  <span className="text-sm font-mono text-gray-300">
                    Flow #{alert.flow_id}
                  </span>
                </div>
                <span className="text-2xl font-bold">
                  {alert.risk_score.toFixed(1)}/100
                </span>
              </div>
              
              <div className="flex items-center space-x-2 text-sm">
                <span className="font-mono">{alert.source_ip}:{alert.source_port}</span>
                <ArrowRight className="h-4 w-4" />
                <span className="font-mono">{alert.destination_ip}:{alert.destination_port}</span>
                <span className="ml-auto px-2 py-1 bg-gray-700 rounded text-xs">
                  {alert.protocol}
                </span>
              </div>
            </div>
          ))}
          
          {alerts.length > 10 && (
            <p className="text-center text-gray-400 text-sm py-2">
              + {alerts.length - 10} more alerts
            </p>
          )}
        </div>
      </div>

      {/* Alert Detail Modal */}
      {selectedAlert && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => setSelectedAlert(null)}
        >
          <div 
            className="bg-gray-800 border border-gray-700 rounded-lg p-6 max-w-2xl w-full mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white flex items-center">
                <AlertCircle className="h-6 w-6 mr-2 text-red-400" />
                Flow Details #{selectedAlert.flow_id}
              </h3>
              <button
                onClick={() => setSelectedAlert(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            
            <div className="space-y-4">
              <div className="flex items-center space-x-4">
                <span className={`px-4 py-2 rounded-lg text-sm font-bold ${getRiskBadgeColor(selectedAlert.risk_level)}`}>
                  {selectedAlert.risk_level.toUpperCase()}
                </span>
                <span className="text-3xl font-bold text-white">
                  {selectedAlert.risk_score.toFixed(2)}/100
                </span>
              </div>
              
              <div className="bg-gray-900/50 rounded-lg p-4 space-y-3">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Source</p>
                    <p className="text-white font-mono">
                      {selectedAlert.source_ip}:{selectedAlert.source_port}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Destination</p>
                    <p className="text-white font-mono">
                      {selectedAlert.destination_ip}:{selectedAlert.destination_port}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Protocol</p>
                    <p className="text-white">{selectedAlert.protocol}</p>
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Bytes Transferred</p>
                    <p className="text-white">{selectedAlert.bytes_transferred.toLocaleString()}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
                <p className="text-yellow-200 text-sm">
                  {selectedAlert.message}
                </p>
              </div>
              
              <button
                onClick={() => setSelectedAlert(null)}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
