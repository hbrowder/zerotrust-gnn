import { Activity, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react'
import type { Summary } from '../types'

interface StatsPanelProps {
  summary: Summary
}

export default function StatsPanel({ summary }: StatsPanelProps) {
  const stats = [
    {
      label: 'Total Flows',
      value: summary.total_flows,
      icon: Activity,
      color: 'text-blue-400',
      bgColor: 'bg-blue-900/30'
    },
    {
      label: 'High Risk Flows',
      value: summary.high_risk_flows,
      icon: AlertTriangle,
      color: 'text-red-400',
      bgColor: 'bg-red-900/30'
    },
    {
      label: 'Medium Risk Flows',
      value: summary.medium_risk_flows,
      icon: TrendingUp,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-900/30'
    },
    {
      label: 'Low Risk Flows',
      value: summary.low_risk_flows,
      icon: CheckCircle,
      color: 'text-green-400',
      bgColor: 'bg-green-900/30'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => {
        const Icon = stat.icon
        return (
          <div
            key={index}
            className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm">{stat.label}</span>
              <div className={`p-2 rounded-lg ${stat.bgColor}`}>
                <Icon className={`h-5 w-5 ${stat.color}`} />
              </div>
            </div>
            <div className="flex items-baseline space-x-2">
              <span className="text-3xl font-bold text-white">{stat.value}</span>
              {index === 0 && (
                <span className="text-sm text-gray-400">
                  (avg: {summary.average_risk_score.toFixed(1)})
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
