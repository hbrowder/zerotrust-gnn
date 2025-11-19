import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import type { Alert } from '../types'

interface NetworkGraphProps {
  alerts: Alert[]
}

export default function NetworkGraph({ alerts }: NetworkGraphProps) {
  const riskCategories = [
    { name: 'Low (0-30)', min: 0, max: 30, color: '#10b981' },
    { name: 'Medium (30-50)', min: 30, max: 50, color: '#f59e0b' },
    { name: 'High (50-70)', min: 50, max: 70, color: '#f97316' },
    { name: 'Critical (70-100)', min: 70, max: 100, color: '#ef4444' }
  ]

  const data = riskCategories.map(category => ({
    name: category.name,
    count: alerts.filter(a => a.risk_score >= category.min && a.risk_score < category.max).length,
    color: category.color
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis 
          dataKey="name" 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af' }}
        />
        <YAxis 
          stroke="#9ca3af"
          tick={{ fill: '#9ca3af' }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1f2937',
            border: '1px solid #374151',
            borderRadius: '0.5rem',
            color: '#fff'
          }}
        />
        <Bar dataKey="count" radius={[8, 8, 0, 0]}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
