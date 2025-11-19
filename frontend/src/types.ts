export interface Alert {
  flow_id: number
  risk_score: number
  risk_level: string
  source_ip: string
  source_port: number
  destination_ip: string
  destination_port: number
  protocol: string
  bytes_transferred: number
  message: string
}

export interface Summary {
  total_flows: number
  high_risk_flows: number
  medium_risk_flows: number
  low_risk_flows: number
  average_risk_score: number
  max_risk_score: number
  alerts_triggered: number
}

export interface ScanResult {
  success: boolean
  timestamp: string
  summary: Summary
  alerts: Alert[]
}
