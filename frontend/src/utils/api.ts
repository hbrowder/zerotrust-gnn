import axios from 'axios'
import type { ScanResult } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const API_KEY = import.meta.env.VITE_API_KEY || ''

export const scanPcapFile = async (file: File): Promise<ScanResult> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    
    reader.onload = async () => {
      try {
        const base64 = btoa(
          new Uint8Array(reader.result as ArrayBuffer)
            .reduce((data, byte) => data + String.fromCharCode(byte), '')
        )
        
        const sessionId = localStorage.getItem('gdpr_session_id')
        
        const response = await axios.post<ScanResult>(
          `${API_BASE_URL}/scan`,
          {
            file: base64,
            risk_threshold: 50,
            session_id: sessionId
          },
          {
            headers: {
              'Content-Type': 'application/json',
              'X-API-Key': API_KEY
            }
          }
        )
        
        resolve(response.data)
      } catch (error: any) {
        if (error.response?.data?.error) {
          reject(new Error(error.response.data.error))
        } else {
          reject(new Error('Failed to scan PCAP file'))
        }
      }
    }
    
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsArrayBuffer(file)
  })
}

export const checkApiHealth = async (): Promise<boolean> => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`)
    return response.data.status === 'healthy'
  } catch {
    return false
  }
}
