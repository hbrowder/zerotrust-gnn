import { useState, useRef } from 'react'
import { Upload, File } from 'lucide-react'
import { scanPcapFile } from '../utils/api'
import type { ScanResult } from '../types'

interface FileUploadProps {
  onScanComplete: (result: ScanResult) => void
  onScanStart: () => void
  onError: (error: string) => void
}

export default function FileUpload({ onScanComplete, onScanStart, onError }: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (file: File) => {
    if (file.name.endsWith('.pcap') || file.name.endsWith('.pcapng')) {
      setSelectedFile(file)
    } else {
      onError('Please select a valid PCAP file (.pcap or .pcapng)')
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleScan = async () => {
    if (!selectedFile) return
    
    try {
      onScanStart()
      const result = await scanPcapFile(selectedFile)
      onScanComplete(result)
    } catch (error: any) {
      onError(error.message)
    }
  }

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6">
      <h2 className="text-xl font-semibold text-white mb-4 flex items-center">
        <Upload className="h-5 w-5 mr-2 text-blue-400" />
        Upload PCAP File
      </h2>
      
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          isDragging
            ? 'border-blue-400 bg-blue-900/20'
            : 'border-gray-600 hover:border-gray-500'
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pcap,.pcapng"
          onChange={handleFileInputChange}
          className="hidden"
        />
        
        {selectedFile ? (
          <div className="space-y-2">
            <File className="h-12 w-12 text-green-400 mx-auto" />
            <p className="text-white font-medium">{selectedFile.name}</p>
            <p className="text-gray-400 text-sm">
              {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="h-12 w-12 text-gray-500 mx-auto" />
            <p className="text-gray-300">
              Drag and drop your PCAP file here, or click to browse
            </p>
            <p className="text-gray-500 text-sm">
              Supports .pcap and .pcapng files
            </p>
          </div>
        )}
      </div>
      
      {selectedFile && (
        <div className="mt-4 flex justify-end space-x-3">
          <button
            onClick={() => setSelectedFile(null)}
            className="px-4 py-2 border border-gray-600 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Clear
          </button>
          <button
            onClick={handleScan}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center space-x-2"
          >
            <Upload className="h-4 w-4" />
            <span>Scan for Anomalies</span>
          </button>
        </div>
      )}
    </div>
  )
}
