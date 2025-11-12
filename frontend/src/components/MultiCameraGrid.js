import React, { useState } from 'react';
import './MultiCameraGrid.css';

const MultiCameraGrid = ({ 
  cameras = [],
  alerts = [],
  onCameraClick,
  systemStatus = {},
  onToggleFaceDetection,
  onTogglePlateDetection,
  onToggleDemographics,
  onToggleFireDetection,
  faceDetectionEnabled = true,
  plateDetectionEnabled = true,
  demographicsEnabled = false,
  fireDetectionEnabled = false
}) => {
  const [selectedCamera, setSelectedCamera] = useState(null);

  // Create array of 9 cameras, fill with placeholders if needed
  const cameraSlots = Array(9).fill(null).map((_, index) => 
    cameras[index] || { id: `CAM-${index + 1}`, status: 'inactive', stream: null }
  );

  const handleCameraClick = (camera) => {
    setSelectedCamera(camera);
    if (onCameraClick) {
      onCameraClick(camera);
    }
  };

  const closeModal = () => {
    setSelectedCamera(null);
  };

  // Get recent alerts (last 24h)
  const recentAlerts = alerts.filter(alert => {
    const alertTime = new Date(alert.timestamp);
    const now = new Date();
    const diffHours = (now - alertTime) / (1000 * 60 * 60);
    return diffHours <= 24;
  });

  // Check if camera has active alerts
  const hasAlert = (cameraId) => {
    return recentAlerts.some(alert => alert.cameraId === cameraId);
  };

  return (
    <div className="multi-camera-container">
      {/* Camera Grid Section */}
      <div className="camera-grid-section">
        <div className="camera-grid">
          {cameraSlots.map((camera, index) => (
            <div 
              key={index} 
              className={`camera-slot ${camera.status === 'active' ? 'active' : 'inactive'}`}
              onClick={() => camera.status === 'active' && handleCameraClick(camera)}
            >
              {/* Camera ID - Top Left */}
              <div className="camera-id">{camera.id}</div>
              
              {/* Alert Indicator - Top Right */}
              {hasAlert(camera.id) && (
                <div className="alert-indicator">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L1 21h22L12 2zm0 3.5L19.5 19h-15L12 5.5zM11 10v4h2v-4h-2zm0 5v2h2v-2h-2z"/>
                  </svg>
                </div>
              )}
              
              {/* Video Stream or Placeholder */}
              <div className="camera-content">
                {camera.status === 'active' && camera.stream ? (
                  <img 
                    src={camera.stream} 
                    alt={`Camera ${camera.id}`}
                    className="camera-stream"
                  />
                ) : (
                  <div className="camera-placeholder">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                      <circle cx="12" cy="13" r="4"/>
                    </svg>
                    <span>{camera.status === 'inactive' ? 'Camera Offline' : 'No Signal'}</span>
                  </div>
                )}
              </div>
              
              {/* Status Indicator */}
              <div className={`status-dot ${camera.status}`}></div>
            </div>
          ))}
        </div>
      </div>

      {/* Alerts Dashboard - Right Side */}
      <div className="right-panels">
        {/* Model Toggles Panel */}
        <div className="models-panel">
          <div className="panel-header">
            <h3>Active Models</h3>
          </div>
          <div className="panel-content">
            <div className="toggle-list">
              {/* Face Detection Toggle */}
              <div className="toggle-item">
                <span className="toggle-label">Face Detection</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={faceDetectionEnabled}
                    onChange={(e) => onToggleFaceDetection && onToggleFaceDetection(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* License Plate Detection Toggle */}
              <div className="toggle-item">
                <span className="toggle-label">License Plate</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={plateDetectionEnabled}
                    onChange={(e) => onTogglePlateDetection && onTogglePlateDetection(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* Demographics Toggle */}
              <div className="toggle-item">
                <span className="toggle-label">Demographics</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={demographicsEnabled}
                    onChange={(e) => onToggleDemographics && onToggleDemographics(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>

              {/* Fire Detection Toggle */}
              <div className="toggle-item">
                <span className="toggle-label">Fire Detection</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={fireDetectionEnabled}
                    onChange={(e) => onToggleFireDetection && onToggleFireDetection(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Alerts Dashboard */}
        <div className="alerts-dashboard">
        <div className="alerts-header">
          <h2>Recent Alerts</h2>
          <span className="alerts-time-range">Last 24 Hours</span>
        </div>
        
        <div className="alerts-list">
          {recentAlerts.length > 0 ? (
            recentAlerts.map((alert, index) => (
              <div key={index} className={`alert-item ${alert.severity || 'medium'}`}>
                <div className="alert-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L1 21h22L12 2zm0 3.5L19.5 19h-15L12 5.5zM11 10v4h2v-4h-2zm0 5v2h2v-2h-2z"/>
                  </svg>
                </div>
                <div className="alert-details">
                  <div className="alert-title">
                    {alert.type === 'unauthorized' && '🚨 Unauthorized Access'}
                    {alert.type === 'fire' && '🔥 Fire Detected'}
                    {alert.type === 'motion' && '👁️ Motion Detected'}
                    {alert.type === 'intrusion' && '⚠️ Perimeter Breach'}
                    {!['unauthorized', 'fire', 'motion', 'intrusion'].includes(alert.type) && '⚠️ Alert'}
                  </div>
                  <div className="alert-camera">{alert.cameraId}</div>
                  <div className="alert-time">{new Date(alert.timestamp).toLocaleString()}</div>
                  {alert.description && (
                    <div className="alert-description">{alert.description}</div>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="no-alerts">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
              </svg>
              <p>No alerts in the last 24 hours</p>
              <span>All systems operational</span>
            </div>
          )}
        </div>
        </div>
      </div>

      {/* Modal for Expanded Camera View */}
      {selectedCamera && (
        <div className="camera-modal-overlay" onClick={closeModal}>
          <div className="camera-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedCamera.id}</h2>
              <button className="modal-close" onClick={closeModal}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18"/>
                  <line x1="6" y1="6" x2="18" y2="18"/>
                </svg>
              </button>
            </div>
            <div className="modal-content">
              {selectedCamera.stream ? (
                <img 
                  src={cameras.find(cam => cam.id === selectedCamera.id)?.stream || selectedCamera.stream}
                  alt={`Camera ${selectedCamera.id}`}
                  className="modal-stream"
                  key={selectedCamera.id}
                />
              ) : (
                <div className="modal-placeholder">
                  <p>No stream available</p>
                </div>
              )}
            </div>
            <div className="modal-info">
              <div className="info-item">
                <span className="info-label">Status:</span>
                <span className={`info-value status-${cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status}`}>
                  {(cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status).toUpperCase()}
                </span>
              </div>
              <div className="info-item">
                <span className="info-label">Location:</span>
                <span className="info-value">{selectedCamera.location || 'Not specified'}</span>
              </div>
              <div className="info-item">
                <span className="info-label">Resolution:</span>
                <span className="info-value">{selectedCamera.resolution || '1920x1080'}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiCameraGrid;