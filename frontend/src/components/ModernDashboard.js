import React, { useEffect } from 'react';
import '../ModernDashboard.css';

const ModernDashboard = ({ 
  videoFrame, 
  systemStatus = {}, 
  recentLogs = [], 
  fireAlerts = [],
  onStartCamera, 
  onStopCamera, 
  onToggleFaceDetection,
  onTogglePlateDetection,
  onToggleDemographics,
  onToggleFireDetection,
  faceDetectionEnabled = true,
  plateDetectionEnabled = true,
  demographicsEnabled = false,
  fireDetectionEnabled = false,
  isConnected = false
}) => {
  // Debug logging
  useEffect(() => {
    console.log('ModernDashboard Props:', {
      hasVideoFrame: !!videoFrame,
      isConnected,
      systemStatus,
      demographicsEnabled,
      fireDetectionEnabled,
      recentLogsCount: recentLogs.length,
      fireAlertsCount: fireAlerts.length
    });
  }, [videoFrame, isConnected, systemStatus, demographicsEnabled, fireDetectionEnabled, recentLogs, fireAlerts]);

  return (
    <div className="modern-grid">
      {/* Video Stream */}
      <div className="video-card">
        <div className="card-header">
          <h2>Live Camera Feed</h2>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
        </div>
        <div className="card-content">
          <div className={`video-display ${(videoFrame && (systemStatus?.streaming || isConnected)) ? 'streaming' : 'no-stream'}`}>
            {videoFrame && (systemStatus?.streaming || isConnected) ? (
              <img 
                src={`data:image/jpeg;base64,${videoFrame}`} 
                alt="Live feed"
                className="video-stream"
              />
            ) : (
              <div className="no-stream-text">
                {(systemStatus?.streaming || isConnected) ? 'Connecting to video feed...' : 'No active stream. Click "Start Stream" to begin.'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Side Panels */}
      <div className="side-panels">
        {/* Model Toggles */}
        <div className="side-card">
          <div className="card-header">
            <h3>Active Models</h3>
          </div>
          <div className="card-content">
            <div className="toggle-list">
              <div className="toggle-item">
                <span>Face Detection</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={faceDetectionEnabled}
                    onChange={(e) => onToggleFaceDetection && onToggleFaceDetection(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <div className="toggle-item">
                <span>Demographics</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={demographicsEnabled || false}
                    onChange={(e) => onToggleDemographics && onToggleDemographics(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <div className="toggle-item">
                <span>License Plate</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={plateDetectionEnabled}
                    onChange={(e) => onTogglePlateDetection && onTogglePlateDetection(e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
              <div className="toggle-item">
                <span>Fire Detection</span>
                <label className="toggle-switch">
                  <input 
                    type="checkbox" 
                    checked={fireDetectionEnabled || false}
                    onChange={(e) => onToggleFireDetection && onToggleFireDetection(e.target.checked)}
                    disabled={!systemStatus?.fire_system_available}
                  />
                  <span className="toggle-slider"></span>
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Alerts Panel */}
        <div className="side-card alerts-card">
          <div className="card-header">
            <h3>Alerts</h3>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="alert-icon">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
              <line x1="12" y1="9" x2="12" y2="13"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
          </div>
          <div className="card-content alerts-content">
            {/* Fire Alerts */}
            {fireAlerts && fireAlerts.length > 0 && fireAlerts.map((alert, idx) => (
              <div key={`fire-${idx}`} className={`alert-item alert-${alert.severity || 'warning'}`}>
                <p>
                  [{new Date().toLocaleTimeString()}] {alert.class === 'fire' ? '🔥' : '💨'} {(alert.class || 'unknown').toUpperCase()} detected
                  <span className="alert-confidence">({((alert.confidence || 0) * 100).toFixed(0)}%)</span>
                </p>
              </div>
            ))}
            
            {/* Recent logs as alerts */}
            {recentLogs && recentLogs.length > 0 && recentLogs.slice(0, 5).map((log, index) => {
              let alertClass = 'info';
              let message = '';
              
              if (log.type === 'fire') {
                alertClass = log.severity === 'critical' ? 'critical' : log.severity === 'high' ? 'high' : 'warning';
                message = `${log.class === 'fire' ? '🔥' : '💨'} ${(log.class || 'unknown').toUpperCase()} detected`;
              } else if (log.type === 'face') {
                if (log.name === 'Unknown') {
                  alertClass = 'warning';
                  message = `Unknown face detected`;
                } else {
                  alertClass = 'info';
                  message = `${log.name} recognized`;
                }
              } else if (log.type === 'plate') {
                alertClass = log.authorised === false ? 'warning' : 'info';
                message = `License plate: ${log.plate || log.plate_number || 'Unknown'}`;
              }
              
              return (
                <div key={index} className={`alert-item alert-${alertClass}`}>
                  <p>[{log.timestamp ? new Date(log.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()}] {message}</p>
                </div>
              );
            })}
            
            {(!recentLogs || recentLogs.length === 0) && (!fireAlerts || fireAlerts.length === 0) && (
              <p className="no-alerts">No recent alerts</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModernDashboard;
