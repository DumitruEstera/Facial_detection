import React from 'react';

const Dashboard = ({ 
  videoFrame, 
  systemStatus, 
  recentLogs, 
  fireAlerts = [],
  onStartCamera, 
  onStopCamera, 
  onSetMode,
  onToggleDemographics,
  onToggleFireDetection,
  demographicsEnabled,
  fireDetectionEnabled
}) => {
  // Helper function to format demographics display
  const formatDemographics = (log) => {
    if (log.name !== 'Unknown' || (!log.age && !log.gender && !log.emotion)) {
      return null;
    }
    
    const parts = [];
    if (log.age) parts.push(`Age: ${log.age}`);
    if (log.gender) parts.push(`${log.gender}`);
    if (log.emotion) parts.push(`${log.emotion}`);
    
    return parts.length > 0 ? ` | ${parts.join(' | ')}` : '';
  };

  return (
    <div className="dashboard">
      <div className="dashboard-grid">
        <div className="video-section">
          <h2>Live Video Feed</h2>
          <div className="video-container">
            {videoFrame ? (
              <img 
                src={`data:image/jpeg;base64,${videoFrame}`} 
                alt="Live feed"
                className="video-frame"
              />
            ) : (
              <div className="no-video">
                <p>📷 No video feed</p>
                <p>Start camera to begin monitoring</p>
              </div>
            )}
          </div>
          
          <div className="video-controls">
            <button 
              onClick={onStartCamera}
              disabled={systemStatus.streaming}
              className="btn btn-primary"
            >
              ▶️ Start Camera
            </button>
            <button 
              onClick={onStopCamera}
              disabled={!systemStatus.streaming}
              className="btn btn-secondary"
            >
              ⏹️ Stop Camera
            </button>
          </div>
          
          <div className="mode-selector">
            <h3>Detection Mode:</h3>
            <div className="mode-buttons">
              {['face', 'plate', 'both'].map(mode => (
                <button
                  key={mode}
                  onClick={() => onSetMode(mode)}
                  className={`btn ${systemStatus.mode === mode ? 'btn-active' : 'btn-outline'}`}
                >
                  {mode === 'face' ? '👤 Face' : mode === 'plate' ? '🚗 Plate' : '🔍 Both'}
                </button>
              ))}
            </div>
          </div>
          
          {/* Demographics toggle if available */}
          {systemStatus.demographics_enabled !== undefined && (
            <div className="demographics-toggle">
              <label>
                <input 
                  type="checkbox" 
                  checked={demographicsEnabled}
                  onChange={(e) => onToggleDemographics && onToggleDemographics(e.target.checked)}
                />
                <span> 🧠 Enable Demographics Analysis</span>
              </label>
            </div>
          )}
          
          {/* Fire detection toggle if available */}
          {systemStatus.fire_system_available && (
            <div className="fire-toggle">
              <label>
                <input 
                  type="checkbox" 
                  checked={fireDetectionEnabled}
                  onChange={(e) => onToggleFireDetection && onToggleFireDetection(e.target.checked)}
                />
                <span> 🔥 Enable Fire Detection</span>
              </label>
            </div>
          )}
        </div>
        
        <div className="info-section">
          {/* Fire Alerts Banner - Show if there are active fire/smoke detections */}
          {fireAlerts && fireAlerts.length > 0 && (
            <div className="fire-alerts-banner">
              {fireAlerts.some(alert => alert.severity === 'critical') && (
                <div className="fire-alert critical">
                  <div className="fire-alert-icon">🚨</div>
                  <div className="fire-alert-content">
                    <div className="fire-alert-title">CRITICAL FIRE ALERT!</div>
                    <div className="fire-alert-details">
                      {fireAlerts.filter(a => a.severity === 'critical').length} critical detection(s)
                    </div>
                  </div>
                </div>
              )}
              {fireAlerts.filter(alert => alert.severity === 'high').length > 0 && (
                <div className="fire-alert high">
                  <div className="fire-alert-icon">🔥</div>
                  <div className="fire-alert-content">
                    <div className="fire-alert-title">HIGH Fire/Smoke Alert</div>
                    <div className="fire-alert-details">
                      {fireAlerts.filter(a => a.severity === 'high').map((alert, idx) => (
                        <div key={idx}>
                          {alert.class.toUpperCase()} detected ({(alert.confidence * 100).toFixed(0)}%)
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              {fireAlerts.filter(alert => ['medium', 'low'].includes(alert.severity)).length > 0 && (
                <div className="fire-alert medium">
                  <div className="fire-alert-icon">⚠️</div>
                  <div className="fire-alert-content">
                    <div className="fire-alert-title">Fire/Smoke Warning</div>
                    <div className="fire-alert-details">
                      {fireAlerts.filter(a => ['medium', 'low'].includes(a.severity)).length} detection(s)
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
          
          <div className="status-card">
            <h3>System Status</h3>
            <div className="status-items">
              <div className="status-item">
                <span>Camera:</span>
                <span className={systemStatus.streaming ? 'status-active' : 'status-inactive'}>
                  {systemStatus.streaming ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="status-item">
                <span>Mode:</span>
                <span>{systemStatus.mode || 'Unknown'}</span>
              </div>
              <div className="status-item">
                <span>Connected Clients:</span>
                <span>{systemStatus.connected_clients || 0}</span>
              </div>
              {systemStatus.demographics_enabled !== undefined && (
                <div className="status-item">
                  <span>Demographics:</span>
                  <span className={systemStatus.demographics_enabled ? 'status-active' : 'status-inactive'}>
                    {systemStatus.demographics_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              )}
              {systemStatus.fire_detection_enabled !== undefined && (
                <div className="status-item">
                  <span>Fire Detection:</span>
                  <span className={systemStatus.fire_detection_enabled ? 'status-active' : 'status-inactive'}>
                    {systemStatus.fire_detection_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
              )}
            </div>
          </div>
          
          <div className="recent-logs">
            <h3>Recent Activity</h3>
            <div className="logs-container">
              {recentLogs.slice(0, 10).map((log, index) => (
                <div key={index} className={`log-item ${log.type}`}>
                  <div className="log-time">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </div>
                  <div className="log-content">
                    {log.type === 'face' ? (
                      <div>
                        <span className={log.name === 'Unknown' ? 'unknown-person' : ''}>
                          👤 {log.name || 'Unknown'} 
                          {log.confidence && log.confidence > 0 && ` (${(log.confidence * 100).toFixed(1)}%)`}
                        </span>
                        {log.name === 'Unknown' && (
                          <div className="demographics-info">
                            {formatDemographics(log)}
                          </div>
                        )}
                      </div>
                    ) : log.type === 'fire' ? (
                      <div>
                        <span className={`fire-log severity-${log.severity}`}>
                          {log.class === 'fire' ? '🔥' : '💨'} {log.class.toUpperCase()} 
                          {` (${(log.confidence * 100).toFixed(1)}%)`}
                          <span className={`severity-badge ${log.severity}`}>
                            {log.severity.toUpperCase()}
                          </span>
                          {log.alert && <span className="alert-badge">⚠️ ALERT</span>}
                        </span>
                      </div>
                    ) : (
                      <span>
                        🚗 {log.plate || log.plate_number || 'Unknown'} 
                        {log.owner && ` - ${log.owner}`}
                        {log.authorised !== undefined && (
                          <span className={log.authorised ? 'authorized' : 'unauthorized'}>
                            {log.authorised ? ' ✅' : ' ❌'}
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                </div>
              ))}
              {recentLogs.length === 0 && (
                <p className="no-logs">No recent activity</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;