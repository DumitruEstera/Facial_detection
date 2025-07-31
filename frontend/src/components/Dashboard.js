import React from 'react';

const Dashboard = ({ videoFrame, systemStatus, recentLogs, onStartCamera, onStopCamera, onSetMode }) => {
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
        </div>
        
        <div className="info-section">
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
                      <span>
                        👤 {log.name || 'Unknown'} 
                        {log.confidence && ` (${(log.confidence * 100).toFixed(1)}%)`}
                      </span>
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