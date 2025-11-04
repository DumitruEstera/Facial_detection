import React from 'react';

const Logs = ({ logs }) => {
  const [filter, setFilter] = React.useState('all');
  
  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true;
    return log.type === filter;
  });

  // Helper function to format demographics for display
  const formatDemographics = (log) => {
    if (!log || log.name !== 'Unknown') return '';
    
    const parts = [];
    if (log.age) parts.push(`Age: ${log.age}`);
    if (log.gender) parts.push(log.gender);
    if (log.emotion) parts.push(log.emotion);
    
    return parts.join(', ');
  };

  return (
    <div className="modern-logs-container">
      <div className="logs-header">
        <h2>📋 Activity Logs</h2>
        <p>View and filter detection activity across all systems</p>
      </div>
      
      <div className="modern-filter-bar">
        <button 
          className={filter === 'all' ? 'filter-btn active' : 'filter-btn'}
          onClick={() => setFilter('all')}
        >
          <span className="filter-icon">📊</span>
          All
        </button>
        <button 
          className={filter === 'face' ? 'filter-btn active' : 'filter-btn'}
          onClick={() => setFilter('face')}
        >
          <span className="filter-icon">👤</span>
          Face Recognition
        </button>
        <button 
          className={filter === 'plate' ? 'filter-btn active' : 'filter-btn'}
          onClick={() => setFilter('plate')}
        >
          <span className="filter-icon">🚗</span>
          License Plates
        </button>
        <button 
          className={filter === 'fire' ? 'filter-btn active' : 'filter-btn'}
          onClick={() => setFilter('fire')}
        >
          <span className="filter-icon">🔥</span>
          Fire Detection
        </button>
      </div>
      
      <div className="modern-table-container">
        <div className="modern-table-header">
          <div className="col-time">Time</div>
          <div className="col-type">Type</div>
          <div className="col-details">Details</div>
          <div className="col-demographics">Demographics</div>
          <div className="col-status">Status</div>
        </div>
        
        <div className="modern-table-body">
          {filteredLogs.map((log, index) => (
            <div key={index} className="modern-table-row">
              <div className="col-time">
                {new Date(log.timestamp).toLocaleString()}
              </div>
              <div className="col-type">
                {log.type === 'face' ? '👤 Face' : log.type === 'fire' ? '🔥 Fire' : '🚗 Plate'}
              </div>
              <div className="col-details">
                {log.type === 'face' ? (
                  <span className={log.name === 'Unknown' ? 'unknown-person' : ''}>
                    <strong>{log.name || 'Unknown'}</strong>
                    {log.employee_id && ` (${log.employee_id})`}
                    {log.confidence && log.confidence > 0 && (
                      <span className="confidence-badge">
                        {(log.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                  </span>
                ) : log.type === 'fire' ? (
                  <span>
                    <strong>{log.class ? log.class.toUpperCase() : 'Detection'}</strong>
                    {log.confidence && (
                      <span className="confidence-badge">
                        {(log.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {log.severity && (
                      <span className={`severity-badge ${log.severity}`}>
                        {log.severity.toUpperCase()}
                      </span>
                    )}
                  </span>
                ) : (
                  <span>
                    <strong>{log.plate || log.plate_number || 'Unknown'}</strong>
                    {log.owner && ` - ${log.owner}`}
                    {log.vehicle_type && ` (${log.vehicle_type})`}
                  </span>
                )}
              </div>
              <div className="col-demographics">
                {log.type === 'face' ? (
                  <span className="demographics-text">
                    {formatDemographics(log) || '-'}
                  </span>
                ) : (
                  '-'
                )}
              </div>
              <div className="col-status">
                {log.type === 'face' ? (
                  log.name === 'Unknown' ? (
                    <span className="status-badge unknown">Unknown</span>
                  ) : (
                    <span className="status-badge recognized">Recognized</span>
                  )
                ) : log.type === 'fire' ? (
                  log.alert ? (
                    <span className="status-badge alert">⚠️ ALERT</span>
                  ) : (
                    <span className="status-badge detected">Detected</span>
                  )
                ) : (
                  <span className={`status-badge ${(log.authorised || log.is_authorized) ? 'authorized' : 'unauthorized'}`}>
                    {(log.authorised || log.is_authorized) ? 'Authorized' : 'Unauthorized'}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
        
        {filteredLogs.length === 0 && (
          <div className="no-logs">
            <p>No {filter === 'all' ? '' : filter} logs available</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Logs;