import React from 'react';

const Logs = ({ logs }) => {
  const [filter, setFilter] = React.useState('all');
  
  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true;
    return log.type === filter;
  });

  return (
    <div className="logs-section">
      <h2>📋 Activity Logs</h2>
      
      <div className="logs-controls">
        <div className="filter-buttons">
          <button 
            className={filter === 'all' ? 'btn btn-active' : 'btn btn-outline'}
            onClick={() => setFilter('all')}
          >
            All
          </button>
          <button 
            className={filter === 'face' ? 'btn btn-active' : 'btn btn-outline'}
            onClick={() => setFilter('face')}
          >
            👤 Face Recognition
          </button>
          <button 
            className={filter === 'plate' ? 'btn btn-active' : 'btn btn-outline'}
            onClick={() => setFilter('plate')}
          >
            🚗 License Plates
          </button>
        </div>
      </div>
      
      <div className="logs-table">
        <div className="table-header">
          <div>Time</div>
          <div>Type</div>
          <div>Details</div>
          <div>Status</div>
        </div>
        
        {filteredLogs.map((log, index) => (
          <div key={index} className="table-row">
            <div className="log-time">
              {new Date(log.timestamp).toLocaleString()}
            </div>
            <div className="log-type">
              {log.type === 'face' ? '👤 Face' : '🚗 Plate'}
            </div>
            <div className="log-details">
              {log.type === 'face' ? (
                <span>
                  <strong>{log.name || 'Unknown'}</strong>
                  {log.employee_id && ` (${log.employee_id})`}
                  {log.confidence && (
                    <span className="confidence">
                      {(log.confidence * 100).toFixed(1)}%
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
            <div className="log-status">
              {log.type === 'face' ? (
                <span className="status-recognized">Recognized</span>
              ) : (
                <span className={log.authorised || log.is_authorized ? 'status-authorized' : 'status-unauthorized'}>
                  {(log.authorised || log.is_authorized) ? 'Authorized' : 'Unauthorized'}
                </span>
              )}
            </div>
          </div>
        ))}
        
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