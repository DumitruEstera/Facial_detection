import React from 'react';

const Statistics = ({ systemStatus }) => {
  const stats = systemStatus.statistics || {};
  
  const statItems = [
    { label: 'Total Persons', value: stats.total_persons || 0, icon: '👤' },
    { label: 'Face Embeddings', value: stats.total_face_embeddings || 0, icon: '🧠' },
    { label: 'Face Accesses', value: stats.total_face_accesses || 0, icon: '🔍' },
    { label: 'License Plates', value: stats.total_plates || 0, icon: '🚗' },
    { label: 'Authorized Plates', value: stats.authorized_plates || 0, icon: '✅' },
    { label: 'Vehicle Accesses', value: stats.total_vehicle_accesses || 0, icon: '🚙' },
    { label: 'Unauthorized Accesses', value: stats.unauthorized_vehicle_accesses || 0, icon: '❌' }
  ];

  return (
    <div className="statistics">
      <h2>📊 System Statistics</h2>
      
      <div className="stats-grid">
        {statItems.map((stat, index) => (
          <div key={index} className="stat-card">
            <div className="stat-icon">{stat.icon}</div>
            <div className="stat-content">
              <div className="stat-value">{stat.value.toLocaleString()}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="system-info">
        <h3>System Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <span>Current Mode:</span>
            <span>{systemStatus.mode || 'Unknown'}</span>
          </div>
          <div className="info-item">
            <span>Camera Status:</span>
            <span className={systemStatus.streaming ? 'status-active' : 'status-inactive'}>
              {systemStatus.streaming ? 'Active' : 'Inactive'}
            </span>
          </div>
          <div className="info-item">
            <span>API Status:</span>
            <span className="status-active">Online</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics;