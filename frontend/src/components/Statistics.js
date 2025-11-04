import React from 'react';

const Statistics = ({ systemStatus }) => {
  const stats = systemStatus.statistics || {};
  
  const statItems = [
    { label: 'Total Persons', value: stats.total_persons || 0, icon: '👤', color: '#60a5fa' },
    { label: 'Face Embeddings', value: stats.total_face_embeddings || 0, icon: '🧠', color: '#a78bfa' },
    { label: 'Face Accesses', value: stats.total_face_accesses || 0, icon: '🔍', color: '#34d399' },
    { label: 'License Plates', value: stats.total_plates || 0, icon: '🚗', color: '#fbbf24' },
    { label: 'Authorized Plates', value: stats.authorized_plates || 0, icon: '✅', color: '#10b981' },
    { label: 'Vehicle Accesses', value: stats.total_vehicle_accesses || 0, icon: '🚙', color: '#f59e0b' },
    { label: 'Unauthorized Accesses', value: stats.unauthorized_vehicle_accesses || 0, icon: '❌', color: '#ef4444' }
  ];

  return (
    <div className="modern-statistics-container">
      <div className="statistics-header">
        <h2>📊 System Statistics</h2>
        <p>Overview of all detection system activities and records</p>
      </div>
      
      <div className="modern-stats-grid">
        {statItems.map((stat, index) => (
          <div key={index} className="modern-stat-card">
            <div className="stat-card-icon" style={{ color: stat.color }}>
              {stat.icon}
            </div>
            <div className="stat-card-content">
              <div className="stat-card-value">{stat.value.toLocaleString()}</div>
              <div className="stat-card-label">{stat.label}</div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="modern-system-info">
        <h3>System Information</h3>
        <div className="system-info-grid">
          <div className="info-card">
            <div className="info-card-header">
              <span className="info-icon">📹</span>
              <span className="info-label">Current Mode</span>
            </div>
            <div className="info-card-value">
              {systemStatus.mode ? systemStatus.mode.toUpperCase() : 'Unknown'}
            </div>
          </div>
          <div className="info-card">
            <div className="info-card-header">
              <span className="info-icon">🎥</span>
              <span className="info-label">Camera Status</span>
            </div>
            <div className="info-card-value">
              <span className={systemStatus.streaming ? 'status-badge active' : 'status-badge inactive'}>
                {systemStatus.streaming ? '● Active' : '○ Inactive'}
              </span>
            </div>
          </div>
          <div className="info-card">
            <div className="info-card-header">
              <span className="info-icon">🌐</span>
              <span className="info-label">API Status</span>
            </div>
            <div className="info-card-value">
              <span className="status-badge active">● Online</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics;