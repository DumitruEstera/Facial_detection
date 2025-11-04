import React from 'react';
import '../ModernDashboard.css';

const ModernLayout = ({ 
  children, 
  activeTab,
  onNavigate,
  isConnected,
  systemStatus,
  onStartCamera,
  onStopCamera
}) => {
  return (
    <div className="modern-dashboard">
      {/* Sidebar */}
      <aside className="modern-sidebar">
        <div className="sidebar-logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
        </div>
        <nav className="sidebar-nav">
          <button 
            className={`sidebar-icon ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => onNavigate && onNavigate('dashboard')}
            title="Dashboard"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M12 16v-4"/>
              <path d="M12 8h.01"/>
            </svg>
          </button>
          <button 
            className={`sidebar-icon ${activeTab === 'person-reg' ? 'active' : ''}`}
            onClick={() => onNavigate && onNavigate('person-reg')}
            title="Register Person"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
              <circle cx="12" cy="7" r="4"/>
            </svg>
          </button>
          <button 
            className={`sidebar-icon ${activeTab === 'plate-reg' ? 'active' : ''}`}
            onClick={() => onNavigate && onNavigate('plate-reg')}
            title="Register Plate"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
              <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
            </svg>
          </button>
          <button 
            className={`sidebar-icon ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => onNavigate && onNavigate('logs')}
            title="Logs"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10 9 9 9 8 9"/>
            </svg>
          </button>
          <button 
            className={`sidebar-icon ${activeTab === 'stats' ? 'active' : ''}`}
            onClick={() => onNavigate && onNavigate('stats')}
            title="Statistics"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="20" x2="12" y2="10"/>
              <line x1="18" y1="20" x2="18" y2="4"/>
              <line x1="6" y1="20" x2="6" y2="16"/>
            </svg>
          </button>
        </nav>
        <div className="sidebar-footer">
          <button 
            className="sidebar-icon"
            onClick={() => window.location.reload()}
            title="Refresh"
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="23 4 23 10 17 10"/>
              <polyline points="1 20 1 14 7 14"/>
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>
            </svg>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="modern-main">
        {/* Top Bar */}
        <header className="modern-header">
          <h1>AI Security Dashboard</h1>
          <div className="header-controls">
            <div className={`connection-badge ${isConnected ? 'connected' : 'disconnected'}`}>
              {isConnected ? 'Connected' : 'Disconnected'}
            </div>
            {activeTab === 'dashboard' && (
              <button 
                className="stream-button"
                onClick={() => {
                  const isCurrentlyStreaming = systemStatus?.streaming || false;
                  if (isCurrentlyStreaming) {
                    onStopCamera && onStopCamera();
                  } else {
                    onStartCamera && onStartCamera();
                  }
                }}
              >
                {(systemStatus?.streaming || false) ? 'Stop Stream' : 'Start Stream'}
              </button>
            )}
          </div>
        </header>

        {/* Page Content */}
        <div className="modern-page-content">
          {children}
        </div>
      </div>
    </div>
  );
};

export default ModernLayout;
