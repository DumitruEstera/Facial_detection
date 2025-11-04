import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import './demographics_styles.css';  // Additional styles for demographics

// Import components
import Dashboard from './components/Dashboard';
import ModernDashboard from './components/ModernDashboard';
import ModernLayout from './components/ModernLayout';
import PersonRegistration from './components/PersonRegistration';
import PlateRegistration from './components/PlateRegistration';
import Logs from './components/Logs';
import Statistics from './components/Statistics';

// API base URL
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [ws, setWs] = useState(null);
  const [demographicsEnabled, setDemographicsEnabled] = useState(true);
  const [fireDetectionEnabled, setFireDetectionEnabled] = useState(true);
  const [fireAlerts, setFireAlerts] = useState([]);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const websocket = new WebSocket(WS_URL);
      
      websocket.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };
      
      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'video_frame') {
          setVideoFrame(data.frame);
          
          // Process face results with demographics
          const newLogs = [];
          if (data.face_results) {
            data.face_results.forEach(result => {
              const logEntry = {
                type: 'face',
                timestamp: data.timestamp,
                ...result
              };
              
              // Mark if has demographics
              if (result.name === 'Unknown' && (result.age || result.gender || result.emotion)) {
                logEntry.hasDemographics = true;
              }
              
              newLogs.push(logEntry);
            });
          }
          
          // Process plate results
          if (data.plate_results) {
            data.plate_results.forEach(result => {
              newLogs.push({
                type: 'plate',
                timestamp: data.timestamp,
                ...result
              });
            });
          }
          
          // Process fire detection results
          if (data.fire_results && data.fire_results.length > 0) {
            const criticalAlerts = data.fire_results.filter(r => r.severity === 'critical');
            const highAlerts = data.fire_results.filter(r => r.severity === 'high');
            
            // Update fire alerts state
            setFireAlerts(data.fire_results);
            
            // Add to logs
            data.fire_results.forEach(result => {
              newLogs.push({
                type: 'fire',
                timestamp: data.timestamp,
                class: result.class,
                confidence: result.confidence,
                severity: result.severity,
                alert: result.alert
              });
            });
            
            // Show browser notification for critical alerts
            if (criticalAlerts.length > 0 && 'Notification' in window) {
              if (Notification.permission === 'granted') {
                new Notification('🚨 CRITICAL FIRE ALERT!', {
                  body: `${criticalAlerts.length} critical fire/smoke detection(s)`,
                  icon: '🔥'
                });
              }
            }
          } else {
            setFireAlerts([]);
          }
          
          if (newLogs.length > 0) {
            setRecentLogs(prev => [...newLogs, ...prev].slice(0, 100));
          }
          
          // Update demographics status if provided
          if (data.demographics_enabled !== undefined) {
            setSystemStatus(prev => ({
              ...prev,
              demographics_enabled: data.demographics_enabled
            }));
          }
          
          // Update fire detection status if provided
          if (data.fire_detection_enabled !== undefined) {
            setSystemStatus(prev => ({
              ...prev,
              fire_detection_enabled: data.fire_detection_enabled
            }));
          }
        }
      };
      
      websocket.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      setWs(websocket);
    };
    
    connectWebSocket();
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []); // Remove ws from dependencies to avoid infinite loop

  // Fetch system status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`);
      const data = await response.json();
      setSystemStatus(data);
      
      // Update demographics state if available
      if (data.demographics_enabled !== undefined) {
        setDemographicsEnabled(data.demographics_enabled);
      }
      
      // Update fire detection state if available
      if (data.fire_detection_enabled !== undefined) {
        setFireDetectionEnabled(data.fire_detection_enabled);
      }
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // API functions
  const startCamera = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/camera/start`, {
        method: 'POST'
      });
      const data = await response.json();
      console.log('Camera started:', data);
      fetchStatus();
    } catch (error) {
      console.error('Error starting camera:', error);
    }
  };

  const stopCamera = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/camera/stop`, {
        method: 'POST'
      });
      const data = await response.json();
      console.log('Camera stopped:', data);
      fetchStatus();
    } catch (error) {
      console.error('Error stopping camera:', error);
    }
  };

  const setMode = async (mode) => {
    try {
      const response = await fetch(`${API_BASE}/api/camera/mode`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ mode })
      });
      const data = await response.json();
      console.log('Mode set:', data);
      fetchStatus();
    } catch (error) {
      console.error('Error setting mode:', error);
    }
  };

  const toggleDemographics = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/demographics/toggle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      });
      const data = await response.json();
      console.log('Demographics toggled:', data);
      setDemographicsEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling demographics:', error);
    }
  };

  const toggleFireDetection = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/fire/toggle`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      });
      const data = await response.json();
      console.log('Fire detection toggled:', data);
      setFireDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling fire detection:', error);
    }
  };

  const registerPerson = async (personData) => {
    try {
      const response = await fetch(`${API_BASE}/api/persons/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(personData)
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error registering person:', error);
      throw error;
    }
  };

  const registerPlate = async (plateData) => {
    try {
      const response = await fetch(`${API_BASE}/api/plates/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(plateData)
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error registering plate:', error);
      throw error;
    }
  };

  // Make toggleDemographics available globally for the Dashboard component
  useEffect(() => {
    window.toggleDemographics = toggleDemographics;
    window.toggleFireDetection = toggleFireDetection;
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
    
    return () => {
      delete window.toggleDemographics;
      delete window.toggleFireDetection;
    };
  }, []);

  return (
    <div className="App">
      <ModernLayout
        activeTab={activeTab}
        onNavigate={setActiveTab}
        isConnected={isConnected}
        systemStatus={systemStatus}
        onStartCamera={startCamera}
        onStopCamera={stopCamera}
      >
        {activeTab === 'dashboard' && (
          <ModernDashboard 
            videoFrame={videoFrame}
            systemStatus={systemStatus}
            recentLogs={recentLogs}
            fireAlerts={fireAlerts}
            onStartCamera={startCamera}
            onStopCamera={stopCamera}
            onSetMode={setMode}
            onToggleDemographics={toggleDemographics}
            onToggleFireDetection={toggleFireDetection}
            demographicsEnabled={demographicsEnabled}
            fireDetectionEnabled={fireDetectionEnabled}
            isConnected={isConnected}
          />
        )}
        {activeTab === 'person-reg' && (
          <PersonRegistration onRegister={registerPerson} />
        )}
        {activeTab === 'plate-reg' && (
          <PlateRegistration onRegister={registerPlate} />
        )}
        {activeTab === 'logs' && (
          <Logs logs={recentLogs} />
        )}
        {activeTab === 'stats' && (
          <div>
            <Statistics systemStatus={systemStatus} />
            {systemStatus.performance && (
              <div className="performance-stats">
                <div className="performance-stat">
                  <span className="performance-stat-label">Unknown Faces</span>
                  <span className="performance-stat-value">
                    {systemStatus.performance.unknown_faces || 0}
                  </span>
                </div>
                <div className="performance-stat">
                  <span className="performance-stat-label">Demographics Analyzed</span>
                  <span className="performance-stat-value">
                    {systemStatus.performance.demographics_analyzed || 0}
                  </span>
                </div>
              </div>
            )}
          </div>
        )}
      </ModernLayout>
    </div>
  );
}

export default App;