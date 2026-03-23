import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import './demographics_styles.css';  // Additional styles for demographics
import MultiCameraGrid from './components/MultiCameraGrid';

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

// Initial camera definitions (static data outside component to avoid redefinition on re-render)
const INITIAL_CAMERAS = [
  { id: 'CAM-01', status: 'inactive', stream: null, location: 'Main Entrance' },
  { id: 'CAM-02', status: 'inactive', stream: null, location: 'Parking Lot' },
  { id: 'CAM-03', status: 'inactive', stream: null, location: 'Rear Exit' },
  { id: 'CAM-04', status: 'inactive', stream: null, location: 'Lobby' },
  { id: 'CAM-05', status: 'inactive', stream: null, location: 'Corridor A' },
  { id: 'CAM-06', status: 'inactive', stream: null, location: 'Corridor B' },
  { id: 'CAM-07', status: 'inactive', stream: null, location: 'Server Room' },
  { id: 'CAM-08', status: 'inactive', stream: null, location: 'Storage Area' },
  { id: 'CAM-09', status: 'inactive', stream: null, location: 'Emergency Exit' }
];

function App() {
  const [cameras, setCameras] = useState(INITIAL_CAMERAS);
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [videoFrame, setVideoFrame] = useState(null);
  const [recentLogs, setRecentLogs] = useState([]);
  const [ws, setWs] = useState(null);
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(true);
  const [plateDetectionEnabled, setPlateDetectionEnabled] = useState(true);
  const [demographicsEnabled, setDemographicsEnabled] = useState(true);
  const [fireDetectionEnabled, setFireDetectionEnabled] = useState(true);
  const [fireAlerts, setFireAlerts] = useState([]);
  const [harEnabled, setHarEnabled] = useState(true);            // NEW: HAR state
  const [harAlerts, setHarAlerts] = useState([]);                 // NEW: HAR alerts

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
          // Update the videoFrame state
          setVideoFrame(data.frame);
          
          // Update the cameras array for multi-camera view
          setCameras(prevCameras => 
            prevCameras.map(camera => 
              camera.id === 'CAM-01'
                ? { 
                    ...camera, 
                    stream: `data:image/jpeg;base64,${data.frame}`,
                    status: 'active'
                  }
                : camera
            )
          );
          
          // Process alerts from detection results
          const newAlerts = [];
          
          // Face detection alerts
          if (data.face_results) {
            data.face_results.forEach(result => {
              if (result.name === 'Unknown') {
                newAlerts.push({
                  cameraId: 'CAM-01',
                  type: 'unauthorized',
                  severity: 'critical',
                  timestamp: new Date().toISOString(),
                  description: `Unauthorized person detected with ${result.confidence}% confidence`
                });
              }
            });
          }
          
          // Fire detection alerts
          if (data.fire_results && data.fire_results.length > 0) {
            newAlerts.push({
              cameraId: 'CAM-01',
              type: 'fire',
              severity: 'critical',
              timestamp: new Date().toISOString(),
              description: 'Fire detected!'
            });
            setFireAlerts(data.fire_results);
          } else {
            setFireAlerts([]);
          }
          
          // NEW: Human Action Recognition alerts
          if (data.har_results && data.har_results.length > 0) {
            const harDetections = data.har_results.filter(r => r.class !== 'normal');
            if (harDetections.length > 0) {
              harDetections.forEach(det => {
                newAlerts.push({
                  cameraId: 'CAM-01',
                  type: 'har',
                  severity: det.severity || 'high',
                  timestamp: new Date().toISOString(),
                  description: `${det.action_label || det.class.toUpperCase()} detected (${(det.confidence * 100).toFixed(0)}% confidence)`
                });
              });
              setHarAlerts(harDetections);
            } else {
              setHarAlerts([]);
            }
          } else {
            setHarAlerts([]);
          }
          
          // Update alerts
          if (newAlerts.length > 0) {
            setAlerts(prevAlerts => [...newAlerts, ...prevAlerts].slice(0, 100));
          }
          
          // Update recent logs
          const newLogs = [];
          if (data.face_results) {
            data.face_results.forEach(result => {
              newLogs.push({
                type: 'face',
                ...result,
                timestamp: data.timestamp
              });
            });
          }
          if (data.plate_results) {
            data.plate_results.forEach(result => {
              newLogs.push({
                type: 'plate',
                ...result,
                timestamp: data.timestamp
              });
            });
          }
          // NEW: HAR logs
          if (data.har_results) {
            data.har_results.forEach(result => {
              if (result.class !== 'normal') {
                newLogs.push({
                  type: 'har',
                  ...result,
                  timestamp: data.timestamp
                });
              }
            });
          }
          
          if (newLogs.length > 0) {
            setRecentLogs(prevLogs => [...newLogs, ...prevLogs].slice(0, 200));
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
  }, []);

  // Fetch system status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/status`);
      const data = await response.json();
      setSystemStatus(data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
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

  const toggleFaceDetection = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/face/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      const data = await response.json();
      console.log('Face detection toggled:', data);
      setFaceDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling face detection:', error);
    }
  };

  const togglePlateDetection = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/plate/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      const data = await response.json();
      console.log('Plate detection toggled:', data);
      setPlateDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling plate detection:', error);
    }
  };

  const toggleDemographics = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/demographics/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        headers: { 'Content-Type': 'application/json' },
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

  // NEW: Toggle HAR
  const toggleHar = async (enabled) => {
    try {
      const response = await fetch(`${API_BASE}/api/har/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      const data = await response.json();
      console.log('HAR toggled:', data);
      setHarEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling HAR:', error);
    }
  };

  const registerPerson = async (personData) => {
    try {
      const response = await fetch(`${API_BASE}/api/persons/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(plateData)
      });
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error registering plate:', error);
      throw error;
    }
  };

  // Make toggle functions available globally for Dashboard component
  useEffect(() => {
    window.toggleDemographics = toggleDemographics;
    window.toggleFireDetection = toggleFireDetection;
    window.toggleHar = toggleHar;     // NEW
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
    
    return () => {
      delete window.toggleDemographics;
      delete window.toggleFireDetection;
      delete window.toggleHar;
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
          <MultiCameraGrid 
            cameras={cameras}
            alerts={alerts}
            onCameraClick={(camera) => console.log('Camera clicked:', camera)}
            systemStatus={systemStatus}
            onToggleFaceDetection={toggleFaceDetection}
            onTogglePlateDetection={togglePlateDetection}
            onToggleDemographics={toggleDemographics}
            onToggleFireDetection={toggleFireDetection}
            onToggleHar={toggleHar}                          // NEW
            faceDetectionEnabled={faceDetectionEnabled}
            plateDetectionEnabled={plateDetectionEnabled}
            demographicsEnabled={demographicsEnabled}
            fireDetectionEnabled={fireDetectionEnabled}
            harEnabled={harEnabled}                          // NEW
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
                <div className="performance-stat">
                  <span className="performance-stat-label">HAR Detections</span>
                  <span className="performance-stat-value">
                    {systemStatus.performance.har_detections || 0}
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