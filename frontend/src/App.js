import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

// Components
import Login from './components/Login';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import CameraGrid from './components/CameraGrid';
import IntelligenceSettings from './components/IntelligenceSettings';
import RecentActivity from './components/RecentActivity';
import PersonRegistration from './components/PersonRegistration';
import PlateRegistration from './components/PlateRegistration';
import Logs from './components/Logs';
import Statistics from './components/Statistics';

// API base URL
const API_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

// Initial camera definitions
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
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return !!localStorage.getItem('auth_user');
  });
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // App state
  const [cameras, setCameras] = useState(INITIAL_CAMERAS);
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [recentLogs, setRecentLogs] = useState([]);
  const [ws, setWs] = useState(null);
  const [faceDetectionEnabled, setFaceDetectionEnabled] = useState(true);
  const [plateDetectionEnabled, setPlateDetectionEnabled] = useState(true);
  const [demographicsEnabled, setDemographicsEnabled] = useState(true);
  const [fireDetectionEnabled, setFireDetectionEnabled] = useState(true);
  const [harEnabled, setHarEnabled] = useState(true);
  const [weaponDetectionEnabled, setWeaponDetectionEnabled] = useState(true);

  // WebSocket connection
  useEffect(() => {
    if (!isAuthenticated) return;

    const connectWebSocket = () => {
      const websocket = new WebSocket(WS_URL);

      websocket.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'video_frame') {
          // Update cameras for multi-camera view
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
          }

          // HAR alerts
          if (data.har_results && data.har_results.length > 0) {
            const harDetections = data.har_results.filter(r => r.class !== 'normal');
            harDetections.forEach(det => {
              newAlerts.push({
                cameraId: 'CAM-01',
                type: 'har',
                severity: det.severity || 'high',
                timestamp: new Date().toISOString(),
                description: `${det.action_label || det.class.toUpperCase()} detected (${(det.confidence * 100).toFixed(0)}% confidence)`
              });
            });
          }

          // Weapon detection alerts
          if (data.weapon_results && data.weapon_results.length > 0) {
            data.weapon_results.forEach(det => {
              newAlerts.push({
                cameraId: 'CAM-01',
                type: 'weapon',
                severity: det.severity || 'critical',
                timestamp: new Date().toISOString(),
                description: `${det.class ? det.class.toUpperCase() : 'WEAPON'} detected (${(det.confidence * 100).toFixed(0)}% confidence)`
              });
            });
          }

          // Update alerts
          if (newAlerts.length > 0) {
            setAlerts(prevAlerts => [...newAlerts, ...prevAlerts].slice(0, 100));
          }

          // Update recent logs
          const newLogs = [];
          if (data.face_results) {
            data.face_results.forEach(result => {
              newLogs.push({ type: 'face', ...result, timestamp: data.timestamp });
            });
          }
          if (data.plate_results) {
            data.plate_results.forEach(result => {
              newLogs.push({ type: 'plate', ...result, timestamp: data.timestamp });
            });
          }
          if (data.har_results) {
            data.har_results.forEach(result => {
              if (result.class !== 'normal') {
                newLogs.push({ type: 'har', ...result, timestamp: data.timestamp });
              }
            });
          }
          if (data.weapon_results) {
            data.weapon_results.forEach(result => {
              newLogs.push({ type: 'weapon', ...result, timestamp: data.timestamp });
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
        setTimeout(connectWebSocket, 3000);
      };

      websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setWs(websocket);
    };

    connectWebSocket();

    return () => {
      if (ws) ws.close();
    };
  }, [isAuthenticated]);

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
    if (!isAuthenticated) return;
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus, isAuthenticated]);

  // API functions
  const startCamera = async () => {
    try {
      await fetch(`${API_BASE}/api/camera/start`, { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error('Error starting camera:', error);
    }
  };

  const stopCamera = async () => {
    try {
      await fetch(`${API_BASE}/api/camera/stop`, { method: 'POST' });
      fetchStatus();
    } catch (error) {
      console.error('Error stopping camera:', error);
    }
  };

  const toggleFaceDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/face/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setFaceDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling face detection:', error);
    }
  };

  const togglePlateDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/plate/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setPlateDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling plate detection:', error);
    }
  };

  const toggleDemographics = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/demographics/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setDemographicsEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling demographics:', error);
    }
  };

  const toggleFireDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/fire/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setFireDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling fire detection:', error);
    }
  };

  const toggleHar = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/har/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setHarEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling HAR:', error);
    }
  };

  const toggleWeaponDetection = async (enabled) => {
    try {
      await fetch(`${API_BASE}/api/weapon/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
      setWeaponDetectionEnabled(enabled);
      fetchStatus();
    } catch (error) {
      console.error('Error toggling weapon detection:', error);
    }
  };

  const registerPerson = async (personData) => {
    try {
      const response = await fetch(`${API_BASE}/api/persons/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(personData)
      });
      return await response.json();
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
      return await response.json();
    } catch (error) {
      console.error('Error registering plate:', error);
      throw error;
    }
  };

  const handleLogin = (user) => {
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem('auth_user');
    setIsAuthenticated(false);
    if (ws) ws.close();
  };

  // Request notification permission
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  // Show login if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="min-h-screen bg-[#F0F2F5] font-sans text-slate-900">
      <Header
        onMenuClick={() => setIsSidebarOpen(true)}
        isConnected={isConnected}
        systemStatus={systemStatus}
        onStartCamera={startCamera}
        onStopCamera={stopCamera}
        activeTab={activeTab}
        onLogout={handleLogout}
      />

      <Sidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeTab={activeTab}
        onNavigate={setActiveTab}
      />

      <main className="p-6">
        {activeTab === 'dashboard' && (
          <div className="max-w-[1600px] mx-auto grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Content - Camera Grid */}
            <div className="lg:col-span-3">
              <CameraGrid cameras={cameras} alerts={alerts} />
            </div>

            {/* Right Sidebar - Settings & Activity */}
            <div className="space-y-6">
              <IntelligenceSettings
                faceDetectionEnabled={faceDetectionEnabled}
                plateDetectionEnabled={plateDetectionEnabled}
                demographicsEnabled={demographicsEnabled}
                fireDetectionEnabled={fireDetectionEnabled}
                harEnabled={harEnabled}
                weaponDetectionEnabled={weaponDetectionEnabled}
                onToggleFaceDetection={toggleFaceDetection}
                onTogglePlateDetection={togglePlateDetection}
                onToggleDemographics={toggleDemographics}
                onToggleFireDetection={toggleFireDetection}
                onToggleHar={toggleHar}
                onToggleWeaponDetection={toggleWeaponDetection}
              />
              <RecentActivity alerts={alerts} />
            </div>
          </div>
        )}

        {activeTab === 'person-reg' && (
          <PersonRegistration onRegister={registerPerson} />
        )}

        {activeTab === 'plate-reg' && (
          <PlateRegistration onRegister={registerPlate} />
        )}

        {activeTab === 'logs' && (
          <div className="max-w-[1600px] mx-auto">
            <Logs logs={recentLogs} />
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="max-w-[1600px] mx-auto">
            <Statistics systemStatus={systemStatus} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
