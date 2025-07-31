import React, { useState, useEffect, useCallback } from 'react';
import './App.css';

// Import components at the top
import Dashboard from './components/Dashboard';
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
          
          // Add new logs
          const newLogs = [];
          if (data.face_results) {
            data.face_results.forEach(result => {
              newLogs.push({
                type: 'face',
                timestamp: data.timestamp,
                ...result
              });
            });
          }
          if (data.plate_results) {
            data.plate_results.forEach(result => {
              newLogs.push({
                type: 'plate',
                timestamp: data.timestamp,
                ...result
              });
            });
          }
          
          if (newLogs.length > 0) {
            setRecentLogs(prev => [...newLogs, ...prev].slice(0, 100));
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

  return (
    <div className="App">
      <header className="app-header">
        <h1>🔒 Security System Dashboard</h1>
        <div className="status-indicator">
          <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>

      <nav className="navigation">
        <button 
          className={activeTab === 'dashboard' ? 'active' : ''}
          onClick={() => setActiveTab('dashboard')}
        >
          📹 Dashboard
        </button>
        <button 
          className={activeTab === 'person-reg' ? 'active' : ''}
          onClick={() => setActiveTab('person-reg')}
        >
          👤 Register Person
        </button>
        <button 
          className={activeTab === 'plate-reg' ? 'active' : ''}
          onClick={() => setActiveTab('plate-reg')}
        >
          🚗 Register Plate
        </button>
        <button 
          className={activeTab === 'logs' ? 'active' : ''}
          onClick={() => setActiveTab('logs')}
        >
          📋 Logs
        </button>
        <button 
          className={activeTab === 'stats' ? 'active' : ''}
          onClick={() => setActiveTab('stats')}
        >
          📊 Statistics
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'dashboard' && (
          <Dashboard 
            videoFrame={videoFrame}
            systemStatus={systemStatus}
            recentLogs={recentLogs}
            onStartCamera={startCamera}
            onStopCamera={stopCamera}
            onSetMode={setMode}
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
          <Statistics systemStatus={systemStatus} />
        )}
      </main>
    </div>
  );
}

export default App;