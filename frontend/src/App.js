import React, { useState, useEffect, useRef } from 'react';
import { Camera, Users, Shield, AlertTriangle, Settings, Clock, Database, Eye, EyeOff, UserPlus, Save, Power } from 'lucide-react';

const MilitarySecurityInterface = () => {
  const [isSystemActive, setIsSystemActive] = useState(false);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [alerts, setAlerts] = useState([]);
  const [knownPersons, setKnownPersons] = useState([
    { id: 1, name: "Col. Ionescu", role: "Comandant", lastSeen: "2025-06-23 14:30", status: "Authorized" },
    { id: 2, name: "Lt. Popescu", role: "Ofițer Security", lastSeen: "2025-06-23 14:25", status: "Authorized" },
    { id: 3, name: "Sgt. Marinescu", role: "Subofițer", lastSeen: "2025-06-23 14:20", status: "Authorized" }
  ]);
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  
  // Updated for separate first and last name
  const [newPersonFirstName, setNewPersonFirstName] = useState('');
  const [newPersonLastName, setNewPersonLastName] = useState('');
  const [newPersonRank, setNewPersonRank] = useState('STUDENT');
  
  const [systemStats, setSystemStats] = useState({
    totalPersons: 42,
    activeAlerts: 0,
    camerasOnline: 8,
    systemUptime: "2 days, 14 hours"
  });
  const [videoFrame, setVideoFrame] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [databaseStatus, setDatabaseStatus] = useState(null);
  
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  // API base URL
  const API_BASE = 'http://localhost:8000';
  const WS_BASE = 'ws://localhost:8000';

  // Rank options for dropdown
  const rankOptions = [
    'STUDENT', 'SOLDIER', 'CORPORAL', 'SERGEANT', 
    'LIEUTENANT', 'CAPTAIN', 'MAJOR', 'COLONEL', 'GENERAL'
  ];

  // Simulate real-time clock
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Check backend connection on component mount
  useEffect(() => {
    checkBackendConnection();
  }, []);

  // WebSocket connection for video streaming
  useEffect(() => {
    if (isSystemActive) {
      connectWebSocket();
    } else {
      disconnectWebSocket();
    }

    return () => {
      disconnectWebSocket();
    };
  }, [isSystemActive]);

  const checkBackendConnection = async () => {
    try {
      const response = await fetch(`${API_BASE}/`);
      const data = await response.json();
      
      if (response.ok) {
        setIsConnected(true);
        setConnectionError(null);
        console.log('✅ Connected to backend:', data);
        
        // Check database status
        const dbTestResponse = await fetch(`${API_BASE}/api/database-test`);
        const dbTestData = await dbTestResponse.json();
        setDatabaseStatus(dbTestData);
        
        // Update system stats from backend
        const statusResponse = await fetch(`${API_BASE}/api/status`);
        const statusData = await statusResponse.json();
        
        setSystemStats(prev => ({
          ...prev,
          totalPersons: statusData.known_faces || prev.totalPersons
        }));
        
        // Add success alert
        const successAlert = {
          id: Date.now(),
          time: new Date().toLocaleString('ro-RO'),
          type: "System Connection",
          description: `Backend conectat cu succes. Database: ${statusData.database_available ? 'disponibil' : 'indisponibil'}`,
          severity: "Success",
          status: "Resolved"
        };
        setAlerts(prev => [successAlert, ...prev.slice(0, 9)]);
        
      } else {
        throw new Error('Backend not responding');
      }
    } catch (error) {
      setIsConnected(false);
      setConnectionError(error.message);
      console.error('❌ Backend connection failed:', error);
      
      // Add connection error alert
      const errorAlert = {
        id: Date.now(),
        time: new Date().toLocaleString('ro-RO'),
        type: "Connection Error",
        description: "Nu s-a putut conecta la serverul backend. Verificați dacă serverul rulează pe portul 8000.",
        severity: "High",
        status: "Active"
      };
      setAlerts(prev => [errorAlert, ...prev]);
    }
  };

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket(`${WS_BASE}/ws/video`);
      
      wsRef.current.onopen = () => {
        console.log('✅ WebSocket connected');
        setConnectionError(null);
      };
      
      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.frame) {
            // Update video frame
            setVideoFrame(`data:image/jpeg;base64,${data.frame}`);
            
            // Update detected faces
            if (data.faces && Array.isArray(data.faces)) {
              setDetectedFaces(data.faces);
              
              // Process alerts for unknown faces
              data.faces.forEach(face => {
                if (face.name === "Unknown" && Math.random() > 0.8) {
                  const newAlert = {
                    id: Date.now() + Math.random(),
                    time: new Date().toLocaleString('ro-RO'),
                    type: "Unknown Person",
                    description: `Persoană neidentificată detectată (confidence: ${face.confidence.toFixed(2)})`,
                    severity: "High",
                    status: "Active"
                  };
                  setAlerts(prev => [newAlert, ...prev.slice(0, 9)]);
                  setSystemStats(prev => ({ ...prev, activeAlerts: prev.activeAlerts + 1 }));
                }
              });
            }
          } else if (data.error) {
            console.error('WebSocket error:', data.error);
          } else if (data.demo_mode) {
            console.log('Running in demo mode:', data.message);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionError('WebSocket connection failed');
      };
      
      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setVideoFrame(null);
        setDetectedFaces([]);
      };
      
    } catch (error) {
      console.error('Error connecting WebSocket:', error);
      setConnectionError('Failed to connect to video stream');
    }
  };

  const disconnectWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setVideoFrame(null);
    setDetectedFaces([]);
  };

  const handleStartSystem = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/start-system`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setIsSystemActive(true);
        const successAlert = {
          id: Date.now(),
          time: new Date().toLocaleString('ro-RO'),
          type: "System Status",
          description: data.message || "Sistemul de recunoaștere facială a fost activat cu succes",
          severity: "Info",
          status: "Resolved"
        };
        setAlerts(prev => [successAlert, ...prev]);
      } else {
        throw new Error(data.detail || 'Failed to start system');
      }
    } catch (error) {
      console.error('Error starting system:', error);
      const errorAlert = {
        id: Date.now(),
        time: new Date().toLocaleString('ro-RO'),
        type: "System Error",
        description: `Eroare la pornirea sistemului: ${error.message}`,
        severity: "High",
        status: "Active"
      };
      setAlerts(prev => [errorAlert, ...prev]);
    }
  };

  const handleStopSystem = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/stop-system`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setIsSystemActive(false);
        const stopAlert = {
          id: Date.now(),
          time: new Date().toLocaleString('ro-RO'),
          type: "System Status",
          description: data.message || "Sistemul de recunoaștere facială a fost oprit",
          severity: "Warning",
          status: "Resolved"
        };
        setAlerts(prev => [stopAlert, ...prev]);
      } else {
        throw new Error(data.detail || 'Failed to stop system');
      }
    } catch (error) {
      console.error('Error stopping system:', error);
    }
  };

  const handleCaptureFace = async () => {
    // Updated validation for separate names
    if (!newPersonFirstName.trim() || !newPersonLastName.trim()) {
      alert("Vă rugăm să introduceți prenumele și numele persoanei!");
      return;
    }
    
    setIsCapturing(true);
    
    try {
      // Updated API call with separate first_name and last_name
      const params = new URLSearchParams({
        first_name: newPersonFirstName.trim(),
        last_name: newPersonLastName.trim(),
        rank: newPersonRank
      });
      
      const response = await fetch(`${API_BASE}/api/register-face?${params}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const data = await response.json();
      
      if (response.ok) {
        const newPerson = {
          id: knownPersons.length + 1,
          name: `${newPersonFirstName} ${newPersonLastName}`,
          role: newPersonRank,
          lastSeen: new Date().toLocaleString('ro-RO'),
          status: "Authorized"
        };
        
        setKnownPersons(prev => [...prev, newPerson]);
        setSystemStats(prev => ({ 
          ...prev, 
          totalPersons: data.total_known_faces || prev.totalPersons + 1 
        }));
        
        // Clear form
        setNewPersonFirstName('');
        setNewPersonLastName('');
        setNewPersonRank('STUDENT');
        
        const captureAlert = {
          id: Date.now(),
          time: new Date().toLocaleString('ro-RO'),
          type: "Face Registration",
          description: data.message || `Noua față a fost înregistrată cu succes pentru: ${newPersonFirstName} ${newPersonLastName}`,
          severity: "Success",
          status: "Resolved"
        };
        setAlerts(prev => [captureAlert, ...prev]);
      } else {
        // Better error handling
        let errorMessage = 'Failed to register face';
        
        if (data.detail) {
          if (Array.isArray(data.detail)) {
            errorMessage = data.detail.map(err => err.msg || err.message || err).join(', ');
          } else if (typeof data.detail === 'object') {
            errorMessage = JSON.stringify(data.detail);
          } else {
            errorMessage = data.detail;
          }
        }
        
        throw new Error(errorMessage);
      }
    } catch (error) {
      console.error('Error registering face:', error);
      let errorMessage = error.message;
      
      // Handle cases where error.message might be an object
      if (typeof errorMessage === 'object') {
        errorMessage = JSON.stringify(errorMessage);
      }
      
      const errorAlert = {
        id: Date.now(),
        time: new Date().toLocaleString('ro-RO'),
        type: "Registration Error",
        description: `Eroare la înregistrarea feței: ${errorMessage}`,
        severity: "High",
        status: "Active"
      };
      setAlerts(prev => [errorAlert, ...prev]);
    } finally {
      setIsCapturing(false);
    }
  };

  const clearAlert = (alertId) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
    setSystemStats(prev => ({ ...prev, activeAlerts: Math.max(0, prev.activeAlerts - 1) }));
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'High': return 'text-red-400 bg-red-900/20 border-red-500';
      case 'Warning': return 'text-yellow-400 bg-yellow-900/20 border-yellow-500';
      case 'Success': return 'text-green-400 bg-green-900/20 border-green-500';
      default: return 'text-blue-400 bg-blue-900/20 border-blue-500';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Shield className="h-8 w-8 text-blue-400" />
            <div>
              <h1 className="text-xl font-bold">Sistem Securitate Militară</h1>
              <p className="text-sm text-gray-400">Recunoaștere Facială YUNET + PostgreSQL - v2.0</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <div className="text-sm text-gray-400">Data și ora curentă</div>
              <div className="font-mono text-lg">
                {currentTime.toLocaleString('ro-RO')}
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                {isConnected ? 'Backend Conectat' : 'Backend Deconectat'}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className={`h-3 w-3 rounded-full ${databaseStatus?.status === 'success' ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                {databaseStatus?.status === 'success' ? 'Database Conectat' : 'Database Deconectat'}
              </span>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className={`h-3 w-3 rounded-full ${isSystemActive ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
              <span className="text-sm">
                {isSystemActive ? 'Sistem Activ' : 'Sistem Oprit'}
              </span>
            </div>
          </div>
        </div>
        
        {/* Connection Error Banner */}
        {connectionError && (
          <div className="mt-4 bg-red-900/20 border border-red-500 text-red-400 px-4 py-2 rounded flex items-center justify-between">
            <span>Eroare conexiune: {connectionError}</span>
            <button 
              onClick={checkBackendConnection}
              className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
            >
              Reconectare
            </button>
          </div>
        )}
        
        {/* Database Status Banner */}
        {databaseStatus && databaseStatus.status === 'error' && (
          <div className="mt-4 bg-yellow-900/20 border border-yellow-500 text-yellow-400 px-4 py-2 rounded">
            <span>Database Warning: {databaseStatus.message}</span>
          </div>
        )}
      </header>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* System Stats */}
          <div className="bg-gray-800 p-4 border-b border-gray-700">
            <div className="grid grid-cols-4 gap-4">
              <div className="bg-gray-700 p-3 rounded-lg flex items-center space-x-3">
                <Users className="h-6 w-6 text-blue-400" />
                <div>
                  <div className="text-sm text-gray-400">Persoane Cunoscute</div>
                  <div className="text-xl font-bold">{systemStats.totalPersons}</div>
                </div>
              </div>
              
              <div className="bg-gray-700 p-3 rounded-lg flex items-center space-x-3">
                <AlertTriangle className="h-6 w-6 text-red-400" />
                <div>
                  <div className="text-sm text-gray-400">Alerte Active</div>
                  <div className="text-xl font-bold">{systemStats.activeAlerts}</div>
                </div>
              </div>
              
              <div className="bg-gray-700 p-3 rounded-lg flex items-center space-x-3">
                <Camera className="h-6 w-6 text-green-400" />
                <div>
                  <div className="text-sm text-gray-400">Camere Online</div>
                  <div className="text-xl font-bold">{systemStats.camerasOnline}</div>
                </div>
              </div>
              
              <div className="bg-gray-700 p-3 rounded-lg flex items-center space-x-3">
                <Database className="h-6 w-6 text-purple-400" />
                <div>
                  <div className="text-sm text-gray-400">Database Status</div>
                  <div className="text-sm font-bold">
                    {databaseStatus?.status === 'success' ? 'Conectat' : 'Deconectat'}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Video Feed */}
          <div className="flex-1 p-6">
            <div className="h-full bg-black rounded-lg relative overflow-hidden border-2 border-gray-600">
              {/* Real Video Stream or Placeholder */}
              <div className="absolute inset-0">
                {videoFrame ? (
                  <div className="relative w-full h-full">
                    <img 
                      src={videoFrame} 
                      alt="Video Feed" 
                      className="w-full h-full object-contain"
                      ref={videoRef}
                    />
                    
                    {/* Camera info overlay */}
                    <div className="absolute top-4 left-4 bg-black/70 p-2 rounded text-sm">
                      <div>Camera 1 - Intrare Principală</div>
                      <div className="text-green-400">LIVE • Database Facial Recognition Active</div>
                      <div className="text-gray-400">Detector: YUNET | Database: PostgreSQL + pgvector</div>
                      <div className="text-blue-400">Faces: {detectedFaces.length}</div>
                    </div>
                    
                    {/* Recording indicator */}
                    <div className="absolute top-4 right-4 flex items-center space-x-2 bg-red-600 px-3 py-1 rounded">
                      <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                      <span className="text-sm font-bold">REC</span>
                    </div>
                    
                    {/* Processing indicator */}
                    {detectedFaces.length > 0 && (
                      <div className="absolute bottom-4 left-4 bg-blue-600 px-3 py-1 rounded text-sm">
                        Procesare: {detectedFaces.length} față(e) detectată(e)
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full bg-gradient-to-br from-gray-800 to-gray-900">
                    {isSystemActive ? (
                      <div className="text-center">
                        <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400 mx-auto mb-4"></div>
                        <p className="text-gray-400 text-lg">Se conectează la camera...</p>
                        <p className="text-gray-500 text-sm">Așteptați stream-ul video</p>
                      </div>
                    ) : (
                      <div className="text-center">
                        <Camera className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                        <p className="text-gray-400 text-lg">Sistemul nu este activ</p>
                        <p className="text-gray-500 text-sm">Apăsați butonul de pornire pentru a începe monitorizarea</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="bg-gray-800 p-4 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <button
                  onClick={isSystemActive ? handleStopSystem : handleStartSystem}
                  disabled={!isConnected || (databaseStatus?.status !== 'success')}
                  className={`px-6 py-2 rounded-lg font-semibold flex items-center space-x-2 transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed ${
                    isSystemActive 
                      ? 'bg-red-600 hover:bg-red-700 text-white' 
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  <Power className="h-4 w-4" />
                  <span>{isSystemActive ? 'Oprire Sistem' : 'Pornire Sistem'}</span>
                </button>
                
                {(!isConnected || databaseStatus?.status !== 'success') && (
                  <span className="text-red-400 text-sm">
                    {!isConnected ? 'Backend deconectat' : 'Database deconectat'}
                  </span>
                )}
              </div>

              {/* Updated Registration Form */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="text"
                    placeholder="Prenume"
                    value={newPersonFirstName}
                    onChange={(e) => setNewPersonFirstName(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 w-28"
                    disabled={!isSystemActive || !isConnected || databaseStatus?.status !== 'success'}
                  />
                  <input
                    type="text"
                    placeholder="Nume"
                    value={newPersonLastName}
                    onChange={(e) => setNewPersonLastName(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 w-28"
                    disabled={!isSystemActive || !isConnected || databaseStatus?.status !== 'success'}
                  />
                  <select
                    value={newPersonRank}
                    onChange={(e) => setNewPersonRank(e.target.value)}
                    className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:border-blue-500"
                    disabled={!isSystemActive || !isConnected || databaseStatus?.status !== 'success'}
                  >
                    {rankOptions.map(rank => (
                      <option key={rank} value={rank}>{rank}</option>
                    ))}
                  </select>
                  <button
                    onClick={handleCaptureFace}
                    disabled={
                      !isSystemActive || 
                      isCapturing || 
                      !newPersonFirstName.trim() || 
                      !newPersonLastName.trim() || 
                      !isConnected ||
                      databaseStatus?.status !== 'success'
                    }
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold flex items-center space-x-2 transition-colors"
                  >
                    <UserPlus className="h-4 w-4" />
                    <span>{isCapturing ? 'Înregistrare...' : 'Înregistrare Față'}</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col">
          {/* Alerts Section */}
          <div className="p-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <AlertTriangle className="h-5 w-5 mr-2 text-red-400" />
              Alerte Securitate
            </h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.length === 0 ? (
                <p className="text-gray-400 text-sm">Nicio alertă activă</p>
              ) : (
                alerts.slice(0, 5).map((alert) => (
                  <div key={alert.id} className={`p-3 rounded border ${getSeverityColor(alert.severity)}`}>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="text-sm font-semibold">{alert.type}</div>
                        <div className="text-xs text-gray-300 mt-1">{alert.description}</div>
                        <div className="text-xs text-gray-400 mt-1">{alert.time}</div>
                      </div>
                      <button
                        onClick={() => clearAlert(alert.id)}
                        className="text-gray-400 hover:text-white ml-2"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Known Persons */}
          <div className="flex-1 p-4">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Database className="h-5 w-5 mr-2 text-green-400" />
              Personal Autorizat
            </h3>
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {knownPersons.map((person) => (
                <div key={person.id} className="bg-gray-700 p-3 rounded">
                  <div className="font-semibold text-sm">{person.name}</div>
                  <div className="text-xs text-gray-400">{person.role}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    Ultima apariție: {person.lastSeen}
                  </div>
                  <div className="mt-2">
                    <span className="text-xs bg-green-900 text-green-300 px-2 py-1 rounded">
                      {person.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MilitarySecurityInterface;