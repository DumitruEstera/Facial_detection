import React, { useState } from 'react';
import { X, Video, VideoOff } from 'lucide-react';

const CameraGrid = ({ cameras = [], alerts = [] }) => {
  const [selectedCamera, setSelectedCamera] = useState(null);

  // Create array of 9 cameras with fallback
  const cameraSlots = Array(9).fill(null).map((_, index) =>
    cameras[index] || { id: `CAM-0${index + 1}`, status: 'inactive', stream: null, location: '' }
  );

  const featuredCamera = cameraSlots[0];
  const smallGridCameras = cameraSlots.slice(1, 5);
  const bottomRowCameras = cameraSlots.slice(5, 9);

  // Check if camera has active alerts
  const hasAlert = (cameraId) => {
    return alerts.some(alert => {
      const alertTime = new Date(alert.timestamp);
      const now = new Date();
      return alert.cameraId === cameraId && (now - alertTime) < 24 * 60 * 60 * 1000;
    });
  };

  const now = new Date();
  const timestamp = now.toISOString().replace('T', ' ').substring(0, 19);

  const renderCameraSlot = (camera, size = 'small') => {
    const isActive = camera.status === 'active' && camera.stream;
    const hasActiveAlert = hasAlert(camera.id);

    return (
      <div
        key={camera.id}
        className="bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm group cursor-pointer"
        onClick={() => isActive && setSelectedCamera(camera)}
      >
        <div className={`relative ${size === 'featured' ? 'aspect-video' : 'aspect-video'} bg-black`}>
          {isActive ? (
            <img
              src={camera.stream}
              alt={`Camera ${camera.id}`}
              className={`w-full h-full object-cover ${size === 'featured' ? 'opacity-90' : 'opacity-80 group-hover:opacity-100 transition-opacity'}`}
            />
          ) : (
            <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
              <VideoOff className="w-8 h-8 mb-2 opacity-40" />
              <span className="text-xs opacity-60">Offline</span>
            </div>
          )}

          {/* Alert indicator */}
          {hasActiveAlert && (
            <div className="absolute top-2 right-2 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
          )}

          {/* Camera ID badge */}
          <div className={`absolute top-${size === 'featured' ? '4' : '2'} left-${size === 'featured' ? '4' : '2'}`}>
            <span className={`px-${size === 'featured' ? '2' : '1.5'} py-${size === 'featured' ? '1' : '0.5'} bg-black/50 backdrop-blur-md text-white text-[${size === 'featured' ? '10px' : '9px'}] font-medium rounded uppercase tracking-wider`}>
              {camera.id}
            </span>
          </div>

          {/* Timestamp for featured */}
          {size === 'featured' && isActive && (
            <div className="absolute top-4 right-4">
              <span className="px-2 py-1 bg-black/50 backdrop-blur-md text-white text-[10px] font-mono rounded">
                {timestamp}
              </span>
            </div>
          )}

          {/* Location label for small cameras */}
          {size === 'small' && camera.location && (
            <div className="absolute bottom-2 left-2">
              <span className="px-1.5 py-0.5 bg-black/50 backdrop-blur-md text-white text-[9px] font-medium rounded uppercase tracking-wider">
                {camera.location}
              </span>
            </div>
          )}

          {/* Status dot */}
          <div className={`absolute bottom-2 right-2 w-2 h-2 rounded-full ${
            isActive ? 'bg-green-500' : 'bg-gray-500'
          }`} />
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-800">Camera Grid</h2>
      </div>

      {/* Featured + Small Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Featured Camera */}
        <div className="xl:col-span-2">
          {renderCameraSlot(featuredCamera, 'featured')}
        </div>

        {/* Small 2x2 Grid */}
        <div className="grid grid-cols-2 gap-4 h-fit">
          {smallGridCameras.map((cam) => renderCameraSlot(cam, 'small'))}
        </div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {bottomRowCameras.map((cam) => renderCameraSlot(cam, 'small'))}
      </div>

      {/* Camera Modal */}
      {selectedCamera && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedCamera(null)}
        >
          <div
            className="bg-white rounded-xl shadow-2xl w-full max-w-4xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
              <div className="flex items-center gap-3">
                <Video className="w-5 h-5 text-[#3374D0]" />
                <h2 className="text-lg font-semibold text-slate-800">{selectedCamera.id}</h2>
                <span className="text-sm text-gray-500">- {selectedCamera.location || 'Unknown'}</span>
              </div>
              <button
                onClick={() => setSelectedCamera(null)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-gray-500" />
              </button>
            </div>
            <div className="bg-black aspect-video">
              {(() => {
                const liveCamera = cameras.find(cam => cam.id === selectedCamera.id);
                const stream = liveCamera?.stream || selectedCamera.stream;
                return stream ? (
                  <img
                    src={stream}
                    alt={`Camera ${selectedCamera.id}`}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-gray-500">
                    <p>No stream available</p>
                  </div>
                );
              })()}
            </div>
            <div className="px-6 py-4 border-t border-gray-200 flex gap-6 text-sm">
              <div>
                <span className="text-gray-500">Status: </span>
                <span className={`font-medium ${
                  (cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status) === 'active'
                    ? 'text-green-600' : 'text-gray-500'
                }`}>
                  {(cameras.find(cam => cam.id === selectedCamera.id)?.status || selectedCamera.status).toUpperCase()}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Location: </span>
                <span className="font-medium text-slate-700">{selectedCamera.location || 'Not specified'}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraGrid;
