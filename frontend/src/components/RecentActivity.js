import React from 'react';

const RecentActivity = ({ alerts = [] }) => {
  // Convert real alerts to display format, take last 10
  const recentAlerts = alerts.slice(0, 10);

  const getAlertType = (alert) => {
    if (alert.type === 'unauthorized' || alert.type === 'fire' || alert.type === 'weapon' || alert.type === 'har') {
      return 'alert';
    }
    return 'info';
  };

  const getAlertMessage = (alert) => {
    switch (alert.type) {
      case 'unauthorized': return 'Unauthorized Access Alert';
      case 'fire': return 'Fire Detection Alert';
      case 'har': return 'Action Recognition Alert';
      case 'weapon': return 'Weapon Detected';
      case 'motion': return 'Motion Detected';
      case 'intrusion': return 'Perimeter Breach';
      default: return alert.description || 'System Event';
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-800 mb-6">Recent Activity</h3>

      <div className="space-y-6 max-h-80 overflow-y-auto pr-2">
        {recentAlerts.length > 0 ? (
          recentAlerts.map((alert, index) => (
            <div key={index} className="flex gap-4">
              <div className={`w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 ${
                getAlertType(alert) === 'alert' ? 'bg-red-500' : 'bg-green-500'
              }`} />
              <div className="space-y-1">
                <p className="text-sm text-slate-700 leading-tight">
                  <span className="font-semibold text-slate-900 mr-2">
                    {formatTime(alert.timestamp)}
                  </span>
                  {getAlertMessage(alert)}
                  {alert.cameraId && (
                    <span className="block text-xs text-gray-400 mt-0.5">({alert.cameraId})</span>
                  )}
                </p>
              </div>
            </div>
          ))
        ) : (
          <div className="flex gap-4">
            <div className="w-2.5 h-2.5 rounded-full mt-1.5 shrink-0 bg-green-500" />
            <div className="space-y-1">
              <p className="text-sm text-slate-700 leading-tight">
                <span className="font-semibold text-slate-900 mr-2">Now</span>
                System Operational - No recent alerts
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default RecentActivity;
