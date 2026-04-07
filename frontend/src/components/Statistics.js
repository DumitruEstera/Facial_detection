import React from 'react';
import { BarChart3, Users, Car, Shield, Eye, Video, Globe, Bell, AlertTriangle } from 'lucide-react';

const Statistics = ({ systemStatus }) => {
  const stats = systemStatus.statistics || {};

  const statItems = [
    { label: 'Total Persons', value: stats.total_persons || 0, color: 'text-blue-600', bg: 'bg-blue-50', Icon: Users },
    { label: 'Face Embeddings', value: stats.total_face_embeddings || 0, color: 'text-purple-600', bg: 'bg-purple-50', Icon: Eye },
    { label: 'Face Accesses', value: stats.total_face_accesses || 0, color: 'text-green-600', bg: 'bg-green-50', Icon: Shield },
    { label: 'License Plates', value: stats.total_plates || 0, color: 'text-amber-600', bg: 'bg-amber-50', Icon: Car },
    { label: 'Authorized Plates', value: stats.authorized_plates || 0, color: 'text-emerald-600', bg: 'bg-emerald-50', Icon: Shield },
    { label: 'Vehicle Accesses', value: stats.total_vehicle_accesses || 0, color: 'text-orange-600', bg: 'bg-orange-50', Icon: Car },
    { label: 'Unauthorized Accesses', value: stats.unauthorized_vehicle_accesses || 0, color: 'text-red-600', bg: 'bg-red-50', Icon: Shield },
    { label: 'Total Alarms', value: stats.total_alarms || 0, color: 'text-indigo-600', bg: 'bg-indigo-50', Icon: Bell },
    { label: 'Unresolved Alarms', value: stats.unresolved_alarms || 0, color: 'text-red-600', bg: 'bg-red-50', Icon: AlertTriangle },
    { label: 'Critical Alarms', value: stats.critical_alarms || 0, color: 'text-rose-600', bg: 'bg-rose-50', Icon: AlertTriangle },
  ];

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <BarChart3 className="w-6 h-6 text-[#3374D0]" />
          <h2 className="text-xl font-semibold text-slate-800">System Statistics</h2>
        </div>
        <p className="text-sm text-gray-500">Overview of all detection system activities and records</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-8">
        {statItems.map((stat, index) => {
          const Icon = stat.Icon;
          return (
            <div key={index} className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 ${stat.bg} rounded-lg flex items-center justify-center`}>
                  <Icon className={`w-5 h-5 ${stat.color}`} />
                </div>
                <div>
                  <div className="text-2xl font-semibold text-slate-800">{stat.value.toLocaleString()}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{stat.label}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* System Info */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-slate-800">System Information</h3>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Video className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">Current Mode</span>
            </div>
            <div className="text-sm font-semibold text-slate-800">
              {systemStatus.mode ? systemStatus.mode.toUpperCase() : 'Unknown'}
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Video className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">Camera Status</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${systemStatus.streaming ? 'bg-green-500' : 'bg-gray-400'}`}></span>
              <span className="text-sm font-semibold text-slate-800">
                {systemStatus.streaming ? 'Active' : 'Inactive'}
              </span>
            </div>
          </div>
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Globe className="w-4 h-4 text-gray-500" />
              <span className="text-xs text-gray-500 font-medium">API Status</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500"></span>
              <span className="text-sm font-semibold text-slate-800">Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Performance Stats */}
      {systemStatus.performance && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden mt-4">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-slate-800">Performance Metrics</h3>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-6">
            <div className="text-center p-3">
              <div className="text-2xl font-semibold text-slate-800">
                {systemStatus.performance.unknown_faces || 0}
              </div>
              <div className="text-xs text-gray-500 mt-1">Unknown Faces</div>
            </div>
            <div className="text-center p-3">
              <div className="text-2xl font-semibold text-slate-800">
                {systemStatus.performance.demographics_analyzed || 0}
              </div>
              <div className="text-xs text-gray-500 mt-1">Demographics Analyzed</div>
            </div>
            <div className="text-center p-3">
              <div className="text-2xl font-semibold text-slate-800">
                {systemStatus.performance.har_detections || 0}
              </div>
              <div className="text-xs text-gray-500 mt-1">HAR Detections</div>
            </div>
            <div className="text-center p-3">
              <div className="text-2xl font-semibold text-slate-800">
                {systemStatus.performance.weapon_detections || 0}
              </div>
              <div className="text-xs text-gray-500 mt-1">Weapon Detections</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Statistics;
