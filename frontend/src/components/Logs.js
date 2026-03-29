import React, { useState } from 'react';
import { FileText } from 'lucide-react';

const filterButtons = [
  { id: 'all', label: 'All' },
  { id: 'face', label: 'Face' },
  { id: 'plate', label: 'Plate' },
  { id: 'fire', label: 'Fire' },
  { id: 'har', label: 'Action' },
  { id: 'weapon', label: 'Weapon' },
];

const Logs = ({ logs }) => {
  const [filter, setFilter] = useState('all');

  const filteredLogs = logs.filter(log => {
    if (filter === 'all') return true;
    return log.type === filter;
  });

  const formatDemographics = (log) => {
    if (!log || log.name !== 'Unknown') return '';
    const parts = [];
    if (log.age) parts.push(`Age: ${log.age}`);
    if (log.gender) parts.push(log.gender);
    if (log.emotion) parts.push(log.emotion);
    return parts.join(', ');
  };

  const getTypeLabel = (type) => {
    switch (type) {
      case 'face': return 'Face';
      case 'plate': return 'Plate';
      case 'fire': return 'Fire';
      case 'har': return 'Action';
      case 'weapon': return 'Weapon';
      default: return type;
    }
  };

  const getTypeColor = (type) => {
    switch (type) {
      case 'face': return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'plate': return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'fire': return 'bg-red-50 text-red-700 border-red-200';
      case 'har': return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'weapon': return 'bg-rose-50 text-rose-700 border-rose-200';
      default: return 'bg-gray-50 text-gray-700 border-gray-200';
    }
  };

  const getStatusBadge = (log) => {
    if (log.type === 'face') {
      return log.name === 'Unknown'
        ? <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">Unknown</span>
        : <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">Recognized</span>;
    }
    if (log.type === 'fire') {
      return log.alert
        ? <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">ALERT</span>
        : <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">Detected</span>;
    }
    if (log.type === 'har') {
      return log.severity === 'critical'
        ? <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">CRITICAL</span>
        : <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-purple-50 text-purple-700 border border-purple-200">{log.severity ? log.severity.toUpperCase() : 'Detected'}</span>;
    }
    if (log.type === 'weapon') {
      return <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">THREAT</span>;
    }
    if (log.type === 'plate') {
      return (log.authorised || log.is_authorized)
        ? <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">Authorized</span>
        : <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">Unauthorized</span>;
    }
    return null;
  };

  const getDetails = (log) => {
    if (log.type === 'face') {
      return (
        <div>
          <span className={`font-medium ${log.name === 'Unknown' ? 'text-red-600' : 'text-slate-800'}`}>
            {log.name || 'Unknown'}
          </span>
          {log.employee_id && <span className="text-gray-500 ml-1">({log.employee_id})</span>}
          {log.confidence > 0 && (
            <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {(log.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
      );
    }
    if (log.type === 'fire') {
      return (
        <div>
          <span className="font-medium text-slate-800">{log.class ? log.class.toUpperCase() : 'Detection'}</span>
          {log.confidence && (
            <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {(log.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
      );
    }
    if (log.type === 'har') {
      return (
        <div>
          <span className="font-medium text-slate-800">{log.action_label || (log.class ? log.class.toUpperCase() : 'Detection')}</span>
          {log.confidence && (
            <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {(log.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
      );
    }
    if (log.type === 'weapon') {
      return (
        <div>
          <span className="font-medium text-slate-800">{log.class ? log.class.toUpperCase() : 'WEAPON'}</span>
          {log.confidence && (
            <span className="ml-2 px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs">
              {(log.confidence * 100).toFixed(1)}%
            </span>
          )}
        </div>
      );
    }
    // Plate
    return (
      <div>
        <span className="font-medium text-slate-800">{log.plate || log.plate_number || 'Unknown'}</span>
        {log.owner && <span className="text-gray-500 ml-1">- {log.owner}</span>}
        {log.vehicle_type && <span className="text-gray-500 ml-1">({log.vehicle_type})</span>}
      </div>
    );
  };

  return (
    <div>
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <FileText className="w-6 h-6 text-[#3374D0]" />
          <h2 className="text-xl font-semibold text-slate-800">Activity Logs</h2>
        </div>
        <p className="text-sm text-gray-500">View and filter detection activity across all systems</p>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap gap-2 mb-6">
        {filterButtons.map(btn => (
          <button
            key={btn.id}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              filter === btn.id
                ? 'bg-[#3374D0] text-white'
                : 'bg-white text-gray-600 border border-gray-200 hover:bg-gray-50'
            }`}
            onClick={() => setFilter(btn.id)}
          >
            {btn.label}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {/* Table Header */}
        <div className="grid grid-cols-12 gap-4 px-6 py-3 border-b border-gray-200 bg-gray-50 text-xs font-medium text-gray-500 uppercase tracking-wider">
          <div className="col-span-3">Time</div>
          <div className="col-span-1">Type</div>
          <div className="col-span-4">Details</div>
          <div className="col-span-2">Demographics</div>
          <div className="col-span-2">Status</div>
        </div>

        {/* Table Body */}
        <div className="divide-y divide-gray-100 max-h-[600px] overflow-y-auto">
          {filteredLogs.map((log, index) => (
            <div key={index} className="grid grid-cols-12 gap-4 px-6 py-3 items-center text-sm hover:bg-gray-50 transition-colors">
              <div className="col-span-3 text-gray-600">
                {new Date(log.timestamp).toLocaleString()}
              </div>
              <div className="col-span-1">
                <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium border ${getTypeColor(log.type)}`}>
                  {getTypeLabel(log.type)}
                </span>
              </div>
              <div className="col-span-4 text-slate-700">
                {getDetails(log)}
              </div>
              <div className="col-span-2 text-gray-500 text-xs">
                {log.type === 'face' ? (formatDemographics(log) || '-') : '-'}
              </div>
              <div className="col-span-2">
                {getStatusBadge(log)}
              </div>
            </div>
          ))}
        </div>

        {filteredLogs.length === 0 && (
          <div className="px-6 py-12 text-center text-gray-500">
            <p className="text-sm">No {filter === 'all' ? '' : filter} logs available</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Logs;
