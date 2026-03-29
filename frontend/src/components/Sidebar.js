import React from 'react';
import { LayoutDashboard, UserPlus, Car, FileText, BarChart3, X } from 'lucide-react';

const navItems = [
  { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'person-reg', icon: UserPlus, label: 'Register Person' },
  { id: 'plate-reg', icon: Car, label: 'Register Plate' },
  { id: 'logs', icon: FileText, label: 'Logs' },
  { id: 'stats', icon: BarChart3, label: 'Statistics' },
];

const Sidebar = ({ isOpen, onClose, activeTab, onNavigate }) => {
  if (!isOpen) return null;

  const handleNavigate = (tabId) => {
    onNavigate(tabId);
    onClose();
  };

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/20 backdrop-blur-sm z-50 transition-opacity"
        onClick={onClose}
      />

      {/* Sidebar Panel */}
      <div className="fixed top-0 left-0 h-full w-[280px] bg-white border-r border-gray-200 z-[60] flex flex-col pt-20 shadow-2xl sidebar-slide-in">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <X className="w-5 h-5 text-gray-500" />
        </button>

        <nav className="flex-1 px-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleNavigate(item.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-[#EBF2FF] text-[#3374D0] font-medium'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>
    </>
  );
};

export default Sidebar;
