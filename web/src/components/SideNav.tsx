import React from 'react';
import { Cloud, Map, LineChart, ShieldAlert, History, Activity, Network } from 'lucide-react';
import { ActiveTab } from '../types';

interface SideNavProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  onOpenTopologyModal: () => void;
}

const SideNavComponent: React.FC<SideNavProps> = ({ activeTab, onSelectTab, onOpenTopologyModal }) => {
  return (
    <nav className="fixed left-0 top-0 h-full w-[72px] z-50 border-r border-[#d8d4c8] bg-[#eef0e9] flex flex-col items-center py-4 gap-2 hidden md:flex rounded-none select-none" aria-label="Navigasi Utama">
      {/* App Logo */}
      <div 
        role="button"
        tabIndex={0}
        onClick={() => onSelectTab('map')}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelectTab('map')}
        className="mb-3 flex flex-col items-center gap-1 group cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
        title="AeroCast Gede - Main Dashboard"
        aria-label="AeroCast Gede - Main Dashboard"
      >
        <div className="w-10 h-10 bg-white flex items-center justify-center border border-[#1f6056] transition-colors rounded-md">
          <Cloud className="w-5 h-5 text-[#1f6056]" />
        </div>
        <span className="text-[9px] font-mono-data tracking-wide font-bold text-[#1f6056]">PANG</span>
      </div>

      {/* Navigation Links */}
      <div className="flex-1 flex flex-col gap-1 w-full">
        <button
          onClick={() => onSelectTab('map')}
          aria-current={activeTab === 'map' ? 'page' : undefined}
          className={`w-full h-12 flex flex-col items-center justify-center gap-0.5 transition-all rounded-none relative border-b border-white/5 ${
            activeTab === 'map'
              ? 'text-[#00f0ff] bg-[#00f0ff]/10 border-l-2 border-l-[#00f0ff]'
              : 'text-[#849495] hover:bg-white/5 hover:text-[#dae2fd]'
          }`}
          title="Map View & Radar Nowcast"
        >
          <Map className="w-4 h-4" />
          <span className="font-mono-data text-[9px] font-bold tracking-wider uppercase">Map</span>
        </button>

        <button
          onClick={() => onSelectTab('forecast')}
          aria-current={activeTab === 'forecast' ? 'page' : undefined}
          className={`w-full h-12 flex flex-col items-center justify-center gap-0.5 transition-all rounded-none relative border-b border-white/5 ${
            activeTab === 'forecast'
              ? 'text-[#00f0ff] bg-[#00f0ff]/10 border-l-2 border-l-[#00f0ff]'
              : 'text-[#849495] hover:bg-white/5 hover:text-[#dae2fd]'
          }`}
          title="Probabilistic Weather Forecast"
        >
          <LineChart className="w-4 h-4" />
          <span className="font-mono-data text-[9px] font-bold tracking-wider uppercase">Fcst</span>
        </button>

        <button
          onClick={() => onSelectTab('risk')}
          aria-current={activeTab === 'risk' ? 'page' : undefined}
          className={`w-full h-12 flex flex-col items-center justify-center gap-0.5 transition-all rounded-none relative border-b border-white/5 ${
            activeTab === 'risk'
              ? 'text-[#00f0ff] bg-[#00f0ff]/10 border-l-2 border-l-[#00f0ff]'
              : 'text-[#849495] hover:bg-white/5 hover:text-[#dae2fd]'
          }`}
          title="Mountain Risk Assessment"
        >
          <ShieldAlert className="w-4 h-4" />
          <span className="font-mono-data text-[9px] font-bold tracking-wider uppercase">Risk</span>
        </button>

        <button
          onClick={() => onSelectTab('retrieval')}
          aria-current={activeTab === 'retrieval' ? 'page' : undefined}
          className={`w-full h-12 flex flex-col items-center justify-center gap-0.5 transition-all rounded-none relative border-b border-white/5 ${
            activeTab === 'retrieval'
              ? 'text-[#00f0ff] bg-[#00f0ff]/10 border-l-2 border-l-[#00f0ff]'
              : 'text-[#849495] hover:bg-white/5 hover:text-[#dae2fd]'
          }`}
          title="Retrieval Historical Analogs"
        >
          <History className="w-4 h-4" />
          <span className="font-mono-data text-[9px] font-bold tracking-wider uppercase">Rtrv</span>
        </button>

        <button
          onClick={() => onSelectTab('eval')}
          aria-current={activeTab === 'eval' ? 'page' : undefined}
          className={`w-full h-12 flex flex-col items-center justify-center gap-0.5 transition-all rounded-none relative border-b border-white/5 ${
            activeTab === 'eval'
              ? 'text-[#00f0ff] bg-[#00f0ff]/10 border-l-2 border-l-[#00f0ff]'
              : 'text-[#849495] hover:bg-white/5 hover:text-[#dae2fd]'
          }`}
          title="Model Evaluation Metrics"
        >
          <Activity className="w-4 h-4" />
          <span className="font-mono-data text-[9px] font-bold tracking-wider uppercase">Eval</span>
        </button>

        <div className="w-full h-[1px] bg-white/10 my-2" />

        <button
          onClick={onOpenTopologyModal}
          aria-haspopup="dialog"
          aria-label="Buka Topologi Graf ST-GNN 5 Simpul"
          className="w-full h-12 flex flex-col items-center justify-center gap-0.5 text-[#1f6056] bg-[#dbeae3] hover:bg-[#c9ddd3] border border-[#1f6056] transition-all rounded-none"
          title="ST-GNN 5-Node Graph Topology"
        >
          <Network className="w-4 h-4 text-[#1f6056]" />
          <span className="font-mono-data text-[8px] font-bold text-[#1f6056] tracking-wider uppercase">Graph</span>
        </button>
      </div>

      {/* Footer links are intentionally omitted: the existing settings action did not have a settings view. */}
    </nav>
  );
};

export const SideNav = React.memo(SideNavComponent);

