import React, { useState, useMemo } from 'react';
import { Search, Bell, User, Menu, X, Check, MapPin } from 'lucide-react';
import { NodeId, ActiveTab } from '../types';
import { NODES_LIST } from '../data/nodesData';

interface TopAppBarProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  radarVisible: boolean;
  onToggleRadar: () => void;
  nodesVisible: boolean;
  onToggleNodes: () => void;
  onSelectNode: (nodeId: NodeId) => void;
  onOpenTopologyModal: () => void;
}

const TopAppBarComponent: React.FC<TopAppBarProps> = ({
  activeTab,
  onSelectTab,
  radarVisible,
  onToggleRadar,
  nodesVisible,
  onToggleNodes,
  onSelectNode,
  onOpenTopologyModal,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearchDropdown, setShowSearchDropdown] = useState(false);
  const [showNotifications, setShowNotifications] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Close open dropdowns/menus on Escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setShowSearchDropdown(false);
        setShowNotifications(false);
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const filteredNodes = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const query = searchQuery.toLowerCase();
    return Object.values(NODES_LIST).filter((node) =>
      node.name.toLowerCase().includes(query) ||
      node.locationName.toLowerCase().includes(query) ||
      node.role.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  return (
    <header className="fixed top-0 right-0 left-0 md:left-[72px] z-40 bg-[#f4f1e9] border-b border-[#d8d4c8] flex justify-between items-center px-4 md:px-6 h-[64px] transition-colors duration-150 select-none">
      {/* Brand & Badges */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-expanded={mobileMenuOpen}
          aria-controls="mobile-nav-menu"
          aria-label="Toggle Menu Navigasi Mobile"
          className="md:hidden text-[#00f0ff] min-w-[44px] min-h-[44px] flex items-center justify-center rounded-lg bg-[#131b2e] border border-white/15 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.08)] focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
        >
          {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
        </button>

        <div 
          role="button"
          tabIndex={0}
          onClick={() => onSelectTab('map')}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelectTab('map')}
          className="flex items-baseline gap-2 cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
          aria-label="AeroCast Gede Dashboard"
        >
          <h1 className="font-headline text-lg md:text-xl font-bold text-[#202720] transition-colors">
            Pangrango Weather Desk
          </h1>
          <span className="text-[10px] font-mono-data text-[#1f6056] font-semibold hidden sm:inline border-l border-[#d8d4c8] pl-2">
            PEMANTAUAN CUACA
          </span>
        </div>

        <div className="hidden lg:flex items-center gap-2 ml-3">
          <span className="px-2 py-0.5 text-[#5f685e] font-mono-data text-[10px] border border-[#d8d4c8] uppercase tracking-wide">
            ERA5 demo dataset
          </span>
          <span className="px-2 py-0.5 bg-[#f5e7c5] text-[#7b4d0b] font-mono-data text-[10px] font-semibold border border-[#b77817] uppercase tracking-wide">
            Data statis
          </span>
        </div>
      </div>

      {/* Layer Quick Toggles on Header */}
      <div className="hidden md:flex items-center gap-2">
        <button
          onClick={onToggleRadar}
          aria-pressed={radarVisible}
          className={`px-3 py-1.5 rounded-lg flex items-center gap-2 text-[10px] font-mono-data font-bold border transition-all ${
            radarVisible
              ? 'bg-[#00f0ff]/15 border-[#00f0ff] text-[#dbfcff] shadow-[0_0_10px_rgba(0,240,255,0.2)]'
              : 'bg-[#131b2e] border-white/15 text-[#94a3b8] hover:text-white shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)]'
          }`}
        >
          <span className={`w-2 h-2 rounded-full ${radarVisible ? 'bg-[#1f6056]' : 'bg-gray-600'}`} />
          SEL ERA5 (5)
          {radarVisible && <Check className="w-3 h-3 text-[#1f6056]" />}
        </button>

        <button
          onClick={onToggleNodes}
          aria-pressed={nodesVisible}
          className={`px-3 py-1.5 rounded-none flex items-center gap-2 text-[10px] font-mono-data font-bold border transition-all ${
            nodesVisible
              ? 'bg-[#dbeae3] border-[#1f6056] text-[#1f6056]'
              : 'bg-[#131b2e] border-white/15 text-[#849495] hover:text-white'
          }`}
        >
          <span className={`w-2 h-2 ${nodesVisible ? 'bg-[#1f6056]' : 'bg-gray-600'}`} />
          SIMPUL GRAPH
          {nodesVisible && <Check className="w-3 h-3 text-[#1f6056]" />}
        </button>

        <button
          onClick={onOpenTopologyModal}
          aria-haspopup="dialog"
          aria-label="Lihat Topologi Graf 5 Simpul"
          className="px-3 py-1.5 rounded-none bg-[#dbeae3] border border-[#1f6056] text-[#1f6056] font-mono-data text-[10px] font-bold hover:bg-[#c9ddd3] transition-colors uppercase tracking-wider"
        >
          TOPOLOGI
        </button>
      </div>

      {/* Search & Profile Actions */}
      <div className="flex items-center gap-2">
        {/* Search Input */}
        <div className="relative hidden sm:block">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-[#849495]" />
            <input
              type="text"
              value={searchQuery}
              role="combobox"
              aria-expanded={showSearchDropdown && searchQuery.trim().length > 0}
              aria-autocomplete="list"
              aria-controls="search-node-results"
              aria-label="Cari simpul observasi atau lokasi"
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setShowSearchDropdown(true);
              }}
              onFocus={() => setShowSearchDropdown(true)}
              placeholder="Cari simpul, koordinat..."
              className="bg-[#131b2e] border border-white/15 rounded-none pl-9 pr-4 py-1.5 text-xs w-[200px] focus:w-[250px] focus:outline-none focus:border-[#00f0ff] transition-all text-[#dae2fd] placeholder:text-[#849495] font-mono-data"
            />
          </div>

          {/* Autocomplete Dropdown */}
          {showSearchDropdown && searchQuery.trim().length > 0 && (
            <div id="search-node-results" role="listbox" className="absolute right-0 top-11 w-[300px] bg-[#070d18] border border-[#00f0ff]/40 rounded-none shadow-2xl p-2 z-50">
              <div className="text-[10px] font-mono-data text-[#849495] px-2 py-1 uppercase tracking-wider font-bold border-b border-white/10 mb-1">
                Area pengamatan
              </div>
              {filteredNodes.length > 0 ? (
                filteredNodes.map((node) => (
                  <button
                    key={node.id}
                    role="option"
                    aria-selected={false}
                    onClick={() => {
                      onSelectNode(node.id);
                      setShowSearchDropdown(false);
                      setSearchQuery('');
                      onSelectTab('map');
                    }}
                    className="w-full text-left px-3 py-2 rounded-none hover:bg-[#00f0ff]/10 flex items-center justify-between group transition-colors border-b border-white/5 last:border-0 focus:outline-none focus:bg-[#00f0ff]/20"
                  >
                    <div>
                      <div className="text-xs font-bold text-[#dbfcff] group-hover:text-[#00f0ff] flex items-center gap-1.5">
                        <MapPin className="w-3 h-3 text-[#00f0ff]" />
                        {node.name} {node.isTarget && '(Target Utama)'}
                      </div>
                      <div className="text-[11px] text-[#849495] font-body truncate">
                        {node.locationName}
                      </div>
                    </div>
                    <span className="font-mono-data text-[10px] text-[#00f0ff]">
                      {node.elevation}m
                    </span>
                  </button>
                ))
              ) : (
                <div className="text-xs text-[#849495] p-3 text-center font-mono-data">
                  Tidak ada simpul yang cocok.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Notifications Button */}
        <div className="relative">
          <button
            onClick={() => setShowNotifications(!showNotifications)}
            aria-expanded={showNotifications}
            aria-haspopup="dialog"
            aria-controls="notifications-drawer"
            aria-label="Peringatan Dini Cuaca"
            className="w-11 h-11 rounded-none flex items-center justify-center text-[#849495] hover:text-[#00f0ff] hover:bg-white/5 transition-colors relative border border-white/15 bg-[#131b2e] focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
            title="Peringatan Dini CUACA"
          >
            <Bell className="w-4.5 h-4.5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-[#ffb4ab] border border-[#070d18] animate-pulse motion-reduce:animate-none" />
          </button>

          {/* Notifications Drawer */}
          {showNotifications && (
            <div id="notifications-drawer" role="dialog" aria-label="Peringatan Dini Cuaca" className="absolute right-0 top-12 w-[320px] bg-[#070d18] border border-[#ffb4ab]/40 rounded-none shadow-2xl p-3 z-50">
              <div className="flex items-center justify-between border-b border-white/10 pb-2 mb-2">
                <span className="font-mono-data text-xs font-bold text-[#ffb4ab] flex items-center gap-1.5 uppercase tracking-wider">
                  <span className="w-2 h-2 bg-[#ffb4ab]" />
                  PERINGATAN DINI CUACA
                </span>
                <button
                  onClick={() => setShowNotifications(false)}
                  aria-label="Tutup Peringatan Dini"
                  className="text-[#849495] hover:text-white min-w-[44px] min-h-[44px] flex items-center justify-center focus:outline-none focus:ring-1 focus:ring-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="space-y-2 text-xs">
                <div className="p-2.5 rounded-none bg-[#93000a]/20 border border-[#ffb4ab]/30">
                  <div className="font-bold text-[#ffdad6] uppercase tracking-wider text-[11px]">Risiko Hipotermia Tinggi</div>
                  <div className="text-[11px] text-[#dae2fd] mt-1 font-body">
                    Puncak Pangrango (3,008 mdpl): Suhu 8.2°C, Windchill -2.4°C Apparent, Kelembapan 98%.
                  </div>
                  <div className="text-[10px] font-mono-data text-[#ffb4ab] mt-1">14:30 WIB</div>
                </div>
                <div className="p-2.5 rounded-none bg-[#131b2e] border border-white/15">
                  <div className="font-bold text-[#dbfcff] uppercase tracking-wider text-[11px]">Awan hujan menguat di lereng barat</div>
                  <div className="text-[11px] text-[#dae2fd] mt-1 font-body">
                    Node LEFT (1,120 mdpl) mencatat intensitas hujan 18.2 mm/jam mengalir ke target MAIN.
                  </div>
                  <div className="text-[10px] font-mono-data text-[#00f0ff] mt-1">14:15 WIB</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* User Profile Badge */}
        <div className="w-9 h-9 rounded-none bg-[#00f0ff]/10 border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff] font-bold text-xs font-mono-data" aria-label="Profil Pengguna">
          <User className="w-4 h-4" />
        </div>
      </div>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div id="mobile-nav-menu" role="navigation" aria-label="Navigasi Mobile" className="absolute top-[64px] left-0 right-0 bg-[#070d18] border-b border-white/15 p-4 z-50 flex flex-col gap-2 md:hidden rounded-none max-h-[calc(100vh-64px)] overflow-y-auto">
          <div className="grid grid-cols-5 gap-1.5">
            <button
              onClick={() => { onSelectTab('map'); setMobileMenuOpen(false); }}
              aria-current={activeTab === 'map' ? 'page' : undefined}
              className={`py-2.5 min-h-[44px] rounded-none text-[11px] font-mono-data font-bold border uppercase tracking-wider flex items-center justify-center ${activeTab === 'map' ? 'bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff]' : 'bg-[#131b2e] border-white/15 text-white'}`}
            >
              MAP
            </button>
            <button
              onClick={() => { onSelectTab('forecast'); setMobileMenuOpen(false); }}
              aria-current={activeTab === 'forecast' ? 'page' : undefined}
              className={`py-2.5 min-h-[44px] rounded-none text-[11px] font-mono-data font-bold border uppercase tracking-wider flex items-center justify-center ${activeTab === 'forecast' ? 'bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff]' : 'bg-[#131b2e] border-white/15 text-white'}`}
            >
              CHART
            </button>
            <button
              onClick={() => { onSelectTab('risk'); setMobileMenuOpen(false); }}
              aria-current={activeTab === 'risk' ? 'page' : undefined}
              className={`py-2.5 min-h-[44px] rounded-none text-[11px] font-mono-data font-bold border uppercase tracking-wider flex items-center justify-center ${activeTab === 'risk' ? 'bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff]' : 'bg-[#131b2e] border-white/15 text-white'}`}
            >
              RISK
            </button>
            <button
              onClick={() => { onSelectTab('retrieval'); setMobileMenuOpen(false); }}
              aria-current={activeTab === 'retrieval' ? 'page' : undefined}
              className={`py-2.5 min-h-[44px] rounded-none text-[11px] font-mono-data font-bold border uppercase tracking-wider flex items-center justify-center ${activeTab === 'retrieval' ? 'bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff]' : 'bg-[#131b2e] border-white/15 text-white'}`}
            >
              FAISS
            </button>
            <button
              onClick={() => { onSelectTab('eval'); setMobileMenuOpen(false); }}
              aria-current={activeTab === 'eval' ? 'page' : undefined}
              className={`py-2.5 min-h-[44px] rounded-none text-[11px] font-mono-data font-bold border uppercase tracking-wider flex items-center justify-center ${activeTab === 'eval' ? 'bg-[#00f0ff]/20 border-[#00f0ff] text-[#00f0ff]' : 'bg-[#131b2e] border-white/15 text-white'}`}
            >
              EVAL
            </button>
          </div>
          <button
            onClick={() => { onOpenTopologyModal(); setMobileMenuOpen(false); }}
            className="w-full py-2.5 min-h-[44px] flex items-center justify-center rounded-none bg-[#dbeae3] border border-[#1f6056] text-[#1f6056] font-mono-data text-xs font-bold uppercase tracking-wider"
          >
            LIHAT TOPOLOGI GRAF 5 NODE
          </button>
        </div>
      )}
    </header>
  );
};

export const TopAppBar = React.memo(TopAppBarComponent);
