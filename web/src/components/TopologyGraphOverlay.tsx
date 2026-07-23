import React, { useState, useEffect, useRef } from 'react';
import { X, Network, Cpu, ArrowRight, Layers, BarChart2, Activity } from 'lucide-react';
import { NodeId, TimelineFrame } from '../types';
import { NODES_LIST } from '../data/nodesData';

interface TopologyGraphOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectNode: (nodeId: NodeId) => void;
  frame: TimelineFrame;
}

export const TopologyGraphOverlay: React.FC<TopologyGraphOverlayProps> = ({
  isOpen,
  onClose,
  onSelectNode,
  frame,
}) => {
  const [selectedGraphNode, setSelectedGraphNode] = useState<NodeId>('MAIN');
  const overlayRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    // Save active element for focus restoration
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Focus overlay container
    if (overlayRef.current) {
      overlayRef.current.focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }

      // Focus trap
      if (e.key === 'Tab' && overlayRef.current) {
        const focusables = overlayRef.current.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];

        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      previousFocusRef.current?.focus();
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const currentTelemetries = frame.telemetries;
  const activeNode = NODES_LIST[selectedGraphNode];
  const nodeIds: NodeId[] = ['UP', 'MAIN', 'DOWN', 'LEFT', 'RIGHT'];

  // Defined vector paths connecting peripheral observation nodes to MAIN target
  const edges = [
    {
      from: 'UP',
      x1: '50%',
      y1: '20%',
      x2: '50%',
      y2: '38%',
      color: '#fed639',
      weight: currentTelemetries.UP?.spatialWeightToMain ?? 0.2,
    },
    {
      from: 'DOWN',
      x1: '50%',
      y1: '80%',
      x2: '50%',
      y2: '62%',
      color: '#d1bcff',
      weight: currentTelemetries.DOWN?.spatialWeightToMain ?? 0.2,
    },
    {
      from: 'LEFT',
      x1: '22%',
      y1: '50%',
      x2: '38%',
      y2: '50%',
      color: '#7df4ff',
      weight: currentTelemetries.LEFT?.spatialWeightToMain ?? 0.2,
    },
    {
      from: 'RIGHT',
      x1: '78%',
      y1: '50%',
      x2: '62%',
      y2: '50%',
      color: '#ffb4ab',
      weight: currentTelemetries.RIGHT?.spatialWeightToMain ?? 0.2,
    },
  ];

  return (
    <div className="fixed inset-0 z-50 bg-[#060b14]/90 backdrop-blur-2xl flex items-center justify-center p-2 sm:p-4 md:p-6 overflow-y-auto animate-in fade-in duration-200">
      <div 
        ref={overlayRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="topology-graph-title"
        className="glass-panel w-full max-w-5xl max-h-[90vh] [@media(max-height:500px)]:max-h-[85vh] overflow-y-auto rounded-3xl border border-[#00f0ff]/30 p-4 sm:p-6 shadow-[0_0_50px_rgba(0,240,255,0.15)] relative my-auto bg-[#0b1326]/95 focus:outline-none"
      >
        {/* Header */}
        <div className="flex justify-between items-center pb-4 mb-5 border-b border-white/10">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-2xl bg-[#00f0ff]/15 border border-[#00f0ff]/40 flex items-center justify-center text-[#00f0ff] shadow-[0_0_15px_rgba(0,240,255,0.2)]">
              <Network className="w-6 h-6" />
            </div>
            <div>
              <h2 id="topology-graph-title" className="font-headline text-lg md:text-xl font-bold text-[#dbfcff] tracking-tight">
                Topologi Spatio-Temporal Graph (ST-GNN Architecture)
              </h2>
              <p className="text-xs text-[#849495] font-body mt-0.5">
                Pemodelan keterkaitan spasial 5 simpul observasi orografis menuju Puncak Mandalawangi
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-11 h-11 rounded-full bg-[#171f33] text-[#849495] hover:text-white hover:bg-white/15 transition-all flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
            aria-label="Tutup Overlay Topologi"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Canvas Star Topology (Left 7 Cols) */}
          <div className="lg:col-span-7 bg-[#040814] rounded-2xl border border-white/10 p-5 flex flex-col justify-between relative min-h-[380px] overflow-hidden">
            <div className="absolute inset-0 bg-[radial-gradient(#00f0ff_1px,transparent_1px)] [background-size:20px_20px] opacity-10 pointer-events-none" />

            <div className="flex justify-between items-center z-10 mb-2">
              <span className="font-mono-data text-xs font-bold text-[#00f0ff] flex items-center gap-2">
                <Cpu className="w-4 h-4 text-[#00f0ff]" /> GRAF STAR TERARAH (A<sub>ij</sub>)
              </span>
              <span className="text-[10px] font-mono-data text-[#00f0ff] bg-[#00f0ff]/10 border border-[#00f0ff]/30 px-2.5 py-1 rounded-full flex items-center gap-1.5">
                <Activity className="w-3 h-3 animate-spin" /> Message Passing Active
              </span>
            </div>

            {/* SVG Interactive Canvas */}
            <div className="relative w-full h-[300px] flex items-center justify-center my-2 select-none" aria-label="Kanvas Interaktif Topologi Graf Star">
              {/* SVG Edges with Flow Animation & Crisp Head Markers */}
              <svg className="absolute inset-0 w-full h-full pointer-events-none z-10">
                <defs>
                  {edges.map((edge) => (
                    <marker
                      key={edge.from}
                      id={`arrow-${edge.from}`}
                      viewBox="0 0 10 10"
                      refX="6"
                      refY="5"
                      markerWidth="6"
                      markerHeight="6"
                      orient="auto-start-reverse"
                    >
                      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill={edge.color} />
                    </marker>
                  ))}
                </defs>
                {edges.map((edge) => (
                  <g key={edge.from}>
                    {/* Outer glow line */}
                    <line
                      x1={edge.x1}
                      y1={edge.y1}
                      x2={edge.x2}
                      y2={edge.y2}
                      stroke={edge.color}
                      strokeWidth="4"
                      strokeOpacity="0.25"
                    />
                    {/* Directional animated flow line */}
                    <line
                      x1={edge.x1}
                      y1={edge.y1}
                      x2={edge.x2}
                      y2={edge.y2}
                      stroke={edge.color}
                      strokeWidth="2.5"
                      strokeDasharray="6,6"
                      markerEnd={`url(#arrow-${edge.from})`}
                      className="edge-flow"
                    />
                  </g>
                ))}
              </svg>

              {/* Edge Weight Tags */}
              <div className="absolute top-[28%] left-[50%] -translate-x-1/2 z-20 bg-[#070e1c] px-2 py-0.5 rounded border border-[#fed639]/40 text-[#fed639] font-mono-data text-[9px] font-bold shadow">
                {(currentTelemetries.UP?.spatialWeightToMain * 100).toFixed(0)}%
              </div>
              <div className="absolute bottom-[28%] left-[50%] -translate-x-1/2 z-20 bg-[#070e1c] px-2 py-0.5 rounded border border-[#d1bcff]/40 text-[#d1bcff] font-mono-data text-[9px] font-bold shadow">
                {(currentTelemetries.DOWN?.spatialWeightToMain * 100).toFixed(0)}%
              </div>
              <div className="absolute top-[50%] left-[28%] -translate-y-1/2 z-20 bg-[#070e1c] px-2 py-0.5 rounded border border-[#7df4ff]/40 text-[#7df4ff] font-mono-data text-[9px] font-bold shadow">
                {(currentTelemetries.LEFT?.spatialWeightToMain * 100).toFixed(0)}%
              </div>
              <div className="absolute top-[50%] right-[28%] -translate-y-1/2 z-20 bg-[#070e1c] px-2 py-0.5 rounded border border-[#ffb4ab]/40 text-[#ffb4ab] font-mono-data text-[9px] font-bold shadow">
                {(currentTelemetries.RIGHT?.spatialWeightToMain * 100).toFixed(0)}%
              </div>

              {/* Node MAIN (Center Target) */}
              <div
                role="button"
                tabIndex={0}
                aria-selected={selectedGraphNode === 'MAIN'}
                aria-label="Simpul MAIN (Target Puncak Mandalawangi 3008 mdpl)"
                onClick={() => setSelectedGraphNode('MAIN')}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedGraphNode('MAIN')}
                className={`absolute z-20 cursor-pointer transform transition-all duration-300 hover:scale-105 flex flex-col items-center focus:outline-none focus:ring-2 focus:ring-[#00f0ff] rounded-2xl ${
                  selectedGraphNode === 'MAIN' ? 'scale-110' : ''
                }`}
                style={{ top: '50%', left: '50%', transform: 'translate(-50%, -50%)' }}
              >
                <div
                  className={`w-20 h-20 rounded-2xl bg-[#00f0ff]/20 border-2 border-[#00f0ff] flex flex-col items-center justify-center shadow-[0_0_30px_rgba(0,240,255,0.5)] ${
                    selectedGraphNode === 'MAIN' ? 'ring-4 ring-[#00f0ff]/50' : ''
                  }`}
                >
                  <span className="font-mono-data text-xs font-black text-[#00f0ff]">MAIN</span>
                  <span className="font-mono-data text-[9px] text-[#dbfcff]">3,008 mdpl</span>
                </div>
                <span className="font-mono-data text-[10px] font-bold text-[#00f0ff] mt-1.5 bg-[#081326] px-2.5 py-0.5 rounded-full border border-[#00f0ff]/40 shadow">
                  Target (Puncak)
                </span>
              </div>

              {/* Node UP */}
              <div
                role="button"
                tabIndex={0}
                aria-selected={selectedGraphNode === 'UP'}
                aria-label="Simpul UP (Utara Cibodas 1450 mdpl)"
                onClick={() => setSelectedGraphNode('UP')}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedGraphNode('UP')}
                className={`absolute z-20 cursor-pointer transform transition-all duration-300 hover:scale-105 flex flex-col items-center focus:outline-none focus:ring-2 focus:ring-[#fed639] rounded-xl ${
                  selectedGraphNode === 'UP' ? 'scale-110' : ''
                }`}
                style={{ top: '4%', left: '50%', transform: 'translate(-50%, 0)' }}
              >
                <div
                  className={`w-13 h-13 p-2 rounded-xl bg-[#fed639]/15 border border-[#fed639] flex flex-col items-center justify-center shadow-[0_0_15px_rgba(254,214,57,0.3)] ${
                    selectedGraphNode === 'UP' ? 'ring-2 ring-[#fed639]' : ''
                  }`}
                >
                  <span className="font-mono-data text-xs font-bold text-[#fed639]">UP</span>
                  <span className="font-mono-data text-[8px] text-[#fff5de]">1,450m</span>
                </div>
                <span className="font-mono-data text-[9px] text-[#b9cacb] mt-1">Utara (Cibodas)</span>
              </div>

              {/* Node DOWN */}
              <div
                role="button"
                tabIndex={0}
                aria-selected={selectedGraphNode === 'DOWN'}
                aria-label="Simpul DOWN (Selatan Sukabumi 980 mdpl)"
                onClick={() => setSelectedGraphNode('DOWN')}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedGraphNode('DOWN')}
                className={`absolute z-20 cursor-pointer transform transition-all duration-300 hover:scale-105 flex flex-col items-center focus:outline-none focus:ring-2 focus:ring-[#d1bcff] rounded-xl ${
                  selectedGraphNode === 'DOWN' ? 'scale-110' : ''
                }`}
                style={{ bottom: '4%', left: '50%', transform: 'translate(-50%, 0)' }}
              >
                <div
                  className={`w-13 h-13 p-2 rounded-xl bg-[#d1bcff]/15 border border-[#d1bcff] flex flex-col items-center justify-center shadow-[0_0_15px_rgba(209,188,255,0.3)] ${
                    selectedGraphNode === 'DOWN' ? 'ring-2 ring-[#d1bcff]' : ''
                  }`}
                >
                  <span className="font-mono-data text-xs font-bold text-[#d1bcff]">DOWN</span>
                  <span className="font-mono-data text-[8px] text-[#ddcdff]">980m</span>
                </div>
                <span className="font-mono-data text-[9px] text-[#b9cacb] mt-1">Selatan (Sukabumi)</span>
              </div>

              {/* Node LEFT */}
              <div
                role="button"
                tabIndex={0}
                aria-selected={selectedGraphNode === 'LEFT'}
                aria-label="Simpul LEFT (Barat Salak 1120 mdpl)"
                onClick={() => setSelectedGraphNode('LEFT')}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedGraphNode('LEFT')}
                className={`absolute z-20 cursor-pointer transform transition-all duration-300 hover:scale-105 flex flex-col items-center focus:outline-none focus:ring-2 focus:ring-[#7df4ff] rounded-xl ${
                  selectedGraphNode === 'LEFT' ? 'scale-110' : ''
                }`}
                style={{ top: '50%', left: '4%', transform: 'translate(0, -50%)' }}
              >
                <div
                  className={`w-13 h-13 p-2 rounded-xl bg-[#7df4ff]/15 border border-[#7df4ff] flex flex-col items-center justify-center shadow-[0_0_15px_rgba(125,244,255,0.3)] ${
                    selectedGraphNode === 'LEFT' ? 'ring-2 ring-[#7df4ff]' : ''
                  }`}
                >
                  <span className="font-mono-data text-xs font-bold text-[#7df4ff]">LEFT</span>
                  <span className="font-mono-data text-[8px] text-[#dbfcff]">1,120m</span>
                </div>
                <span className="font-mono-data text-[9px] text-[#b9cacb] mt-1">Barat (Salak)</span>
              </div>

              {/* Node RIGHT */}
              <div
                role="button"
                tabIndex={0}
                aria-selected={selectedGraphNode === 'RIGHT'}
                aria-label="Simpul RIGHT (Timur Cianjur 1250 mdpl)"
                onClick={() => setSelectedGraphNode('RIGHT')}
                onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setSelectedGraphNode('RIGHT')}
                className={`absolute z-20 cursor-pointer transform transition-all duration-300 hover:scale-105 flex flex-col items-center focus:outline-none focus:ring-2 focus:ring-[#ffb4ab] rounded-xl ${
                  selectedGraphNode === 'RIGHT' ? 'scale-110' : ''
                }`}
                style={{ top: '50%', right: '4%', transform: 'translate(0, -50%)' }}
              >
                <div
                  className={`w-13 h-13 p-2 rounded-xl bg-[#ffb4ab]/15 border border-[#ffb4ab] flex flex-col items-center justify-center shadow-[0_0_15px_rgba(255,180,171,0.3)] ${
                    selectedGraphNode === 'RIGHT' ? 'ring-2 ring-[#ffb4ab]' : ''
                  }`}
                >
                  <span className="font-mono-data text-xs font-bold text-[#ffb4ab]">RIGHT</span>
                  <span className="font-mono-data text-[8px] text-[#ffdad6]">1,250m</span>
                </div>
                <span className="font-mono-data text-[9px] text-[#b9cacb] mt-1">Timur (Cianjur)</span>
              </div>
            </div>

            <div className="text-[11px] font-mono-data text-[#849495] text-center border-t border-white/5 pt-2">
              Klik pada simpul untuk melihat parameter fitur spasial & kontribusi atensi.
            </div>
          </div>

          {/* Node Details & Matrix (Right 5 Cols) */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-3.5">
            {/* Inspector Card */}
            <div className="bg-[#131b2e] rounded-2xl border border-white/10 p-4 shadow-lg">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <span className="text-[9px] font-mono-data text-[#849495] uppercase tracking-wider block">
                    INSPEKSI SIMPUL ST-GNN
                  </span>
                  <h3 className="font-headline text-base font-bold text-[#00f0ff] flex items-center gap-2">
                    {activeNode.name}
                  </h3>
                </div>
                <span
                  className="px-2.5 py-0.5 rounded-full text-[10px] font-mono-data font-bold border"
                  style={{ backgroundColor: `${activeNode.color}20`, borderColor: activeNode.color, color: activeNode.color }}
                >
                  {activeNode.elevation} mdpl
                </span>
              </div>

              <p className="text-xs text-[#b9cacb] mb-3">{activeNode.locationName}</p>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono-data bg-[#0b1326] p-3 rounded-xl border border-white/5">
                <div>
                  <span className="text-[#849495] text-[10px] block">Koordinat:</span>
                  <span className="text-[#dae2fd] font-bold">{activeNode.lat}°, {activeNode.lng}°</span>
                </div>
                <div>
                  <span className="text-[#849495] text-[10px] block">Peran Graf:</span>
                  <span className="text-[#00f0ff] font-bold">{activeNode.role}</span>
                </div>
                <div>
                  <span className="text-[#849495] text-[10px] block">Curah Hujan (Now):</span>
                  <span className="text-[#fed639] font-bold">
                    {(currentTelemetries[activeNode.id]?.rainIntensityMmH ?? 0).toFixed(1)} mm/h
                  </span>
                </div>
                <div>
                  <span className="text-[#849495] text-[10px] block">Atensi ke MAIN:</span>
                  <span className="text-[#d1bcff] font-bold">
                    {((currentTelemetries[activeNode.id]?.spatialWeightToMain ?? 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Adjacency Matrix */}
            <div className="bg-[#131b2e] rounded-2xl border border-white/10 p-4 shadow-lg">
              <h4 className="font-mono-data text-xs font-bold text-[#b9cacb] mb-2.5 flex items-center gap-1.5 uppercase">
                <Layers className="w-4 h-4 text-[#00f0ff]" /> Matriks Atensi Spasial A<sub>ij</sub>
              </h4>

              <div className="overflow-x-auto">
                <table className="w-full text-center font-mono-data text-[10px]">
                  <thead>
                    <tr className="text-[#849495] border-b border-white/10">
                      <th className="p-1.5 text-left">From \ To</th>
                      {nodeIds.map((id) => (
                        <th key={id} className="p-1.5 text-[#00f0ff] font-bold">{id}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-[#dae2fd]">
                    {nodeIds.map((fromId) => (
                      <tr key={fromId} className={fromId === selectedGraphNode ? 'bg-white/5' : ''}>
                        <td className="p-1.5 text-left text-[#b9cacb] font-bold">{fromId}</td>
                        {nodeIds.map((toId) => {
                          let val = '0.00';
                          if (fromId === toId) val = '1.00';
                          else if (toId === 'MAIN') {
                            val = (currentTelemetries[fromId]?.spatialWeightToMain || 0.25).toFixed(2);
                          }
                          return (
                            <td
                              key={toId}
                              className={`p-1.5 ${
                                toId === 'MAIN' && fromId !== 'MAIN'
                                  ? 'bg-[#00f0ff]/15 text-[#00f0ff] font-bold rounded'
                                  : 'text-[#849495]'
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Elevation Cross-Section Chart */}
            <div className="bg-[#131b2e] rounded-2xl border border-white/10 p-3.5 shadow-lg">
              <div className="font-mono-data text-[10px] text-[#849495] font-bold uppercase mb-1.5 flex items-center gap-1.5">
                <BarChart2 className="w-3.5 h-3.5 text-[#fed639]" /> Elevasi Topografi 5 Node (mdpl)
              </div>
              <div className="flex items-end justify-between h-16 gap-2 pt-2 px-2 border-b border-white/10">
                {Object.values(NODES_LIST).map((n) => {
                  const heightPercent = (n.elevation / 3008) * 100;
                  const isSelected = n.id === selectedGraphNode;
                  return (
                    <div key={n.id} className="flex-1 flex flex-col items-center gap-1">
                      <div
                        className={`w-full rounded-t-md transition-all duration-300 ${
                          isSelected ? 'shadow-[0_0_12px_rgba(0,240,255,0.4)]' : ''
                        }`}
                        style={{
                          height: `${heightPercent}%`,
                          backgroundColor: n.color,
                          opacity: isSelected ? 1 : 0.45,
                        }}
                      />
                      <span className={`font-mono-data text-[8px] ${isSelected ? 'text-[#00f0ff] font-bold' : 'text-[#849495]'}`}>
                        {n.id}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-5 pt-4 border-t border-white/10 flex justify-between items-center">
          <div className="text-xs font-mono-data text-[#849495]">
            AeroCast Gede • Gunung Pangrango (3,008m)
          </div>
          <button
            onClick={() => {
              onSelectNode(selectedGraphNode);
              onClose();
            }}
            className="px-5 py-2.5 rounded-xl bg-[#00f0ff] text-[#002022] font-mono-data text-xs font-bold hover:bg-[#7df4ff] transition-all shadow-[0_0_20px_rgba(0,240,255,0.3)] flex items-center gap-2"
          >
            Fokuskan Simpul di Peta Utama <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

