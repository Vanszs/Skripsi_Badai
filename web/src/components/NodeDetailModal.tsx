import React, { useEffect, useRef } from 'react';
import { NodeId, TimelineFrame } from '../types';
import { NODES_LIST } from '../data/nodesData';
import { X, MapPin, Thermometer, Droplets, Wind, Gauge, Shield, ArrowDownRight, Compass } from 'lucide-react';

interface NodeDetailModalProps {
  nodeId: NodeId | null;
  frame: TimelineFrame;
  onClose: () => void;
}

const NodeDetailModalComponent: React.FC<NodeDetailModalProps> = ({
  nodeId,
  frame,
  onClose,
}) => {
  const modalRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!nodeId) return;

    // Store triggering element for focus restoration
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Auto focus container or close button
    if (modalRef.current) {
      modalRef.current.focus();
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }

      // Focus trapping
      if (e.key === 'Tab' && modalRef.current) {
        const focusables = modalRef.current.querySelectorAll<HTMLElement>(
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
  }, [nodeId, onClose]);

  if (!nodeId || !NODES_LIST[nodeId]) return null;

  const node = NODES_LIST[nodeId];

  const telemetry = frame.telemetries[nodeId];
  if (!telemetry) return null;

  return (
    <div className="fixed inset-0 z-50 bg-[#060b14]/85 backdrop-blur-xl flex items-center justify-center p-2 sm:p-4 overflow-y-auto animate-in fade-in duration-200">
      <div 
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="node-detail-modal-title"
        className="glass-panel w-full max-w-xl max-h-[90vh] [@media(max-height:500px)]:max-h-[85vh] overflow-y-auto rounded-3xl border border-[#00f0ff]/35 p-4 sm:p-6 shadow-[0_0_40px_rgba(0,240,255,0.15)] relative bg-[#0b1326]/95 focus:outline-none"
      >
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 w-11 h-11 rounded-full bg-[#171f33] text-[#849495] hover:text-white hover:bg-white/15 transition-all flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-[#00f0ff]"
          aria-label="Tutup Detail Simpul"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-start gap-3.5 mb-5 pb-4 border-b border-white/10">
          <div
            className="w-12 h-12 rounded-2xl flex items-center justify-center font-mono-data font-black text-base shadow-lg border"
            style={{ backgroundColor: `${node.color}20`, borderColor: node.color, color: node.color }}
          >
            {node.id}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 id="node-detail-modal-title" className="font-headline text-lg font-bold text-[#dbfcff] tracking-tight">
                Simpul Observasi: {node.name}
              </h2>
              {node.isTarget && (
                <span className="px-2.5 py-0.5 rounded-full bg-[#00f0ff]/20 text-[#00f0ff] font-mono-data text-[10px] font-bold border border-[#00f0ff]/40 shadow-[0_0_10px_rgba(0,240,255,0.3)]">
                  Target Utama
                </span>
              )}
            </div>
            <p className="text-xs text-[#849495] font-body flex items-center gap-1.5 mt-1">
              <MapPin className="w-3.5 h-3.5 text-[#00f0ff]" />
              {node.locationName} <span className="text-[#00f0ff] font-mono-data">({node.role})</span>
            </p>
          </div>
        </div>

        {/* Spatial Coordinates & Elevation Badge */}
        <div className="grid grid-cols-3 gap-2 mb-5 p-3.5 bg-[#131b2e] rounded-2xl border border-white/5 font-mono-data text-xs shadow-inner">
          <div>
            <span className="text-[#849495] block text-[10px] uppercase">Lintang (LAT)</span>
            <span className="text-[#dae2fd] font-bold">{node.lat.toFixed(4)}° S</span>
          </div>
          <div>
            <span className="text-[#849495] block text-[10px] uppercase">Bujur (LNG)</span>
            <span className="text-[#dae2fd] font-bold">{node.lng.toFixed(4)}° E</span>
          </div>
          <div>
            <span className="text-[#849495] block text-[10px] uppercase">Elevasi Orografis</span>
            <span className="text-[#00f0ff] font-bold">{node.elevation} mdpl</span>
          </div>
        </div>

        {/* Telemetry Metrics */}
        <div className="flex justify-between items-center mb-3">
          <h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-[#00f0ff]" /> Telemetri Real-Time ({frame.timeWib})
          </h3>
          <span className="text-[10px] font-mono-data text-[#849495]">Offset: {frame.offset}</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-6">
          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-white/5 shadow-sm hover:border-[#00f0ff]/30 transition-all">
            <div className="flex items-center gap-1.5 text-[#00f0ff] font-mono-data text-[10px] uppercase font-bold">
              <Droplets className="w-3.5 h-3.5" /> Curah Hujan
            </div>
            <div className="font-mono-data text-xl font-black text-[#dae2fd] mt-1.5">
              {telemetry.rainIntensityMmH.toFixed(1)} <span className="text-xs text-[#849495] font-normal">mm/h</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-white/5 shadow-sm hover:border-[#ffb4ab]/30 transition-all">
            <div className="flex items-center gap-1.5 text-[#ffb4ab] font-mono-data text-[10px] uppercase font-bold">
              <Thermometer className="w-3.5 h-3.5" /> Suhu
            </div>
            <div className="font-mono-data text-xl font-black text-[#dae2fd] mt-1.5">
              {telemetry.tempCelsius.toFixed(1)} <span className="text-xs text-[#849495] font-normal">°C</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-white/5 shadow-sm hover:border-[#fed639]/30 transition-all">
            <div className="flex items-center gap-1.5 text-[#fed639] font-mono-data text-[10px] uppercase font-bold">
              <Wind className="w-3.5 h-3.5" /> Kecepatan Angin
            </div>
            <div className="font-mono-data text-xl font-black text-[#dae2fd] mt-1.5">
              {telemetry.windSpeedKmh} <span className="text-xs text-[#849495] font-normal">km/h</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-white/5 shadow-sm hover:border-[#d1bcff]/30 transition-all">
            <div className="flex items-center gap-1.5 text-[#d1bcff] font-mono-data text-[10px] uppercase font-bold">
              <Gauge className="w-3.5 h-3.5" /> Tekanan Barometrik
            </div>
            <div className="font-mono-data text-xl font-black text-[#dae2fd] mt-1.5">
              {telemetry.pressureHpa} <span className="text-xs text-[#849495] font-normal">hPa</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-white/5 shadow-sm hover:border-[#7df4ff]/30 transition-all">
            <div className="flex items-center gap-1.5 text-[#7df4ff] font-mono-data text-[10px] uppercase font-bold">
              <Shield className="w-3.5 h-3.5" /> Kelembapan
            </div>
            <div className="font-mono-data text-xl font-black text-[#dae2fd] mt-1.5">
              {telemetry.humidityPercent} <span className="text-xs text-[#849495] font-normal">%</span>
            </div>
          </div>

          <div className="p-3.5 bg-[#171f33]/90 rounded-2xl border border-[#00f0ff]/30 shadow-sm">
            <div className="flex items-center gap-1.5 text-[#00f0ff] font-mono-data text-[10px] uppercase font-bold">
              <ArrowDownRight className="w-3.5 h-3.5" /> Atensi GNN
            </div>
            <div className="font-mono-data text-xl font-black text-[#00f0ff] mt-1.5">
              {(telemetry.spatialWeightToMain * 100).toFixed(0)}% <span className="text-[9px] text-[#849495] font-normal">ke MAIN</span>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={onClose}
            className="px-5 py-2.5 rounded-xl bg-[#00f0ff] text-[#002022] font-mono-data text-xs font-bold hover:bg-[#7df4ff] transition-all shadow-[0_0_15px_rgba(0,240,255,0.3)]"
          >
            Tutup Panel Node
          </button>
        </div>
      </div>
    </div>
  );
};

export const NodeDetailModal = React.memo(NodeDetailModalComponent);

