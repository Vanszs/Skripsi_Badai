import React, { useState, useCallback, useEffect, useMemo } from 'react';
import { NodeId, ActiveTab, HistoricalAnalog, DashboardLiveSnapshot } from './types';
import { CURRENT_MODEL_METRICS, TIMELINE_FRAMES, SAMPLE_NOWCAST } from './data/nodesData';
import { fetchDashboardLive } from './services/dashboardLive';
import { SideNav } from './components/SideNav';
import { TopAppBar } from './components/TopAppBar';
import { MapView } from './components/MapView';
import { TimelineScrubber } from './components/TimelineScrubber';
import { ForecastChart } from './components/ForecastChart';
import { RiskAssessmentPanel } from './components/RiskAssessmentPanel';
import { ModelPerformancePanel } from './components/ModelPerformancePanel';
import { RetrievalAnalogsPanel } from './components/RetrievalAnalogsPanel';
import { NodeDetailModal } from './components/NodeDetailModal';
import { TopologyGraphOverlay } from './components/TopologyGraphOverlay';
import { Download, Maximize2, Database, X, CloudRain, MapPinned, ChevronDown, ChevronUp } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('map');
  const [selectedNodeId, setSelectedNodeId] = useState<NodeId>('MAIN');
  const [radarVisible, setRadarVisible] = useState(true);
  const [nodesVisible, setNodesVisible] = useState(true);
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);
  const [isTopologyModalOpen, setIsTopologyModalOpen] = useState(false);
  const [activeNodeDetailId, setActiveNodeDetailId] = useState<NodeId | null>(null);
  const [selectedAnalog, setSelectedAnalog] = useState<HistoricalAnalog | null>(null);
  const [liveSnapshot, setLiveSnapshot] = useState<DashboardLiveSnapshot | null>(null);
  const [isLiveLoading, setIsLiveLoading] = useState(true);
  useEffect(() => {
    let controller: AbortController | undefined;
    const refresh = async () => {
      controller?.abort();
      controller = new AbortController();
      try { setLiveSnapshot(await fetchDashboardLive(controller.signal)); }
      catch (error) { if ((error as DOMException).name !== 'AbortError') setLiveSnapshot(null); }
      finally { setIsLiveLoading(false); }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 60_000);
    return () => { controller?.abort(); clearInterval(timer); };
  }, []);
  const dashboardFrame = liveSnapshot?.frame ?? TIMELINE_FRAMES[0];
  const nowcast = liveSnapshot?.nowcast ?? SAMPLE_NOWCAST;
  const dataStatus = liveSnapshot?.source === 'live'
    ? `Data API diperbarui ${new Date(liveSnapshot.generatedAt).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })} WIB`
    : isLiveLoading ? 'Menghubungkan sumber API…' : 'Contoh data — sumber API belum terhubung';
  const nowcastRows = useMemo(() => ({ precipitation: nowcast.precipitation, wind: nowcast.wind_speed_10m, humidity: nowcast.relative_humidity_2m }), [nowcast]);
  const handleExportData = useCallback(() => {
    const blob = new Blob([JSON.stringify({ app: 'Pangrango Weather Desk', selectedNode: selectedNodeId, nowcast, modelMetrics: CURRENT_MODEL_METRICS }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `Pangrango_ERA5_${selectedNodeId}_T+1h.json`; anchor.click(); URL.revokeObjectURL(url);
  }, [selectedNodeId, nowcast]);
  return <div className="app-shell min-h-[100dvh] h-[100dvh] w-full overflow-hidden flex font-body select-none">
    <SideNav activeTab={activeTab} onSelectTab={setActiveTab} onOpenTopologyModal={() => setIsTopologyModalOpen(true)} />
    <TopAppBar activeTab={activeTab} onSelectTab={setActiveTab} radarVisible={radarVisible} onToggleRadar={() => setRadarVisible((value) => !value)} nodesVisible={nodesVisible} onToggleNodes={() => setNodesVisible((value) => !value)} onSelectNode={setSelectedNodeId} onOpenTopologyModal={() => setIsTopologyModalOpen(true)} />
    <main className="flex-1 relative md:ml-[72px] h-full flex flex-col pt-[64px]">
      {activeTab === 'map' && <div className="flex-1 flex flex-col h-full overflow-hidden relative"><div className="flex-1 relative z-0 p-3 md:p-4 pb-0"><MapView selectedNodeId={selectedNodeId} onSelectNode={setSelectedNodeId} currentFrame={dashboardFrame} radarVisible={radarVisible} nodesVisible={nodesVisible} onOpenNodeDetail={setActiveNodeDetailId} />
        <div className="absolute top-6 left-6 z-20 hidden lg:block pointer-events-none"><div className="w-[260px] border border-[#d8d4c8] bg-white p-4"><div className="flex items-center gap-2 text-[#1f6056]"><MapPinned className="w-4 h-4" /><span className="font-mono-data text-[10px] font-semibold uppercase">Tampilan grid ERA5</span></div><p className="mt-2 font-headline text-base font-bold text-[#202720]">Contoh lima sel ERA5.</p><p className="mt-1 text-xs leading-relaxed text-[#5f685e]">Nowcast T+1 jam untuk MAIN. {dataStatus}</p></div></div>
        <div className={`absolute top-6 right-6 z-20 border px-2.5 py-1.5 text-[10px] font-mono-data font-semibold ${liveSnapshot?.source === 'live' ? 'border-[#1f6056] bg-[#eef6ef] text-[#1f6056]' : 'border-[#a46b13] bg-[#fff8df] text-[#78520f]'}`}>{liveSnapshot?.source === 'live' ? 'DATA API' : 'CONTOH DATA'}</div>
        <div className="absolute bottom-6 left-0 right-0 px-6 z-20 pointer-events-auto flex justify-center"><TimelineScrubber nowWib={dashboardFrame.timeWib} validAtWib={nowcast.precipitation.validAtWib} horizonLabel={nowcast.precipitation.horizon} dataStatus={dataStatus} isLive={liveSnapshot?.source === 'live'} /></div></div>
        <section className={`analysis-drawer bg-white border-t border-[#d8d4c8] z-20 overflow-hidden ${isAnalysisExpanded ? 'absolute inset-x-0 bottom-0 top-0 z-30 p-4 md:p-5 overflow-y-auto' : 'relative h-[154px] sm:h-[168px] flex-shrink-0 p-4 md:p-5 -mt-3'}`} aria-label="Nowcast dan perbandingan sampel"><div className="flex items-center justify-between mb-3.5 border-b border-white/10 pb-2.5"><h2 className="font-headline text-sm md:text-base font-bold text-[#202720] flex items-center gap-2"><CloudRain className="w-4 h-4 text-[#1f6056]" />Nowcast T+1 jam dan perbandingan sampel</h2><div className="flex items-center gap-2"><button onClick={() => setIsAnalysisExpanded((value) => !value)} aria-expanded={isAnalysisExpanded} className="analysis-drawer__toggle">{isAnalysisExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}</button><button onClick={handleExportData} className="p-1.5 rounded-lg bg-[#131b2e] text-[#94a3b8]" title="Unduh JSON"><Download className="w-4 h-4" /></button><button onClick={() => setIsTopologyModalOpen(true)} className="p-1.5 rounded-lg bg-[#131b2e] text-[#94a3b8]" title="Perbesar topologi"><Maximize2 className="w-4 h-4" /></button></div></div><div className="analysis-drawer__content grid grid-cols-1 md:grid-cols-4 gap-3.5"><div className="md:col-span-2 glass-panel p-3.5 flex flex-col"><ForecastChart precipitation={nowcastRows.precipitation} wind={nowcastRows.wind} humidity={nowcastRows.humidity} /></div><div className="md:col-span-1 glass-panel p-3.5 flex flex-col"><RiskAssessmentPanel /></div><div className="md:col-span-1 glass-panel p-3.5 flex flex-col"><ModelPerformancePanel metrics={CURRENT_MODEL_METRICS} onOpenEval={() => setActiveTab('eval')} /></div><div className="md:col-span-4 glass-panel p-3.5"><RetrievalAnalogsPanel onSelectAnalog={setSelectedAnalog} onOpenAll={() => setActiveTab('comparison')} /></div></div>{!isAnalysisExpanded && <div className="analysis-drawer__fade" aria-hidden="true" />}</section></div>}
      {activeTab === 'forecast' && <div className="flex-1 p-6 overflow-y-auto"><div className="glass-panel-cyan p-6"><h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 uppercase tracking-wider">Nowcast T+1 jam: tiga keluaran, 50 sampel</h2><ForecastChart precipitation={nowcastRows.precipitation} wind={nowcastRows.wind} humidity={nowcastRows.humidity} /></div></div>}
      {activeTab === 'comparison' && <div className="flex-1 p-6 overflow-y-auto"><div className="glass-panel-cyan p-6"><h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 flex items-center gap-2 uppercase tracking-wider"><Database className="w-5 h-5 text-[#00f0ff]" />Perbandingan sampel</h2><RetrievalAnalogsPanel onSelectAnalog={setSelectedAnalog} /></div></div>}
      {activeTab === 'eval' && <div className="flex-1 p-6 overflow-y-auto"><div className="glass-panel-cyan p-6"><h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 uppercase tracking-wider">Evaluasi model T+1 jam</h2><ModelPerformancePanel metrics={CURRENT_MODEL_METRICS} onOpenEval={() => undefined} /></div></div>}
    </main>
    <TopologyGraphOverlay isOpen={isTopologyModalOpen} onClose={() => setIsTopologyModalOpen(false)} onSelectNode={(nodeId) => { setSelectedNodeId(nodeId); setActiveTab('map'); }} frame={dashboardFrame} />
    <NodeDetailModal nodeId={activeNodeDetailId} frame={dashboardFrame} onClose={() => setActiveNodeDetailId(null)} />
    {selectedAnalog && <div className="fixed inset-0 z-50 bg-[#070d18]/80 backdrop-blur-sm flex items-center justify-center p-4"><div className="bg-[#070d18] w-full max-w-lg rounded-xl border border-[#00f0ff]/50 p-6 relative"><button onClick={() => setSelectedAnalog(null)} className="absolute top-4 right-4 p-1.5" aria-label="Tutup"><X className="w-4 h-4" /></button><h3 className="font-headline text-base font-bold text-[#00f0ff] mb-1 uppercase">{selectedAnalog.datetime}</h3><span className="font-mono-data text-xs text-[#fed639]">Skor perbandingan sampel: {selectedAnalog.similarityPercent}%</span><p className="text-sm text-[#dae2fd] my-4">{selectedAnalog.outcomeSummary}</p><button onClick={() => setSelectedAnalog(null)} className="px-4 py-2 rounded-lg bg-[#00f0ff] text-[#002022] font-mono-data text-xs font-bold">Tutup</button></div></div>}
  </div>;
}
