import React, { useState, useCallback, useEffect } from 'react';
import { NodeId, ActiveTab, TimelineOffset, HistoricalAnalog, DashboardLiveSnapshot } from './types';
import { CURRENT_RISK_METRICS, CURRENT_MODEL_METRICS, PROBABILISTIC_FORECAST_DATA, TIMELINE_FRAMES } from './data/nodesData';
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
import { Download, Maximize2, Activity, AlertTriangle, Database, X, CloudRain, MapPinned, ChevronDown, ChevronUp } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('map');
  const [selectedNodeId, setSelectedNodeId] = useState<NodeId>('MAIN');
  const [currentOffset, setCurrentOffset] = useState<TimelineOffset>('Now');
  const [isPlaying, setIsPlaying] = useState(false);
  const [radarVisible, setRadarVisible] = useState(true);
  const [nodesVisible, setNodesVisible] = useState(true);
  const [liveSnapshot, setLiveSnapshot] = useState<DashboardLiveSnapshot | null>(null);
  const [isLiveLoading, setIsLiveLoading] = useState(true);
  const [isAnalysisExpanded, setIsAnalysisExpanded] = useState(false);

  // Modals state
  const [isTopologyModalOpen, setIsTopologyModalOpen] = useState(false);
  const [activeNodeDetailId, setActiveNodeDetailId] = useState<NodeId | null>(null);
  const [selectedAnalog, setSelectedAnalog] = useState<HistoricalAnalog | null>(null);

  // Stable Callbacks
  const handleSelectTab = useCallback((tab: ActiveTab) => setActiveTab(tab), []);
  const handleSelectNode = useCallback((nodeId: NodeId) => setSelectedNodeId(nodeId), []);
  const handleOffsetChange = useCallback((offset: TimelineOffset) => setCurrentOffset(offset), []);
  const handleTogglePlay = useCallback(() => setIsPlaying((prev) => !prev), []);
  const handleToggleRadar = useCallback(() => setRadarVisible((prev) => !prev), []);
  const handleToggleNodes = useCallback(() => setNodesVisible((prev) => !prev), []);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined;
    let controller: AbortController | undefined;
    const refresh = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        setLiveSnapshot(await fetchDashboardLive(controller.signal));
      } catch (error) {
        if ((error as DOMException).name !== 'AbortError') {
          setLiveSnapshot(null);
        }
      } finally {
        setIsLiveLoading(false);
      }
    };
    void refresh();
    timer = setInterval(() => void refresh(), 60_000);
    return () => { controller?.abort(); if (timer) clearInterval(timer); };
  }, []);

  const sampleFrame = TIMELINE_FRAMES.find((frame) => frame.offset === currentOffset) ?? TIMELINE_FRAMES[2];
  const dashboardFrame = liveSnapshot?.frame ?? sampleFrame;
  const mainForecast = liveSnapshot?.mainForecast ?? PROBABILISTIC_FORECAST_DATA.MAIN;
  const dashboardRisk = liveSnapshot?.riskMetrics ?? CURRENT_RISK_METRICS;
  const dataStatus = liveSnapshot?.source === 'live'
    ? `Diperbarui ${new Date(liveSnapshot.generatedAt).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })} WIB`
    : isLiveLoading
      ? 'Memuat sumber cuaca…'
      : 'Contoh data - sumber cuaca belum terhubung';

  const handleOpenTopologyModal = useCallback(() => setIsTopologyModalOpen(true), []);
  const handleCloseTopologyModal = useCallback(() => setIsTopologyModalOpen(false), []);
  const handleOpenNodeDetail = useCallback((id: NodeId) => setActiveNodeDetailId(id), []);
  const handleCloseNodeDetail = useCallback(() => setActiveNodeDetailId(null), []);
  const handleSelectAnalog = useCallback((analog: HistoricalAnalog | null) => setSelectedAnalog(analog), []);
  const handleCloseAnalog = useCallback(() => setSelectedAnalog(null), []);

  const handleOpenRiskTab = useCallback(() => setActiveTab('risk'), []);
  const handleOpenEvalTab = useCallback(() => setActiveTab('eval'), []);
  const handleOpenRetrievalTab = useCallback(() => setActiveTab('retrieval'), []);

  const handleSelectNodeAndFocusMap = useCallback((nodeId: NodeId) => {
    setSelectedNodeId(nodeId);
    setActiveTab('map');
  }, []);

  const handleExportData = useCallback(() => {
    const report = {
      app: 'AeroCast Gede',
      timestamp: new Date().toISOString(),
      selectedNode: selectedNodeId,
      currentOffset,
      riskMetrics: CURRENT_RISK_METRICS,
      modelMetrics: CURRENT_MODEL_METRICS,
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AeroCast_Gede_Report_${selectedNodeId}_${currentOffset}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedNodeId, currentOffset]);

  return (
    <div className="app-shell min-h-[100dvh] h-[100dvh] w-full overflow-hidden flex font-body select-none">
      {/* Side Navigation Rail */}
      <SideNav
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        onOpenTopologyModal={handleOpenTopologyModal}
      />

      {/* Top Application Header */}
      <TopAppBar
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        radarVisible={radarVisible}
        onToggleRadar={handleToggleRadar}
        nodesVisible={nodesVisible}
        onToggleNodes={handleToggleNodes}
        onSelectNode={handleSelectNode}
        onOpenTopologyModal={handleOpenTopologyModal}
      />

      {/* Main Screen Layout */}
      <main className="flex-1 relative md:ml-[72px] h-full flex flex-col pt-[64px]">
        {/* VIEW TAB 1: Main Viewport (Map + Bento Grid) */}
        {activeTab === 'map' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden relative">
            {/* Upper Section: Interactive Map */}
            <div className="flex-1 relative z-0 p-3 md:p-4 pb-0">
              <MapView
                selectedNodeId={selectedNodeId}
                onSelectNode={handleSelectNode}
                currentFrame={dashboardFrame}
                radarVisible={radarVisible}
                nodesVisible={nodesVisible}
                onOpenNodeDetail={handleOpenNodeDetail}
              />

              <div className="absolute top-6 left-6 z-20 hidden lg:block pointer-events-none">
                <div className="w-[260px] border border-[#d8d4c8] bg-white p-4">
                  <div className="flex items-center gap-2 text-[#1f6056]">
                    <MapPinned className="w-4 h-4" />
                    <span className="font-mono-data text-[10px] font-semibold uppercase">Tampilan grid ERA5</span>
                  </div>
                  <p className="mt-2 font-headline text-base font-bold text-[#202720]">Kondisi cuaca di lima area sekitar puncak.</p>
                  <p className="mt-1 text-xs leading-relaxed text-[#5f685e]">Prediksi utama berlaku untuk area MAIN. {dataStatus}</p>
                </div>
              </div>

              <div className={`absolute top-6 right-6 z-20 border px-2.5 py-1.5 text-[10px] font-mono-data font-semibold ${
                liveSnapshot?.source === 'live'
                  ? 'border-[#1f6056] bg-[#eef6ef] text-[#1f6056]'
                  : 'border-[#a46b13] bg-[#fff8df] text-[#78520f]'
              }`}>
                {liveSnapshot?.source === 'live' ? dataStatus : 'CONTOH DATA'}
              </div>

              {/* Timeline Scrubber Floated over bottom of Map */}
              <div className="absolute bottom-6 left-0 right-0 px-6 z-20 pointer-events-auto flex justify-center">
                <TimelineScrubber
                  currentOffset={currentOffset}
                  onOffsetChange={handleOffsetChange}
                  isPlaying={isPlaying}
                  onTogglePlay={handleTogglePlay}
                  dataStatus={dataStatus}
                  isLive={liveSnapshot?.source === 'live'}
                />
              </div>
            </div>

            {/* Lower Section: Technical Bento Grid Panel */}
            <section className={`analysis-drawer bg-white border-t border-[#d8d4c8] z-20 overflow-hidden ${
              isAnalysisExpanded
                ? 'absolute inset-x-0 bottom-0 top-0 z-30 p-4 md:p-5 overflow-y-auto'
                : 'relative h-[154px] sm:h-[168px] flex-shrink-0 p-4 md:p-5 -mt-3'
            }`} aria-label="Analisis hujan dan pembanding historis">
              {/* Header inside Bento Grid */}
              <div className="flex items-center justify-between mb-3.5 border-b border-white/10 pb-2.5">
                <h2 className="font-headline text-sm md:text-base font-bold text-[#202720] flex items-center gap-2">
                  <CloudRain className="w-4 h-4 text-[#1f6056]" />
                  Analisis hujan dan pembanding historis
                </h2>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsAnalysisExpanded((expanded) => !expanded)}
                    aria-expanded={isAnalysisExpanded}
                    aria-label={isAnalysisExpanded ? 'Ringkas analisis' : 'Perluas analisis'}
                    className="analysis-drawer__toggle"
                    title={isAnalysisExpanded ? 'Ringkas analisis' : 'Perluas analisis'}
                  >
                    {isAnalysisExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
                  </button>
                  <button
                    onClick={handleExportData}
                    className="p-1.5 rounded-lg bg-[#131b2e] text-[#94a3b8] hover:text-[#00f0ff] hover:bg-[#1a253d] transition-all border border-white/12 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)]"
                    title="Unduh Laporan JSON"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={handleOpenTopologyModal}
                    className="p-1.5 rounded-lg bg-[#131b2e] text-[#94a3b8] hover:text-[#00f0ff] hover:bg-[#1a253d] transition-all border border-white/12 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.06)]"
                    title="Perbesar Topologi Graf"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Bento Grid Layout */}
              <div className="analysis-drawer__content grid grid-cols-1 md:grid-cols-4 gap-3.5">
                {/* 1. Probabilistic Forecast Chart (2 Cols) */}
                <div className="md:col-span-2 glass-panel p-3.5 flex flex-col">
                  <ForecastChart selectedNodeId="MAIN" forecastData={mainForecast} />
                </div>

                {/* 2. Risk Assessment (1 Col) */}
                <div className="md:col-span-1 glass-panel p-3.5 flex flex-col">
                  <RiskAssessmentPanel
                    riskMetrics={dashboardRisk}
                    onOpenDetails={handleOpenRiskTab}
                  />
                </div>

                {/* 3. Model Performance Metrics (1 Col) */}
                <div className="md:col-span-1 glass-panel p-3.5 flex flex-col">
                  <ModelPerformancePanel
                    metrics={CURRENT_MODEL_METRICS}
                    onOpenEval={handleOpenEvalTab}
                  />
                </div>

                {/* 4. Retrieval Historical Analogs (Full Width Row) */}
                <div className="md:col-span-4 glass-panel p-3.5">
                  <RetrievalAnalogsPanel
                    onSelectAnalog={handleSelectAnalog}
                    onOpenAll={handleOpenRetrievalTab}
                  />
                </div>
              </div>
              {!isAnalysisExpanded && <div className="analysis-drawer__fade" aria-hidden="true" />}
            </section>
          </div>
        )}

        {/* VIEW TAB 2: Forecast Standalone View */}
        {activeTab === 'forecast' && (
          <div className="flex-1 p-6 overflow-y-auto space-y-6">
            <div className="glass-panel-cyan p-6">
              <h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 uppercase tracking-wider">
                Perkiraan hujan di area utama
              </h2>
              <div className="h-[380px]">
                <ForecastChart selectedNodeId="MAIN" forecastData={mainForecast} />
              </div>
            </div>
          </div>
        )}

        {/* VIEW TAB 3: Risk Assessment Standalone View */}
        {activeTab === 'risk' && (
          <div className="flex-1 p-6 overflow-y-auto space-y-6">
            <div className="bg-[#060e20] p-6 rounded-xl border border-[#ffb4ab]/30 shadow-[inset_0_1px_0_0_rgba(255,180,171,0.15)]">
              <h2 className="font-headline text-lg font-bold text-[#ffb4ab] mb-4 flex items-center gap-2 uppercase tracking-wider">
                <AlertTriangle className="w-5 h-5 text-[#ffb4ab]" /> Hal yang perlu diwaspadai saat mendaki
              </h2>
              <RiskAssessmentPanel riskMetrics={dashboardRisk} onOpenDetails={handleOpenRiskTab} />
            </div>
          </div>
        )}

        {/* VIEW TAB 4: Retrieval Database View */}
        {activeTab === 'retrieval' && (
          <div className="flex-1 p-6 overflow-y-auto space-y-6">
            <div className="glass-panel-cyan p-6">
              <h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 flex items-center gap-2 uppercase tracking-wider">
                <Database className="w-5 h-5 text-[#00f0ff]" /> Kejadian cuaca serupa sebelumnya
              </h2>
              <RetrievalAnalogsPanel onSelectAnalog={handleSelectAnalog} />
            </div>
          </div>
        )}

        {/* VIEW TAB 5: Model Evaluation View */}
        {activeTab === 'eval' && (
          <div className="flex-1 p-6 overflow-y-auto space-y-6">
            <div className="glass-panel-cyan p-6">
              <h2 className="font-headline text-lg font-bold text-[#dbfcff] mb-4 flex items-center gap-2 uppercase tracking-wider">
                <Activity className="w-5 h-5 text-[#00f0ff]" /> Penjelasan ketepatan perkiraan
              </h2>
              <ModelPerformancePanel metrics={CURRENT_MODEL_METRICS} onOpenEval={handleOpenEvalTab} />
            </div>
          </div>
        )}
      </main>

      {/* ST-GNN 5-Node Topology Modal */}
      <TopologyGraphOverlay
        isOpen={isTopologyModalOpen}
        onClose={handleCloseTopologyModal}
        onSelectNode={handleSelectNodeAndFocusMap}
        frame={dashboardFrame}
      />

      {/* Node Detail Modal */}
      <NodeDetailModal
        nodeId={activeNodeDetailId}
        frame={dashboardFrame}
        onClose={handleCloseNodeDetail}
      />

      {/* Analog Detail Dialog */}
      {selectedAnalog && (
        <div className="fixed inset-0 z-50 bg-[#070d18]/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#070d18] w-full max-w-lg rounded-xl border border-[#00f0ff]/50 p-6 shadow-2xl relative shadow-[inset_0_1px_0_0_rgba(0,240,255,0.2)]">
            <button
              onClick={handleCloseAnalog}
              className="absolute top-4 right-4 p-1.5 rounded-lg bg-[#131b2e] text-[#849495] hover:text-white border border-white/15 transition-all"
            >
              <X className="w-4 h-4" />
            </button>
            <h3 className="font-headline text-base font-bold text-[#00f0ff] mb-1 uppercase">
              {selectedAnalog.datetime}
            </h3>
            <span className="font-mono-data text-xs text-[#fed639] bg-[#fed639]/10 px-2.5 py-0.5 rounded-full border border-[#fed639]/30 inline-block mb-3">
              Kemiripan Geometri Radar: {selectedAnalog.similarityPercent}%
            </span>
            <p className="text-sm text-[#dae2fd] mb-4 font-body">
              {selectedAnalog.outcomeSummary}
            </p>
            <div className="grid grid-cols-2 gap-3 font-mono-data text-xs bg-[#131b2e] p-3 rounded-lg border border-white/15 mb-4">
              <div>
                <span className="text-[#849495] block text-[10px] uppercase font-semibold">Pola Badai:</span>
                <span className="text-[#dbfcff] font-bold">{selectedAnalog.patternType}</span>
              </div>
              <div>
                <span className="text-[#849495] block text-[10px] uppercase font-semibold">Akumulasi Hujan:</span>
                <span className="text-[#00f0ff] font-bold">{selectedAnalog.accumulatedRainMm} mm</span>
              </div>
            </div>
            <div className="flex justify-end">
              <button
                onClick={handleCloseAnalog}
                className="px-4 py-2 rounded-lg bg-[#00f0ff] text-[#002022] font-mono-data text-xs font-bold hover:bg-[#7df4ff] uppercase tracking-wider transition-all shadow-[0_0_12px_rgba(0,240,255,0.3)]"
              >
                Tutup Analogi Historis
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

