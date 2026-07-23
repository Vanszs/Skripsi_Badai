import React from 'react';
import { ModelMetrics } from '../types';

interface ModelPerformancePanelProps {
  metrics: ModelMetrics;
  onOpenEval: () => void;
}

const ModelPerformancePanelComponent: React.FC<ModelPerformancePanelProps> = ({
  metrics,
  onOpenEval,
}) => {
  return (
    <div role="region" aria-label="Panel akurasi perkiraan cuaca" className="w-full h-full flex flex-col justify-between space-y-3">
      {/* Header Bar */}
      <div className="flex justify-between items-center pb-1.5 border-b border-white/10">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-[#00f0ff] animate-pulse"></span>
          <h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase tracking-wider">
            SEBERAPA AKURAT PERKIRAANNYA
          </h3>
        </div>
        <button
          onClick={onOpenEval}
          aria-label="Buka Halaman Evaluasi Metrik Model Lengkap"
          className="text-[10px] font-mono-data text-[#00f0ff] hover:text-cyan-300 hover:underline cursor-pointer transition-colors focus:outline-none focus:ring-1 focus:ring-[#00f0ff]"
        >
          Lihat penjelasan →
        </button>
      </div>

      {/* Group 1: Metrik Deterministik */}
      <div className="space-y-1.5">
        <div className="font-mono-data text-[9px] text-[#849495] font-bold uppercase tracking-wider flex items-center gap-1">
          <span className="text-[#00f0ff]">■</span> Ketepatan angka hujan
        </div>
        <div className="grid grid-cols-3 gap-2">
          {/* RMSE */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#38bdf8] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              RMSE
            </div>
            <div className="font-mono-data text-base text-[#38bdf8] font-extrabold mt-0.5 flex items-baseline justify-between">
              <span>{metrics.rmse}</span>
              <span className="text-[8px] text-slate-500 font-normal">mm/h</span>
            </div>
          </div>

          {/* MAE */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#34d399] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              MAE
            </div>
            <div className="font-mono-data text-base text-[#34d399] font-extrabold mt-0.5 flex items-baseline justify-between">
              <span>{metrics.mae ?? 0.85}</span>
              <span className="text-[8px] text-slate-500 font-normal">mm/h</span>
            </div>
          </div>

          {/* Korelasi */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#818cf8] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Kesesuaian pola
            </div>
            <div className="font-mono-data text-base text-[#818cf8] font-extrabold mt-0.5">
              {metrics.corr ?? 0.455}
            </div>
          </div>
        </div>
      </div>

      {/* Group 2: Metrik Probabilistik & Deteksi */}
      <div className="space-y-1.5 flex-1 flex flex-col justify-between">
        <div className="font-mono-data text-[9px] text-[#849495] font-bold uppercase tracking-wider flex items-center gap-1">
          <span className="text-[#c084fc]">■</span> Keandalan saat cuaca berubah
        </div>

        <div className="grid grid-cols-3 gap-2">
          {/* CRPS */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#c084fc] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              CRPS
            </div>
            <div className="font-mono-data text-base text-[#c084fc] font-extrabold mt-0.5">
              {metrics.crps ?? 0.830}
            </div>
          </div>

          {/* Brier Score */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#f472b6] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Brier Score
            </div>
            <div className="font-mono-data text-base text-[#f472b6] font-extrabold mt-0.5">
              {metrics.brierScore ?? 0.020}
            </div>
          </div>

          {/* Reliability */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#7df4ff] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Keandalan
            </div>
            <div className="font-mono-data text-base text-[#7df4ff] font-extrabold mt-0.5">
              {metrics.reliabilityPercent}%
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {/* CSI */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#00f0ff] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Hujan terdeteksi
            </div>
            <div className="font-mono-data text-base text-[#00f0ff] font-extrabold mt-0.5">
              {metrics.csi}
            </div>
          </div>

          {/* POD */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#fed639] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Hujan terlewat
            </div>
            <div className="font-mono-data text-base text-[#fed639] font-extrabold mt-0.5">
              {metrics.pod}
            </div>
          </div>

          {/* FAR */}
          <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10 border-l-[3px] border-l-[#ffb4ab] flex flex-col justify-between hover:border-white/20 transition-all shadow-sm">
            <div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">
              Peringatan keliru
            </div>
            <div className="font-mono-data text-base text-[#ffb4ab] font-extrabold mt-0.5">
              {metrics.far}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export const ModelPerformancePanel = React.memo(ModelPerformancePanelComponent);

