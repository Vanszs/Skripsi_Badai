import React from 'react';
import { ModelMetrics } from '../types';

interface ModelPerformancePanelProps { metrics: ModelMetrics; onOpenEval: () => void; }
const metric = (label: string, value: number, suffix = '') => <div className="bg-[#0b1120] rounded-lg p-2 border border-white/10"><div className="font-mono-data text-[9px] text-slate-400 font-semibold truncate">{label}</div><div className="font-mono-data text-base text-[#00f0ff] font-extrabold mt-0.5">{value.toFixed(3)}{suffix}</div></div>;
export const ModelPerformancePanel: React.FC<ModelPerformancePanelProps> = ({ metrics, onOpenEval }) => <div role="region" aria-label="Metrik evaluasi model" className="w-full h-full flex flex-col justify-between space-y-3">
  <div className="flex justify-between items-center pb-1.5 border-b border-white/10"><h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase tracking-wider">Metrik uji presipitasi</h3><button onClick={onOpenEval} className="text-[10px] font-mono-data text-[#00f0ff] hover:underline">Lihat penjelasan</button></div>
  <p className="font-mono-data text-[9px] text-[#849495]">`core/results/result_test/full_model/metrics.json` • ambang 2 mm</p>
  <div className="grid grid-cols-3 gap-2">{metric('RMSE (mm/jam)', metrics.rmse)}{metric('MAE (mm/jam)', metrics.mae)}{metric('Korelasi', metrics.corr)}</div>
  <div className="grid grid-cols-2 gap-2">{metric('CRPS', metrics.crps)}{metric('Brier 2 mm', metrics.brierScore)}</div>
  <div className="grid grid-cols-3 gap-2">{metric('CSI 2 mm', metrics.csi)}{metric('POD 2 mm', metrics.pod)}{metric('FAR 2 mm', metrics.far)}</div>
  <p className="text-[10px] text-[#b9cacb]">POD: proporsi kejadian ≥2 mm yang terdeteksi. FAR: proporsi deteksi yang salah.</p>
</div>;
