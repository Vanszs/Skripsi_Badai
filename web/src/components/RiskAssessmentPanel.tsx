import React from 'react';
import { Info } from 'lucide-react';

export const RiskAssessmentPanel: React.FC = () => <div role="region" aria-label="Batasan demonstrasi" className="w-full h-full flex flex-col justify-center gap-2">
  <div className="flex items-center gap-2 text-[#b9cacb]"><Info className="w-4 h-4 text-[#1f6056]" /><h3 className="font-mono-data text-xs font-bold uppercase">Batasan demonstrasi</h3></div>
  <p className="text-xs leading-relaxed text-[#dae2fd]">Tampilan ini hanya menunjukkan contoh keluaran cuaca. Tidak memodelkan risiko hipotermia atau longsor, serta bukan saran keselamatan.</p>
</div>;
