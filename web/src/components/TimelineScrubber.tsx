import React from 'react';
import { Clock, RefreshCw } from 'lucide-react';

interface TimelineScrubberProps {
  nowWib: string;
  validAtWib: string;
  horizonLabel: string;
  dataStatus: string;
  isLive: boolean;
}

/**
 * One-step nowcast info bar.
 * The model produces a single T+1h forecast, so there is nothing to scrub — only info.
 */
const TimelineScrubberComponent: React.FC<TimelineScrubberProps> = ({
  nowWib,
  validAtWib,
  horizonLabel,
  dataStatus,
  isLive,
}) => {
  return (
    <div
      role="region"
      aria-label="Informasi horizon nowcast"
      className="glass-panel w-full max-w-4xl rounded-2xl p-3.5 flex flex-wrap items-center gap-4 border border-slate-700/60 bg-slate-900/90 backdrop-blur-xl shadow-[0_10px_35px_rgba(0,0,0,0.6)]"
    >
      <div className="flex items-center gap-2 flex-shrink-0">
        <RefreshCw className="w-4 h-4 text-cyan-400" />
        <span className="font-mono-data text-[10px] font-bold text-[#849495] uppercase tracking-wider">
          One-Step Nowcast
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs font-mono-data">
        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#070d18] border border-white/10 rounded-lg">
          <Clock className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-amber-300 font-bold">Obs. {nowWib}</span>
        </div>
        <span className="text-[#5f685e]">→</span>
        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-[#070d18] border border-[#1f6056]/50 rounded-lg">
          <Clock className="w-3.5 h-3.5 text-[#1f6056]" />
          <span className="text-[#1f6056] font-bold">{horizonLabel}</span>
          <span className="text-[#dae2fd] font-bold">{validAtWib}</span>
        </div>
      </div>

      <div className="hidden lg:block max-w-[200px] text-right ml-auto">
        <div className="text-[11px] font-mono-data text-slate-400 leading-tight truncate">
          {dataStatus}
        </div>
      </div>

      <div
        className={`flex items-center gap-1.5 font-mono-data text-xs font-bold px-3 py-1.5 rounded-xl tracking-wider ${
          isLive
            ? 'text-[#1f6056] bg-[#dbeae3] border border-[#1f6056]'
            : 'text-[#78520f] bg-[#fff8df] border border-[#a46b13]'
        }`}
        aria-label={isLive ? 'Data terbaru' : 'Data contoh'}
      >
        <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-[#1f6056]' : 'bg-[#a46b13]'}`} />
        {isLive ? 'TERBARU' : 'CONTOH'}
      </div>
    </div>
  );
};

export const TimelineScrubber = React.memo(TimelineScrubberComponent);
