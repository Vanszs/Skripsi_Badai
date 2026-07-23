import React, { useMemo } from 'react';
import { HistoricalAnalog } from '../types';
import { HISTORICAL_ANALOGS } from '../data/historicalAnalogs';
import { Database, ArrowUpRight, CloudRain, Wind, Thermometer, Droplets, Sparkles, Layers } from 'lucide-react';

interface RetrievalAnalogsPanelProps {
  onSelectAnalog?: (analog: HistoricalAnalog) => void;
  onOpenAll?: () => void;
  limitK?: number; // Default k=3
}

const RetrievalAnalogsPanelComponent: React.FC<RetrievalAnalogsPanelProps> = ({
  onSelectAnalog,
  onOpenAll,
  limitK = 3,
}) => {
  // Take k=3 nearest FAISS historical analogs
  const topAnalogs = useMemo(() => HISTORICAL_ANALOGS.slice(0, limitK), [limitK]);

  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-[#00f0ff]/10 border border-[#00f0ff]/30 text-[#00f0ff]">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-mono-data text-xs font-bold text-[#dae2fd] uppercase tracking-wider flex items-center gap-1.5">
              FAISS RETRIEVAL ANALOGS (k={limitK} NEAREST HISTORICAL STATES)
              <span className="text-[10px] font-normal px-1.5 py-0.5 rounded bg-[#00f0ff]/10 text-[#00f0ff] border border-[#00f0ff]/30 lowercase font-mono">
                IndexIVFFlat
              </span>
            </h3>
            <p className="text-[10px] font-mono-data text-[#849495] mt-0.5">
              Top-{limitK} vektor kondisi atmosfer paling mirip dari database ERA5 (2005–2021)
            </p>
          </div>
        </div>

        <button
          onClick={onOpenAll}
          aria-label={`Lihat Semua Database (${HISTORICAL_ANALOGS.length} Analogs)`}
          className="text-[11px] font-mono-data text-[#00f0ff] hover:text-[#7df4ff] hover:underline flex items-center gap-1 bg-[#171f33]/60 px-2.5 py-1 rounded-lg border border-white/10 transition-colors focus:outline-none focus:ring-1 focus:ring-[#00f0ff]"
        >
          <Layers className="w-3 h-3" />
          Lihat Semua Database ({HISTORICAL_ANALOGS.length})
          <ArrowUpRight className="w-3 h-3" />
        </button>
      </div>

      {/* Grid Cards Container - k=3 Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {topAnalogs.map((analog, idx) => {
          const rank = analog.rank || idx + 1;

          // Rank styling tokens
          const rankStyles = {
            1: {
              badgeBg: 'bg-[#00f0ff]/15 text-[#00f0ff] border-[#00f0ff]/40',
              accentColor: '#00f0ff',
              barGradient: 'from-[#00f0ff] to-[#00b8c4]',
              borderHover: 'hover:border-[#00f0ff]/60',
              shadowGlow: 'shadow-[0_0_15px_rgba(0,240,255,0.1)]',
            },
            2: {
              badgeBg: 'bg-[#38bdf8]/15 text-[#38bdf8] border-[#38bdf8]/40',
              accentColor: '#38bdf8',
              barGradient: 'from-[#38bdf8] to-[#0284c7]',
              borderHover: 'hover:border-[#38bdf8]/60',
              shadowGlow: 'shadow-[0_0_15px_rgba(56,189,248,0.1)]',
            },
            3: {
              badgeBg: 'bg-[#a855f7]/15 text-[#c084fc] border-[#a855f7]/40',
              accentColor: '#c084fc',
              barGradient: 'from-[#c084fc] to-[#7e22ce]',
              borderHover: 'hover:border-[#a855f7]/60',
              shadowGlow: 'shadow-[0_0_15px_rgba(168,85,247,0.1)]',
            },
          }[rank] || {
            badgeBg: 'bg-[#849495]/15 text-[#b9cacb] border-[#849495]/40',
            accentColor: '#b9cacb',
            barGradient: 'from-[#b9cacb] to-[#64748b]',
            borderHover: 'hover:border-white/30',
            shadowGlow: '',
          };

          return (
            <div
              key={analog.id}
              role="button"
              tabIndex={0}
              aria-label={`Analogi FAISS #${rank} ${analog.datetime} Kemiripan ${analog.similarityPercent.toFixed(1)} persen`}
              onClick={() => onSelectAnalog?.(analog)}
              onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && onSelectAnalog?.(analog)}
              className={`bg-[#131b2e]/80 hover:bg-[#172138] rounded-xl border border-white/10 p-3 flex flex-col justify-between cursor-pointer transition-all duration-200 hover:scale-[1.015] focus:outline-none focus:ring-2 focus:ring-[#00f0ff] ${rankStyles.borderHover} ${rankStyles.shadowGlow} group relative overflow-hidden`}
            >
              {/* Subtle Ambient Background Gradient */}
              <div
                className="absolute -top-12 -right-12 w-24 h-24 rounded-full blur-2xl opacity-20 pointer-events-none transition-opacity group-hover:opacity-35"
                style={{ backgroundColor: rankStyles.accentColor }}
              />

              {/* Top Row: Rank Badge + Datetime */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`font-mono-data text-[10px] font-extrabold px-2 py-0.5 rounded-md border ${rankStyles.badgeBg} flex items-center gap-1`}
                    >
                      <Sparkles className="w-2.5 h-2.5" />
                      #{rank} FAISS ANALOG
                    </span>
                  </div>

                  <span className="font-mono-data text-[11px] font-bold text-[#dae2fd]">
                    {analog.datetime}
                  </span>
                </div>

                {/* Pattern Classification Badge */}
                <div className="mb-2.5">
                  <span className="inline-block font-mono-data text-[10px] text-[#b9cacb] bg-[#0b1326] px-2 py-0.5 rounded border border-white/5 truncate max-w-full">
                    {analog.patternType}
                  </span>
                </div>

                {/* Similarity Score & L2 Metric */}
                <div className="bg-[#0b1326]/70 rounded-lg p-2 border border-white/5 mb-3">
                  <div className="flex items-baseline justify-between mb-1">
                    <div className="flex items-baseline gap-1">
                      <span className="font-mono-data text-xs text-[#849495] uppercase">Kemiripan:</span>
                      <span
                        className="font-mono-data text-base font-black tracking-tight"
                        style={{ color: rankStyles.accentColor }}
                      >
                        {analog.similarityPercent.toFixed(1)}%
                      </span>
                    </div>

                    {analog.distanceL2 !== undefined && (
                      <span className="font-mono-data text-[10px] text-[#849495] bg-[#171f33] px-1.5 py-0.5 rounded border border-white/5">
                        L2 Dist: <strong className="text-[#dbfcff]">{analog.distanceL2.toFixed(3)}</strong>
                      </span>
                    )}
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full h-1.5 bg-[#171f33] rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${rankStyles.barGradient} transition-all duration-500`}
                      style={{ width: `${Math.min(100, Math.max(0, analog.similarityPercent))}%` }}
                    />
                  </div>
                </div>

                {/* Weather Features Grid (4 Crisp Metrics) */}
                <div className="grid grid-cols-2 gap-1.5 font-mono-data text-[11px] mb-2.5">
                  {/* Rain Rate */}
                  <div className="bg-[#0b1326]/50 p-1.5 rounded-md border border-white/5 flex items-center gap-1.5">
                    <CloudRain className="w-3.5 h-3.5 text-[#00f0ff] shrink-0" />
                    <div className="min-w-0">
                      <span className="text-[9px] text-[#849495] block leading-none">Curah Hujan</span>
                      <span className="font-bold text-[#00f0ff] truncate block">
                        {analog.rainRateMmH !== undefined ? `${analog.rainRateMmH} mm/jam` : `${analog.accumulatedRainMm} mm`}
                      </span>
                    </div>
                  </div>

                  {/* Wind Speed */}
                  <div className="bg-[#0b1326]/50 p-1.5 rounded-md border border-white/5 flex items-center gap-1.5">
                    <Wind className="w-3.5 h-3.5 text-[#fed639] shrink-0" />
                    <div className="min-w-0">
                      <span className="text-[9px] text-[#849495] block leading-none">Kec. Angin</span>
                      <span className="font-bold text-[#fed639] truncate block">
                        {analog.peakWindKmh} km/jam
                      </span>
                    </div>
                  </div>

                  {/* Temperature */}
                  <div className="bg-[#0b1326]/50 p-1.5 rounded-md border border-white/5 flex items-center gap-1.5">
                    <Thermometer className="w-3.5 h-3.5 text-[#ffb4ab] shrink-0" />
                    <div className="min-w-0">
                      <span className="text-[9px] text-[#849495] block leading-none">Suhu Puncak</span>
                      <span className="font-bold text-[#ffb4ab] truncate block">
                        {analog.tempCelsius !== undefined ? `${analog.tempCelsius}°C` : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Humidity */}
                  <div className="bg-[#0b1326]/50 p-1.5 rounded-md border border-white/5 flex items-center gap-1.5">
                    <Droplets className="w-3.5 h-3.5 text-[#7df4ff] shrink-0" />
                    <div className="min-w-0">
                      <span className="text-[9px] text-[#849495] block leading-none">Kelembapan</span>
                      <span className="font-bold text-[#7df4ff] truncate block">
                        {analog.humidityPercent !== undefined ? `${analog.humidityPercent}%` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bottom Outcome Summary */}
              <div className="pt-2 border-t border-white/5 flex items-center justify-between">
                <p className="text-[10px] text-[#b9cacb] line-clamp-1 flex-1 pr-2">
                  {analog.outcomeSummary}
                </p>
                <span
                  className="font-mono-data text-[10px] font-bold underline transition-colors shrink-0 flex items-center gap-0.5"
                  style={{ color: rankStyles.accentColor }}
                >
                  Detail <ArrowUpRight className="w-2.5 h-2.5" />
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const RetrievalAnalogsPanel = React.memo(RetrievalAnalogsPanelComponent);
