import React, { useMemo } from 'react';
import { CloudRain, Wind, Droplets, Activity } from 'lucide-react';
import { NowcastDistribution } from '../types';
import { SAMPLE_NOWCAST } from '../data/nodesData';

interface ForecastChartProps {
  precipitation?: NowcastDistribution;
  wind?: NowcastDistribution;
  humidity?: NowcastDistribution;
}

/**
 * One-step nowcast (T+1h) panel.
 * Truth-of-source: the diffusion model emits 50 samples for 3 variables at a single horizon.
 * We render each variable as a percentile band + observed-now marker, NOT a multi-step line.
 */
const ForecastChartComponent: React.FC<ForecastChartProps> = ({
  precipitation,
  wind,
  humidity,
}) => {
  const rows = useMemo(
    () => [
      {
        key: 'precip',
        label: 'Curah Hujan',
        icon: CloudRain,
        accent: '#1f6056',
        data: precipitation ?? SAMPLE_NOWCAST.precipitation,
        observedLabel: 'Obs. sekarang',
      },
      {
        key: 'wind',
        label: 'Angin 10m',
        icon: Wind,
        accent: '#fed639',
        data: wind ?? SAMPLE_NOWCAST.wind_speed_10m,
      },
      {
        key: 'humidity',
        label: 'Kelembapan 2m',
        icon: Droplets,
        accent: '#7df4ff',
        data: humidity ?? SAMPLE_NOWCAST.relative_humidity_2m,
        observedLabel: 'Obs. sekarang',
      },
    ],
    [precipitation, wind, humidity],
  );

  const horizonLabel = rows[0]?.data.horizon ?? 'T+1h';
  const validAt = rows[0]?.data.validAtWib ?? '';

  return (
    <div className="w-full h-full flex flex-col gap-2">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase tracking-wider">
            Nowcast {horizonLabel} (50 Sampel Difusi)
          </h3>
          <span className="text-[10px] font-mono-data text-[#1f6056] bg-[#1f6056]/10 px-2 py-0.5 rounded border border-[#1f6056]/30">
            Berlaku {validAt}
          </span>
        </div>
        <span className="text-[10px] font-mono-data font-bold text-[#1f6056] bg-[#dbeae3] px-2 py-1 border border-[#1f6056]">
          AREA MAIN
        </span>
      </div>

      {/* Rows */}
      <div className="flex flex-col gap-2 flex-1 justify-center">
        {rows.map((row) => (
          <NowcastRow key={row.key} {...row} />
        ))}
      </div>

      {/* Footer legend */}
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 text-[10px] font-mono-data text-[#849495] pt-2 border-t border-white/5">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="flex items-center gap-1 whitespace-nowrap">
            <span className="w-2.5 h-2.5 bg-[#1f6056]/15 border border-[#1f6056]/40 inline-block" />
            P10–P90
          </span>
          <span className="flex items-center gap-1 whitespace-nowrap">
            <span className="w-2.5 h-2.5 bg-[#1f6056]/40 border border-[#1f6056]/70 inline-block" />
            P25–P75
          </span>
          <span className="flex items-center gap-1 whitespace-nowrap">
            <span className="w-3 h-0.5 bg-[#1f6056] inline-block" />
            Median (P50)
          </span>
          <span className="flex items-center gap-1 whitespace-nowrap">
            <Activity className="w-3 h-3 text-[#fed639]" />
            Observasi
          </span>
        </div>
      </div>
    </div>
  );
};

interface NowcastRowProps {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  accent: string;
  data: NowcastDistribution;
  observedLabel?: string;
}

const NowcastRow: React.FC<NowcastRowProps> = ({ label, icon: Icon, accent, data, observedLabel }) => {
  const min = Math.min(data.p10, data.observedNow ?? data.p10);
  const max = Math.max(data.p90, data.observedNow ?? data.p90);
  const span = Math.max(0.01, max - min);
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - min) / span) * 100));

  return (
    <div className="bg-[#0b1326]/40 border border-white/10 rounded-lg p-2.5">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <Icon className="w-3.5 h-3.5" style={{ color: accent }} />
          <span className="font-mono-data text-[11px] font-bold text-[#dae2fd] uppercase tracking-wider">
            {label}
          </span>
          <span className="text-[9px] font-mono-data text-[#849495]">{data.unit}</span>
        </div>
        <div className="flex items-center gap-3 text-[10px] font-mono-data">
          <span className="text-[#849495]">
            P50 <span className="font-bold text-[#dae2fd]">{data.p50}</span>
          </span>
          <span className="text-[#849495]">
            μ <span className="font-bold text-[#dae2fd]">{data.mean}</span>
          </span>
        </div>
      </div>

      <div className="relative h-7 w-full">
        {/* Track */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2 bg-[#070d18] rounded-full border border-white/10" />

        {/* P10-P90 band */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-2 rounded-full border"
          style={{
            left: `${pct(data.p10)}%`,
            width: `${pct(data.p90) - pct(data.p10)}%`,
            background: `${accent}25`,
            borderColor: `${accent}50`,
          }}
        />

        {/* P25-P75 band */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-2 rounded-full border"
          style={{
            left: `${pct(data.p25)}%`,
            width: `${pct(data.p75) - pct(data.p25)}%`,
            background: `${accent}55`,
            borderColor: `${accent}90`,
          }}
        />

        {/* Median line */}
        <div
          className="absolute top-0 bottom-0 w-0.5"
          style={{ left: `${pct(data.p50)}%`, background: accent }}
        />

        {/* Observed marker (if available) */}
        {data.observedNow !== undefined && (
          <>
            <div
              className="absolute top-0 bottom-0 w-px bg-[#fed639]"
              style={{ left: `${pct(data.observedNow)}%` }}
            />
            <div
              className="absolute -top-1 w-2 h-2 rotate-45 bg-[#fed639] border border-[#070d18]"
              style={{ left: `calc(${pct(data.observedNow)}% - 4px)` }}
            />
          </>
        )}
      </div>

      {/* Numeric axis ticks */}
      <div className="flex justify-between text-[9px] font-mono-data text-[#849495] mt-1">
        <span>{min.toFixed(1)}</span>
        <span>
          {observedLabel && data.observedNow !== undefined
            ? `${observedLabel}: ${data.observedNow.toFixed(1)}`
            : `${data.samples.length} sampel`}
        </span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  );
};

export const ForecastChart = React.memo(ForecastChartComponent);
