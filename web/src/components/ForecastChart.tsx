import React, { useMemo, useId } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { ForecastPoint, NodeId } from '../types';
import { PROBABILISTIC_FORECAST_DATA, NODES_LIST } from '../data/nodesData';

interface ForecastChartProps {
  selectedNodeId: NodeId;
  forecastData?: ForecastPoint[];
}

// Custom tooltip with zero-rain handling & portal z-index safety
const CustomTooltip = React.memo(({ active, payload, label, nodeName }: any) => {
  if (!active || !payload || !payload.length) return null;
  const data = payload.find((p: any) => p?.payload)?.payload;
  if (!data) return null;

  const isZeroRain = data.p90 === 0 && data.p10 === 0;

  return (
    <div className="glass-panel p-2.5 rounded-xl border border-[#1f6056]/40 shadow-2xl font-mono-data text-xs space-y-1 min-w-[210px] pointer-events-none bg-[#0b1326]/95 backdrop-blur-md">
      <div className="flex justify-between items-center text-[#1f6056] font-bold border-b border-white/10 pb-1 mb-1">
        <span>Pukul {label} WIB</span>
        <span className="text-[10px] text-[#849495]">{nodeName}</span>
      </div>
      <div className="flex justify-between gap-4 text-[#ffb4ab]">
        <span className="text-[#849495]">Kemungkinan tertinggi:</span>
        <span className="font-bold">{data.p90.toFixed(1)} mm/jam</span>
      </div>
      <div className="flex justify-between gap-4 text-[#7df4ff]">
        <span className="text-[#849495]">Kemungkinan atas:</span>
        <span>{data.p75.toFixed(1)} mm/jam</span>
      </div>
      <div className="flex justify-between gap-4 text-[#1f6056] font-bold">
        <span>Perkiraan utama:</span>
        <span>{data.p50.toFixed(1)} mm/jam</span>
      </div>
      <div className="flex justify-between gap-4 text-[#7df4ff]">
        <span className="text-[#849495]">Kemungkinan bawah:</span>
        <span>{data.p25.toFixed(1)} mm/jam</span>
      </div>
      <div className="flex justify-between gap-4 text-[#006970]">
        <span className="text-[#849495]">Kemungkinan terendah:</span>
        <span>{data.p10.toFixed(1)} mm/jam</span>
      </div>
      {data.observed !== undefined && (
        <div className="flex justify-between gap-4 text-[#fed639] border-t border-white/10 pt-1 mt-1 font-bold">
          <span>Hujan yang terukur:</span>
          <span>{data.observed.toFixed(1)} mm/jam</span>
        </div>
      )}
      <div className="flex justify-between gap-4 text-[10px] text-[#849495] border-t border-white/5 pt-1">
        <span>Selisih kemungkinan:</span>
        <span className="text-white font-semibold">
          {isZeroRain ? '0.0 mm/jam (Kering)' : `±${(data.spread / 2).toFixed(1)} mm/jam`}
        </span>
      </div>
    </div>
  );
});

const ForecastChartComponent: React.FC<ForecastChartProps> = ({
  selectedNodeId,
  forecastData,
}) => {
  const baseId = useId().replace(/:/g, '');
  const band1090Id = `band1090-${baseId}`;
  const band2575Id = `band2575-${baseId}`;

  const rawData = forecastData ?? PROBABILISTIC_FORECAST_DATA[selectedNodeId] ?? PROBABILISTIC_FORECAST_DATA.MAIN;
  const currentNode = NODES_LIST[selectedNodeId];

  // Process data with strict monotonic quantile validation & non-negative clamping
  const chartData = useMemo(() => {
    return rawData.map((d) => {
      const p10 = Math.max(0, d.p10);
      const p50 = Math.max(p10, d.p50);
      const p90 = Math.max(p50, d.p90);

      // Quantile P25 and P75 with strict monotonic bound enforcement
      const rawP25 = d.p25 ?? Number((p10 + 0.25 * (p90 - p10)).toFixed(1));
      const rawP75 = d.p75 ?? Number((p10 + 0.75 * (p90 - p10)).toFixed(1));

      const safeP25 = Math.max(p10, Math.min(p50, rawP25));
      const safeP75 = Math.max(p50, Math.min(p90, rawP75));

      const observed = d.observed !== undefined ? Math.max(0, d.observed) : undefined;
      const rawSpread = p90 - p10;
      const spread = Number(rawSpread.toFixed(1));

      return {
        ...d,
        p10: Number(p10.toFixed(1)),
        p25: Number(safeP25.toFixed(1)),
        p50: Number(p50.toFixed(1)),
        p75: Number(safeP75.toFixed(1)),
        p90: Number(p90.toFixed(1)),
        observed: observed !== undefined ? Number(observed.toFixed(1)) : undefined,
        range10_90: [Number(p10.toFixed(1)), Number(p90.toFixed(1))],
        range25_75: [Number(safeP25.toFixed(1)), Number(safeP75.toFixed(1))],
        spread,
      };
    });
  }, [rawData]);

  return (
    <div className="w-full h-full flex flex-col justify-between gap-1">
      {/* Chart Header & Node Selector */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-1">
        <div className="flex items-center gap-2 flex-wrap">
          <h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase tracking-wider">
            PERKIRAAN HUJAN DI AREA UTAMA
          </h3>
          <span className="text-[10px] font-mono-data text-[#1f6056] bg-[#1f6056]/10 px-2 py-0.5 rounded border border-[#1f6056]/30">
            Rentang kemungkinan hujan
          </span>
        </div>

        <span className="text-[10px] font-mono-data font-bold text-[#1f6056] bg-[#dbeae3] px-2 py-1 border border-[#1f6056]">AREA MAIN</span>
      </div>

      {/* Main Recharts Container with responsive height & safe margins */}
      <div className="w-full h-[210px] sm:h-[230px] relative" role="img" aria-label="Grafik perkiraan hujan area utama">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 15, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id={band1090Id} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1f6056" stopOpacity={0.18} />
                <stop offset="95%" stopColor="#1f6056" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id={band2575Id} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#1f6056" stopOpacity={0.40} />
                <stop offset="95%" stopColor="#1f6056" stopOpacity={0.08} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#d8d4c8" />
            <XAxis
              dataKey="timeLabel"
              tick={{ fill: '#5f685e', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: '#d8d4c8' }}
              tickLine={{ stroke: '#d8d4c8' }}
              minTickGap={15}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#5f685e', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
              axisLine={{ stroke: '#d8d4c8' }}
              tickLine={{ stroke: '#d8d4c8' }}
              tickFormatter={(val: number) => `${val}`}
              domain={[0, 'auto']}
              allowDataOverflow={false}
              width={35}
            />
            <Tooltip
              content={<CustomTooltip nodeName={currentNode.name} />}
              wrapperStyle={{ zIndex: 1000, pointerEvents: 'none', outline: 'none' }}
              allowEscapeViewBox={{ x: true, y: true }}
              offset={12}
              isAnimationActive={false}
              cursor={{ stroke: '#1f6056', strokeWidth: 1, strokeDasharray: '3 3' }}
            />

            {/* P10-P90 Outer Ensemble Spread Band */}
            <Area
              type="monotone"
              dataKey="range10_90"
              stroke="#1f6056"
              strokeWidth={1}
              strokeDasharray="2 2"
              fill={`url(#${band1090Id})`}
              isAnimationActive={false}
            />

            {/* P25-P75 Inner IQR Band */}
            <Area
              type="monotone"
              dataKey="range25_75"
              stroke="rgba(31, 96, 86, 0.38)"
              strokeWidth={1}
              fill={`url(#${band2575Id})`}
              isAnimationActive={false}
            />

            {/* P50 Ensemble Median Line */}
            <Line
              type="monotone"
              dataKey="p50"
              stroke="#1f6056"
              strokeWidth={2.5}
              dot={{ r: 3, fill: '#1f6056', stroke: '#ffffff', strokeWidth: 1.5 }}
              activeDot={{ r: 6, fill: '#ffffff', stroke: '#1f6056' }}
              isAnimationActive={false}
            />

            {/* Observed Actual Rain Points */}
            <Line
              type="monotone"
              dataKey="observed"
              stroke="#b77817"
              strokeWidth={2}
              strokeDasharray="4 4"
              dot={{ r: 3.5, fill: '#b77817', stroke: '#ffffff', strokeWidth: 1 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Chart Footer Legend - Fully Responsive & Truncation Safe */}
      <div className="flex flex-wrap items-center justify-between text-[10px] font-mono-data text-[#849495] pt-2 border-t border-white/5 gap-y-1.5 gap-x-3">
        <div className="flex items-center gap-2.5 flex-wrap">
          <span className="flex items-center gap-1 text-[#1f6056] font-bold whitespace-nowrap">
            <span className="w-2.5 h-0.5 bg-[#1f6056] inline-block" /> P50 (Median)
          </span>
          <span className="flex items-center gap-1 text-[#7df4ff] whitespace-nowrap">
            <span className="w-2.5 h-2.5 bg-[#1f6056]/30 border border-[#1f6056]/50 inline-block rounded-xs" /> Band P25–P75
          </span>
          <span className="flex items-center gap-1 text-[#1f6056]/70 whitespace-nowrap">
            <span className="w-2.5 h-2.5 bg-[#1f6056]/10 border border-[#1f6056]/30 inline-block rounded-xs" /> Band P10–P90
          </span>
          <span className="flex items-center gap-1 text-[#fed639] whitespace-nowrap">
            <span className="w-2 h-2 rounded-full bg-[#fed639] inline-block" /> Observasi
          </span>
        </div>
        <span className="text-[#1f6056] font-bold truncate ml-auto max-w-[180px] sm:max-w-none text-right">
          {currentNode.locationName}
        </span>
      </div>
    </div>
  );
};

export const ForecastChart = React.memo(ForecastChartComponent);
