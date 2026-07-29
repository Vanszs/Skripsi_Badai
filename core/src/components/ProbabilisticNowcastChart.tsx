import React, { useState, useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

export interface HourlyEnsembleSample {
  step: string; // e.g. 't+1', 't+2', ... 't+6'
  samples: number[]; // 50 Monte Carlo ensemble samples (mm/hr)
  crps?: number;
  brierScore10?: number;
}

export interface ProbabilisticNowcastProps {
  data: HourlyEnsembleSample[];
  selectedStepIndex?: number;
}

const calculatePercentile = (arr: number[], q: number): number => {
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
};

export const ProbabilisticNowcastChart: React.FC<ProbabilisticNowcastProps> = ({
  data,
  selectedStepIndex = 0
}) => {
  const [threshold, setThreshold] = useState<number>(10);
  const [activeStepIdx, setActiveStepIdx] = useState<number>(selectedStepIndex);

  // Transform raw ensemble samples into percentile summary for Recharts
  const chartData = useMemo(() => {
    return data.map((item) => {
      const p10 = calculatePercentile(item.samples, 0.1);
      const p25 = calculatePercentile(item.samples, 0.25);
      const p50 = calculatePercentile(item.samples, 0.5);
      const p75 = calculatePercentile(item.samples, 0.75);
      const p90 = calculatePercentile(item.samples, 0.9);

      // Calculate exceedance probability directly from 50 samples
      const countExceed = item.samples.filter((s) => s > threshold).length;
      const exceedanceProb = (countExceed / item.samples.length) * 100;

      return {
        step: item.step,
        // Band ranges (stacked format for Recharts Area)
        p10_p90_range: [p10, p90],
        p25_p75_range: [p25, p75],
        p10,
        p25,
        p50,
        p75,
        p90,
        exceedanceProb,
        crps: item.crps ?? 0,
        brierScore10: item.brierScore10 ?? 0
      };
    });
  }, [data, threshold]);

  // Generate histogram distribution for the selected forecast step
  const histogramData = useMemo(() => {
    const activeSamples = data[activeStepIdx]?.samples || [];
    if (activeSamples.length === 0) return [];

    const min = Math.min(...activeSamples, 0);
    const max = Math.max(...activeSamples, 20);
    const binCount = 10;
    const stepSize = Math.max((max - min) / binCount, 1);

    const bins = Array.from({ length: binCount }, (_, i) => {
      const start = min + i * stepSize;
      const end = start + stepSize;
      return {
        range: `${start.toFixed(1)}-${end.toFixed(1)}`,
        count: 0,
        isHeavyRain: end > 10
      };
    });

    activeSamples.forEach((val) => {
      const binIdx = Math.min(
        Math.floor((val - min) / stepSize),
        binCount - 1
      );
      if (bins[binIdx]) {
        bins[binIdx].count += 1;
      }
    });

    return bins;
  }, [data, activeStepIdx]);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '16px', background: '#fff', borderRadius: '8px' }}>
      <h3 style={{ margin: '0 0 12px 0' }}>Probabilistic Nowcast Ensemble & Risk Curve</h3>

      {/* Threshold Selector */}
      <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', alignItems: 'center' }}>
        <span>Exceedance Threshold:</span>
        {[2, 5, 10].map((t) => (
          <button
            key={t}
            onClick={() => setThreshold(t)}
            style={{
              padding: '6px 12px',
              borderRadius: '4px',
              border: '1px solid #ccc',
              background: threshold === t ? '#2563eb' : '#f3f4f6',
              color: threshold === t ? '#fff' : '#000',
              cursor: 'pointer'
            }}
          >
            &gt; {t} mm/hr
          </button>
        ))}
      </div>

      {/* Main Nowcast Time-Series Chart */}
      <div style={{ width: '100%', height: 320 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="step" />
            <YAxis label={{ value: 'Precipitation (mm/hr)', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            {/* Uncertainty Band P10-P90 */}
            <Area
              type="monotone"
              dataKey="p10_p90_range"
              stroke="none"
              fill="#93c5fd"
              fillOpacity={0.4}
              name="Uncertainty P10-P90"
            />
            {/* Uncertainty Band P25-P75 */}
            <Area
              type="monotone"
              dataKey="p25_p75_range"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.5}
              name="Uncertainty P25-P75"
            />
            {/* Median Line P50 */}
            <Line
              type="monotone"
              dataKey="p50"
              stroke="#1d4ed8"
              strokeWidth={2.5}
              dot={{ r: 4 }}
              name="Median (P50)"
            />
            <ReferenceLine y={threshold} stroke="#dc2626" strokeDasharray="4 4" label={`Threshold ${threshold} mm/hr`} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Ensemble Histogram & Step Selector */}
      <div style={{ marginTop: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <h4>50-Sample Ensemble Distribution at Step</h4>
          <select
            value={activeStepIdx}
            onChange={(e) => setActiveStepIdx(Number(e.target.value))}
            style={{ padding: '4px 8px', borderRadius: '4px' }}
          >
            {data.map((item, idx) => (
              <option key={item.step} value={idx}>
                {item.step} (Exceedance Prob: {chartData[idx]?.exceedanceProb.toFixed(1)}%)
              </option>
            ))}
          </select>
        </div>

        <div style={{ width: '100%', height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis label={{ value: 'Sample Count', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="count" fill="#60a5fa" name="Ensemble Samples" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
