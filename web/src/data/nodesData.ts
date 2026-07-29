import { ObservationNode, NodeId, TimelineFrame, ModelMetrics, NowcastDistribution } from '../types';

export const NODES_LIST: Record<NodeId, ObservationNode> = {
  MAIN: { id: 'MAIN', name: 'MAIN', locationName: 'Sel ERA5 target', role: 'Target Utama Prediksi', lat: -6.75, lng: 107.00, elevation: 1529, isTarget: true, color: '#1f6056' },
  UP: { id: 'UP', name: 'UP', locationName: 'Sel ERA5 utara', role: 'Konteks Utara', lat: -6.50, lng: 107.00, elevation: 162, isTarget: false, color: '#a46b13' },
  DOWN: { id: 'DOWN', name: 'DOWN', locationName: 'Sel ERA5 selatan', role: 'Konteks Selatan', lat: -7.00, lng: 107.00, elevation: 0, isTarget: false, color: '#9c3f1d' },
  LEFT: { id: 'LEFT', name: 'LEFT', locationName: 'Sel ERA5 barat', role: 'Konteks Barat', lat: -6.75, lng: 106.75, elevation: 823, isTarget: false, color: '#2c7569' },
  RIGHT: { id: 'RIGHT', name: 'RIGHT', locationName: 'Sel ERA5 timur', role: 'Konteks Timur', lat: -6.75, lng: 107.25, elevation: 288, isTarget: false, color: '#5f7965' },
};

export const TIMELINE_FRAMES: TimelineFrame[] = [{
  offset: 'Now',
  timeWib: '14:30 WIB',
  description: 'Contoh masukan lima sel ERA5 untuk nowcast T+1 jam.',
  telemetries: {
    MAIN: { nodeId: 'MAIN', tempCelsius: 8.2, dewPointCelsius: 8.0, rainIntensityMmH: 14.8, humidityPercent: 98, windSpeedKmh: 34, windDirectionDeg: 220, pressureHpa: 705 },
    UP: { nodeId: 'UP', tempCelsius: 17.0, dewPointCelsius: 16.2, rainIntensityMmH: 6.4, humidityPercent: 92, windSpeedKmh: 22, windDirectionDeg: 220, pressureHpa: 855 },
    DOWN: { nodeId: 'DOWN', tempCelsius: 20.0, dewPointCelsius: 19.1, rainIntensityMmH: 16.5, humidityPercent: 95, windSpeedKmh: 28, windDirectionDeg: 190, pressureHpa: 898 },
    LEFT: { nodeId: 'LEFT', tempCelsius: 18.5, dewPointCelsius: 18.0, rainIntensityMmH: 18.2, humidityPercent: 96, windSpeedKmh: 32, windDirectionDeg: 250, pressureHpa: 882 },
    RIGHT: { nodeId: 'RIGHT', tempCelsius: 18.1, dewPointCelsius: 16.9, rainIntensityMmH: 5.0, humidityPercent: 88, windSpeedKmh: 18, windDirectionDeg: 205, pressureHpa: 872 },
  },
}];

function makeDistribution(variable: NowcastDistribution['variable'], unit: string, samples: number[], observedNow?: number): NowcastDistribution {
  const sorted = [...samples].sort((a, b) => a - b);
  const pick = (q: number) => sorted[Math.floor(q * (sorted.length - 1))];
  const mean = sorted.reduce((sum, value) => sum + value, 0) / sorted.length;
  return { horizon: 'T+1h', validAtWib: '15:30 WIB', samples: sorted, mean: Number(mean.toFixed(2)), p10: Number(pick(.1).toFixed(2)), p25: Number(pick(.25).toFixed(2)), p50: Number(pick(.5).toFixed(2)), p75: Number(pick(.75).toFixed(2)), p90: Number(pick(.9).toFixed(2)), observedNow, unit, variable };
}

function samples(center: number, spread: number): number[] {
  return Array.from({ length: 50 }, (_, index) => Math.max(0, center + spread * (((index + .5) / 50) * 2 - 1) + Math.sin(index * 1.7) * spread * .05));
}

export const SAMPLE_NOWCAST = {
  precipitation: makeDistribution('precipitation', 'mm/jam', samples(12.8, 5.4), 14.8),
  wind_speed_10m: makeDistribution('wind_speed_10m', 'm/s', samples(8.4, 2.1)),
  relative_humidity_2m: makeDistribution('relative_humidity_2m', '%', samples(94.0, 3.5), 98),
};

export const CURRENT_MODEL_METRICS: ModelMetrics = {
  rmse: 0.7919542193412781,
  mae: 0.35300207138061523,
  corr: 0.6390322075552407,
  crps: 0.2517963140936897,
  brierScore: 0.04941500296156928,
  pod: 0.7398373983739838,
  far: 0.7064516129032258,
  csi: 0.26608187134502925,
};
