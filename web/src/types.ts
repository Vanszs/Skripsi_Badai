export type NodeId = 'MAIN' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';

export interface ObservationNode {
  id: NodeId;
  name: string;
  locationName: string;
  role: 'Target Utama Prediksi' | 'Konteks Utara' | 'Konteks Selatan' | 'Konteks Barat' | 'Konteks Timur';
  lat: number;
  lng: number;
  elevation: number;
  isTarget: boolean;
  color: string;
}

export interface NodeTelemetry {
  nodeId: NodeId;
  tempCelsius: number;
  dewPointCelsius: number;
  rainIntensityMmH: number;
  humidityPercent: number;
  windSpeedKmh: number;
  windDirectionDeg: number;
  pressureHpa: number;
}

export interface NowcastDistribution {
  horizon: 'T+1h';
  validAtWib: string;
  samples: number[];
  mean: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
  observedNow?: number;
  unit: string;
  variable: 'precipitation' | 'wind_speed_10m' | 'relative_humidity_2m';
}

export interface TimelineFrame {
  offset: 'Now';
  timeWib: string;
  description: string;
  telemetries: Partial<Record<NodeId, NodeTelemetry>>;
}

export type DashboardDataSource = 'live' | 'sample';
export type NowcastVariableSet = {
  precipitation: NowcastDistribution;
  wind_speed_10m: NowcastDistribution;
  relative_humidity_2m: NowcastDistribution;
};
export interface DashboardLiveSnapshot {
  source: DashboardDataSource;
  generatedAt: string;
  frame: TimelineFrame;
  nowcast: NowcastVariableSet;
}

export interface ModelMetrics {
  csi: number;
  pod: number;
  far: number;
  rmse: number;
  mae: number;
  crps: number;
  brierScore: number;
  corr: number;
}

export interface HistoricalAnalog {
  id: string;
  rank?: number;
  datetime: string;
  patternType: string;
  similarityPercent: number;
  distanceL2?: number;
  accumulatedRainMm: number;
  rainRateMmH?: number;
  peakWindKmh: number;
  tempCelsius?: number;
  humidityPercent?: number;
  pressureHpa?: number;
  outcomeSummary: string;
}

export type ActiveTab = 'map' | 'forecast' | 'comparison' | 'eval' | 'topology';
