export type NodeId = 'MAIN' | 'UP' | 'DOWN' | 'LEFT' | 'RIGHT';

export interface ObservationNode {
  id: NodeId;
  name: string;
  locationName: string;
  role: 'Target Utama Prediksi' | 'Konteks Lereng Utara' | 'Konteks Lereng Selatan' | 'Konteks Lereng Barat' | 'Konteks Lereng Timur';
  lat: number;
  lng: number;
  elevation: number; // mdpl
  isTarget: boolean;
  color: string;
}

export interface NodeTelemetry {
  nodeId: NodeId;
  tempCelsius: number;
  dewPointCelsius: number;
  rainIntensityMmH: number; // mm/h
  humidityPercent: number;
  windSpeedKmh: number;
  windDirectionDeg: number;
  pressureHpa: number;
  confidencePercent: number;
  spatialWeightToMain: number; // ST-GNN spatial attention weight (0 to 1)
}

export interface ForecastPoint {
  timeLabel: string;
  timestamp: string;
  p10: number; // Lower bound mm/h
  p25?: number; // 25th percentile mm/h
  p50: number; // Median mm/h
  p75?: number; // 75th percentile mm/h
  p90: number; // Upper bound mm/h
  observed?: number;
}

export type TimelineOffset = 'T-2h' | 'T-1h' | 'Now' | 'T+1h' | 'T+2h';

export type RiskLevel = 'LOW' | 'MODERATE' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'EXTREME';

export interface TimelineFrame {
  offset: TimelineOffset;
  timeWib: string;
  description: string;
  // ponytail: retained while static sample frames exist; remove when API replaces frame fixtures.
  radarIntensityFactor: number;
  telemetries: Partial<Record<NodeId, NodeTelemetry>>;
}

export type DashboardDataSource = 'live' | 'sample';

/** Payload contract for GET VITE_DASHBOARD_LIVE_URL (default: /api/v1/dashboard/live). */
export interface DashboardLiveSnapshot {
  source: DashboardDataSource;
  generatedAt: string;
  frame: TimelineFrame;
  mainForecast: ForecastPoint[];
  riskMetrics?: RiskMetrics;
}

export interface RiskMetrics {
  hypothermiaRiskLevel: RiskLevel;
  hypothermiaElevationMin: number;
  windChillCelsius: number;
  apparentTempCelsius: number;
  orographicAlert: string;
  slopeInstabilityRisk: RiskLevel;
}

export interface ModelMetrics {
  csi: number; // Critical Success Index
  pod: number; // Probability of Detection
  far: number; // False Alarm Ratio
  reliabilityPercent: number;
  rmse: number;
  mae?: number;
  crps?: number;
  brierScore?: number;
  corr?: number;
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
  radarSnapshotUrl?: string;
}

export type ActiveTab = 'map' | 'forecast' | 'risk' | 'retrieval' | 'eval' | 'topology';
