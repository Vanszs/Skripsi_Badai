import { ObservationNode, NodeId, TimelineFrame, RiskMetrics, ModelMetrics, ForecastPoint } from '../types';

export const NODES_LIST: Record<NodeId, ObservationNode> = {
  MAIN: {
    id: 'MAIN',
    name: 'MAIN',
    locationName: 'Sel ERA5 target',
    role: 'Target Utama Prediksi',
    lat: -6.75,
    lng: 107.00,
    elevation: 1529,
    isTarget: true,
    color: '#1f6056',
  },
  UP: {
    id: 'UP',
    name: 'UP',
    locationName: 'Sel ERA5 utara',
    role: 'Konteks Lereng Utara',
    lat: -6.50,
    lng: 107.00,
    elevation: 162,
    isTarget: false,
    color: '#a46b13',
  },
  DOWN: {
    id: 'DOWN',
    name: 'DOWN',
    locationName: 'Sel ERA5 selatan',
    role: 'Konteks Lereng Selatan',
    lat: -7.00,
    lng: 107.00,
    elevation: 0,
    isTarget: false,
    color: '#9c3f1d',
  },
  LEFT: {
    id: 'LEFT',
    name: 'LEFT',
    locationName: 'Sel ERA5 barat',
    role: 'Konteks Lereng Barat',
    lat: -6.75,
    lng: 106.75,
    elevation: 823,
    isTarget: false,
    color: '#2c7569',
  },
  RIGHT: {
    id: 'RIGHT',
    name: 'RIGHT',
    locationName: 'Sel ERA5 timur',
    role: 'Konteks Lereng Timur',
    lat: -6.75,
    lng: 107.25,
    elevation: 288,
    isTarget: false,
    color: '#5f7965',
  },
};

export const TIMELINE_FRAMES: TimelineFrame[] = [
  {
    offset: 'T-2h',
    timeWib: '12:30 WIB',
    description: 'Awan konvektif orografis mulai tumbuh di lereng barat & selatan',
    radarIntensityFactor: 0.4,
    telemetries: {
      MAIN: { nodeId: 'MAIN', tempCelsius: 11.2, dewPointCelsius: 9.8, rainIntensityMmH: 2.1, humidityPercent: 88, windSpeedKmh: 18, windDirectionDeg: 230, pressureHpa: 710, confidencePercent: 94, spatialWeightToMain: 1.0 },
      UP: { nodeId: 'UP', tempCelsius: 19.5, dewPointCelsius: 16.0, rainIntensityMmH: 0.5, humidityPercent: 78, windSpeedKmh: 12, windDirectionDeg: 210, pressureHpa: 860, confidencePercent: 91, spatialWeightToMain: 0.22 },
      DOWN: { nodeId: 'DOWN', tempCelsius: 22.8, dewPointCelsius: 19.2, rainIntensityMmH: 4.8, humidityPercent: 85, windSpeedKmh: 15, windDirectionDeg: 200, pressureHpa: 905, confidencePercent: 89, spatialWeightToMain: 0.35 },
      LEFT: { nodeId: 'LEFT', tempCelsius: 21.0, dewPointCelsius: 18.1, rainIntensityMmH: 3.5, humidityPercent: 84, windSpeedKmh: 20, windDirectionDeg: 240, pressureHpa: 890, confidencePercent: 90, spatialWeightToMain: 0.28 },
      RIGHT: { nodeId: 'RIGHT', tempCelsius: 20.2, dewPointCelsius: 16.8, rainIntensityMmH: 0.8, humidityPercent: 79, windSpeedKmh: 10, windDirectionDeg: 220, pressureHpa: 878, confidencePercent: 88, spatialWeightToMain: 0.15 },
    }
  },
  {
    offset: 'T-1h',
    timeWib: '13:30 WIB',
    description: 'Adveksi kelembapan menguat menuju Mandalawangi Puncak',
    radarIntensityFactor: 0.75,
    telemetries: {
      MAIN: { nodeId: 'MAIN', tempCelsius: 9.4, dewPointCelsius: 8.9, rainIntensityMmH: 8.4, humidityPercent: 94, windSpeedKmh: 26, windDirectionDeg: 225, pressureHpa: 708, confidencePercent: 92, spatialWeightToMain: 1.0 },
      UP: { nodeId: 'UP', tempCelsius: 18.2, dewPointCelsius: 16.5, rainIntensityMmH: 3.2, humidityPercent: 86, windSpeedKmh: 18, windDirectionDeg: 215, pressureHpa: 858, confidencePercent: 90, spatialWeightToMain: 0.25 },
      DOWN: { nodeId: 'DOWN', tempCelsius: 21.5, dewPointCelsius: 19.8, rainIntensityMmH: 10.2, humidityPercent: 91, windSpeedKmh: 22, windDirectionDeg: 195, pressureHpa: 902, confidencePercent: 88, spatialWeightToMain: 0.38 },
      LEFT: { nodeId: 'LEFT', tempCelsius: 19.8, dewPointCelsius: 18.6, rainIntensityMmH: 12.0, humidityPercent: 92, windSpeedKmh: 28, windDirectionDeg: 245, pressureHpa: 886, confidencePercent: 93, spatialWeightToMain: 0.31 },
      RIGHT: { nodeId: 'RIGHT', tempCelsius: 19.0, dewPointCelsius: 17.2, rainIntensityMmH: 2.1, humidityPercent: 83, windSpeedKmh: 14, windDirectionDeg: 210, pressureHpa: 875, confidencePercent: 87, spatialWeightToMain: 0.18 },
    }
  },
  {
    offset: 'Now',
    timeWib: '14:30 WIB',
    description: 'Puncak intensitas hujan konvektif di Puncak Pangrango (3,008m)',
    radarIntensityFactor: 1.0,
    telemetries: {
      MAIN: { nodeId: 'MAIN', tempCelsius: 8.2, dewPointCelsius: 8.0, rainIntensityMmH: 14.8, humidityPercent: 98, windSpeedKmh: 34, windDirectionDeg: 220, pressureHpa: 705, confidencePercent: 89, spatialWeightToMain: 1.0 },
      UP: { nodeId: 'UP', tempCelsius: 17.0, dewPointCelsius: 16.2, rainIntensityMmH: 6.4, humidityPercent: 92, windSpeedKmh: 22, windDirectionDeg: 220, pressureHpa: 855, confidencePercent: 92, spatialWeightToMain: 0.27 },
      DOWN: { nodeId: 'DOWN', tempCelsius: 20.0, dewPointCelsius: 19.1, rainIntensityMmH: 16.5, humidityPercent: 95, windSpeedKmh: 28, windDirectionDeg: 190, pressureHpa: 898, confidencePercent: 91, spatialWeightToMain: 0.41 },
      LEFT: { nodeId: 'LEFT', tempCelsius: 18.5, dewPointCelsius: 18.0, rainIntensityMmH: 18.2, humidityPercent: 96, windSpeedKmh: 32, windDirectionDeg: 250, pressureHpa: 882, confidencePercent: 94, spatialWeightToMain: 0.34 },
      RIGHT: { nodeId: 'RIGHT', tempCelsius: 18.1, dewPointCelsius: 16.9, rainIntensityMmH: 5.0, humidityPercent: 88, windSpeedKmh: 18, windDirectionDeg: 205, pressureHpa: 872, confidencePercent: 86, spatialWeightToMain: 0.20 },
    }
  },
  {
    offset: 'T+1h',
    timeWib: '15:30 WIB',
    description: 'Prediksi ST-GNN: Pergeseran inti hujan ke lereng utara & timur',
    radarIntensityFactor: 0.7,
    telemetries: {
      MAIN: { nodeId: 'MAIN', tempCelsius: 7.8, dewPointCelsius: 7.5, rainIntensityMmH: 11.2, humidityPercent: 96, windSpeedKmh: 29, windDirectionDeg: 210, pressureHpa: 706, confidencePercent: 86, spatialWeightToMain: 1.0 },
      UP: { nodeId: 'UP', tempCelsius: 16.5, dewPointCelsius: 15.8, rainIntensityMmH: 12.8, humidityPercent: 94, windSpeedKmh: 25, windDirectionDeg: 215, pressureHpa: 856, confidencePercent: 89, spatialWeightToMain: 0.36 },
      DOWN: { nodeId: 'DOWN', tempCelsius: 20.5, dewPointCelsius: 18.8, rainIntensityMmH: 7.0, humidityPercent: 90, windSpeedKmh: 20, windDirectionDeg: 185, pressureHpa: 900, confidencePercent: 87, spatialWeightToMain: 0.28 },
      LEFT: { nodeId: 'LEFT', tempCelsius: 19.2, dewPointCelsius: 17.8, rainIntensityMmH: 8.5, humidityPercent: 89, windSpeedKmh: 24, windDirectionDeg: 240, pressureHpa: 884, confidencePercent: 90, spatialWeightToMain: 0.26 },
      RIGHT: { nodeId: 'RIGHT', tempCelsius: 17.5, dewPointCelsius: 16.5, rainIntensityMmH: 14.1, humidityPercent: 93, windSpeedKmh: 22, windDirectionDeg: 210, pressureHpa: 870, confidencePercent: 85, spatialWeightToMain: 0.32 },
    }
  },
  {
    offset: 'T+2h',
    timeWib: '16:30 WIB',
    description: 'Peluruhan sistem hujan konvektif, penurunan suhu signifikan',
    radarIntensityFactor: 0.35,
    telemetries: {
      MAIN: { nodeId: 'MAIN', tempCelsius: 6.9, dewPointCelsius: 6.5, rainIntensityMmH: 4.1, humidityPercent: 92, windSpeedKmh: 20, windDirectionDeg: 200, pressureHpa: 709, confidencePercent: 82, spatialWeightToMain: 1.0 },
      UP: { nodeId: 'UP', tempCelsius: 16.0, dewPointCelsius: 15.0, rainIntensityMmH: 5.2, humidityPercent: 88, windSpeedKmh: 16, windDirectionDeg: 200, pressureHpa: 859, confidencePercent: 85, spatialWeightToMain: 0.29 },
      DOWN: { nodeId: 'DOWN', tempCelsius: 21.0, dewPointCelsius: 18.2, rainIntensityMmH: 2.1, humidityPercent: 82, windSpeedKmh: 14, windDirectionDeg: 180, pressureHpa: 903, confidencePercent: 84, spatialWeightToMain: 0.20 },
      LEFT: { nodeId: 'LEFT', tempCelsius: 19.8, dewPointCelsius: 17.0, rainIntensityMmH: 3.0, humidityPercent: 82, windSpeedKmh: 15, windDirectionDeg: 230, pressureHpa: 887, confidencePercent: 86, spatialWeightToMain: 0.22 },
      RIGHT: { nodeId: 'RIGHT', tempCelsius: 17.0, dewPointCelsius: 15.8, rainIntensityMmH: 6.8, humidityPercent: 87, windSpeedKmh: 17, windDirectionDeg: 200, pressureHpa: 873, confidencePercent: 83, spatialWeightToMain: 0.25 },
    }
  }
];

export const PROBABILISTIC_FORECAST_DATA: Record<NodeId, ForecastPoint[]> = {
  MAIN: [
    { timeLabel: '12:00', timestamp: 'T-2h.5', p10: 0.5, p50: 1.2, p90: 2.8, observed: 1.0 },
    { timeLabel: '12:30', timestamp: 'T-2h', p10: 1.0, p50: 2.1, p90: 4.2, observed: 2.1 },
    { timeLabel: '13:00', timestamp: 'T-1h.5', p10: 3.2, p50: 5.4, p90: 8.9, observed: 5.2 },
    { timeLabel: '13:30', timestamp: 'T-1h', p10: 5.5, p50: 8.4, p90: 13.1, observed: 8.4 },
    { timeLabel: '14:00', timestamp: 'Now.5', p10: 8.0, p50: 12.0, p90: 17.5, observed: 12.2 },
    { timeLabel: '14:30', timestamp: 'Now', p10: 10.2, p50: 14.8, p90: 22.4, observed: 14.8 },
    { timeLabel: '15:00', timestamp: 'T+0.5h', p10: 8.8, p50: 13.5, p90: 20.1 },
    { timeLabel: '15:30', timestamp: 'T+1h', p10: 6.9, p50: 11.2, p90: 17.8 },
    { timeLabel: '16:00', timestamp: 'T+1.5h', p10: 4.5, p50: 7.5, p90: 12.4 },
    { timeLabel: '16:30', timestamp: 'T+2h', p10: 2.1, p50: 4.1, p90: 8.0 },
    { timeLabel: '17:00', timestamp: 'T+2.5h', p10: 0.8, p50: 2.0, p90: 4.5 },
  ],
  UP: [
    { timeLabel: '12:00', timestamp: 'T-2h.5', p10: 0.0, p50: 0.2, p90: 0.8, observed: 0.1 },
    { timeLabel: '12:30', timestamp: 'T-2h', p10: 0.1, p50: 0.5, p90: 1.5, observed: 0.5 },
    { timeLabel: '13:00', timestamp: 'T-1h.5', p10: 0.8, p50: 1.8, p90: 3.8, observed: 1.7 },
    { timeLabel: '13:30', timestamp: 'T-1h', p10: 1.5, p50: 3.2, p90: 6.2, observed: 3.2 },
    { timeLabel: '14:00', timestamp: 'Now.5', p10: 2.8, p50: 4.8, p90: 8.5, observed: 4.6 },
    { timeLabel: '14:30', timestamp: 'Now', p10: 3.9, p50: 6.4, p90: 10.8, observed: 6.4 },
    { timeLabel: '15:00', timestamp: 'T+0.5h', p10: 6.5, p50: 10.1, p90: 15.2 },
    { timeLabel: '15:30', timestamp: 'T+1h', p10: 8.0, p50: 12.8, p90: 18.5 },
    { timeLabel: '16:00', timestamp: 'T+1.5h', p10: 5.2, p50: 8.9, p90: 13.8 },
    { timeLabel: '16:30', timestamp: 'T+2h', p10: 2.8, p50: 5.2, p90: 9.0 },
    { timeLabel: '17:00', timestamp: 'T+2.5h', p10: 1.0, p50: 2.4, p90: 5.0 },
  ],
  DOWN: [
    { timeLabel: '12:00', timestamp: 'T-2h.5', p10: 1.2, p50: 2.8, p90: 5.0, observed: 2.5 },
    { timeLabel: '12:30', timestamp: 'T-2h', p10: 2.5, p50: 4.8, p90: 8.2, observed: 4.8 },
    { timeLabel: '13:00', timestamp: 'T-1h.5', p10: 4.8, p50: 7.9, p90: 12.5, observed: 7.8 },
    { timeLabel: '13:30', timestamp: 'T-1h', p10: 6.8, p50: 10.2, p90: 15.8, observed: 10.2 },
    { timeLabel: '14:00', timestamp: 'Now.5', p10: 9.5, p50: 13.8, p90: 20.2, observed: 13.5 },
    { timeLabel: '14:30', timestamp: 'Now', p10: 11.8, p50: 16.5, p90: 24.1, observed: 16.5 },
    { timeLabel: '15:00', timestamp: 'T+0.5h', p10: 8.2, p50: 12.0, p90: 18.0 },
    { timeLabel: '15:30', timestamp: 'T+1h', p10: 4.5, p50: 7.0, p90: 11.5 },
    { timeLabel: '16:00', timestamp: 'T+1.5h', p10: 2.1, p50: 4.0, p90: 7.2 },
    { timeLabel: '16:30', timestamp: 'T+2h', p10: 0.8, p50: 2.1, p90: 4.5 },
    { timeLabel: '17:00', timestamp: 'T+2.5h', p10: 0.2, p50: 0.8, p90: 2.1 },
  ],
  LEFT: [
    { timeLabel: '12:00', timestamp: 'T-2h.5', p10: 0.8, p50: 1.8, p90: 3.5, observed: 1.6 },
    { timeLabel: '12:30', timestamp: 'T-2h', p10: 1.8, p50: 3.5, p90: 6.0, observed: 3.5 },
    { timeLabel: '13:00', timestamp: 'T-1h.5', p10: 4.2, p50: 7.2, p90: 11.0, observed: 7.0 },
    { timeLabel: '13:30', timestamp: 'T-1h', p10: 7.5, p50: 12.0, p90: 17.5, observed: 12.0 },
    { timeLabel: '14:00', timestamp: 'Now.5', p10: 10.2, p50: 15.4, p90: 22.0, observed: 15.1 },
    { timeLabel: '14:30', timestamp: 'Now', p10: 12.5, p50: 18.2, p90: 26.5, observed: 18.2 },
    { timeLabel: '15:00', timestamp: 'T+0.5h', p10: 8.8, p50: 13.1, p90: 19.4 },
    { timeLabel: '15:30', timestamp: 'T+1h', p10: 5.2, p50: 8.5, p90: 13.2 },
    { timeLabel: '16:00', timestamp: 'T+1.5h', p10: 2.8, p50: 5.1, p90: 8.8 },
    { timeLabel: '16:30', timestamp: 'T+2h', p10: 1.2, p50: 3.0, p90: 5.6 },
    { timeLabel: '17:00', timestamp: 'T+2.5h', p10: 0.4, p50: 1.2, p90: 2.8 },
  ],
  RIGHT: [
    { timeLabel: '12:00', timestamp: 'T-2h.5', p10: 0.0, p50: 0.3, p90: 1.0, observed: 0.2 },
    { timeLabel: '12:30', timestamp: 'T-2h', p10: 0.2, p50: 0.8, p90: 2.0, observed: 0.8 },
    { timeLabel: '13:00', timestamp: 'T-1h.5', p10: 0.5, p50: 1.2, p90: 3.0, observed: 1.1 },
    { timeLabel: '13:30', timestamp: 'T-1h', p10: 0.9, p50: 2.1, p90: 4.8, observed: 2.1 },
    { timeLabel: '14:00', timestamp: 'Now.5', p10: 1.8, p50: 3.5, p90: 7.2, observed: 3.4 },
    { timeLabel: '14:30', timestamp: 'Now', p10: 2.9, p50: 5.0, p90: 9.2, observed: 5.0 },
    { timeLabel: '15:00', timestamp: 'T+0.5h', p10: 6.2, p50: 9.8, p90: 15.0 },
    { timeLabel: '15:30', timestamp: 'T+1h', p10: 9.1, p50: 14.1, p90: 20.8 },
    { timeLabel: '16:00', timestamp: 'T+1.5h', p10: 6.8, p50: 10.5, p90: 16.2 },
    { timeLabel: '16:30', timestamp: 'T+2h', p10: 3.8, p50: 6.8, p90: 11.0 },
    { timeLabel: '17:00', timestamp: 'T+2.5h', p10: 1.5, p50: 3.1, p90: 5.8 },
  ]
};

export const CURRENT_RISK_METRICS: RiskMetrics = {
  hypothermiaRiskLevel: 'HIGH',
  hypothermiaElevationMin: 2500,
  windChillCelsius: -2.4,
  apparentTempCelsius: -2.4,
  orographicAlert: 'Peringatan Dini: Adveksi Udara Lembap & Hujan Ekstrem Orografis Mandalawangi (Elev > 2.8k mdpl)',
  slopeInstabilityRisk: 'HIGH',
};

export const CURRENT_MODEL_METRICS: ModelMetrics = {
  // Full-model precipitation evaluation, result_test/full_model/metrics.json.
  csi: 0.2661,
  pod: 0.7398,
  far: 0.7065,
  reliabilityPercent: 74,
  rmse: 0.792,
  mae: 0.353,
  crps: 0.252,
  brierScore: 0.0494,
  corr: 0.639,
};
