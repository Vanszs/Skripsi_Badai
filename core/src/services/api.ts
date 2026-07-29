import axios, { AxiosInstance } from 'axios';
import { useQuery, UseQueryResult } from '@tanstack/react-query';

// --- TYPES ---
export interface NodeInfo {
  name: string;
  role: 'main' | 'surrounding';
  alias: string;
  lat: number;
  lon: number;
  elevation_m: number;
}

export interface WeatherSample {
  timestamp: string;
  precipitation: number; // mm/hour
  wind_speed_10m: number; // m/s
  relative_humidity_2m: number; // %
  temperature_2m?: number; // °C
}

export interface NodeData {
  node_name: string;
  lat: number;
  lon: number;
  historical: WeatherSample[];
}

export interface EnsemblePrediction {
  sample_id: number;
  precipitation: number[]; // 6h forecast
  wind_speed_10m: number[];
  relative_humidity_2m: number[];
}

export interface FAISSAnalog {
  analog_id: number;
  distance: number;
  timestamp: string;
  precipitation_pattern: number[];
}

export interface NowcastResponse {
  timestamp: string;
  target_node: string;
  nodes: NodeData[];
  ensemble_predictions: EnsemblePrediction[]; // 50 samples
  faiss_analogs: FAISSAnalog[]; // k=3
  ensemble_summary: {
    precipitation_median: number[];
    precipitation_p10: number[];
    precipitation_p90: number[];
  };
}

// --- CONSTANTS ---
export const PANGRANGO_NODES: NodeInfo[] = [
  { name: 'MAIN', role: 'main', alias: 'Puncak', lat: -6.75, lon: 107.00, elevation_m: 1529.0 },
  { name: 'UP', role: 'surrounding', alias: 'Up', lat: -6.50, lon: 107.00, elevation_m: 162.0 },
  { name: 'DOWN', role: 'surrounding', alias: 'Down', lat: -7.00, lon: 107.00, elevation_m: 0.0 },
  { name: 'LEFT', role: 'surrounding', alias: 'Left', lat: -6.75, lon: 106.75, elevation_m: 823.0 },
  { name: 'RIGHT', role: 'surrounding', alias: 'Right', lat: -6.75, lon: 107.25, elevation_m: 288.0 },
];

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
  headers: { 'Content-Type': 'application/json' },
});

// --- REALISTIC MOCK ENGINE ---
// ponytail: deterministic seeded log1p z-score pseudo-random generator
function pseudorand(seed: number): number {
  const x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}

// Real statistics: log1p transformed precipitation, median ~0-25 mm/h
function generateSamplePrecip(seed: number): number {
  const r = pseudorand(seed);
  if (r < 0.6) return 0; // 60% dry hours
  const logVal = pseudorand(seed + 1) * 2.5; // z-score log1p scale
  const precip = Math.expm1(logVal);
  return Math.min(Math.round(precip * 100) / 100, 25.0);
}

export function generateMockNowcast(targetTimestamp?: string): NowcastResponse {
  const now = targetTimestamp ? new Date(targetTimestamp) : new Date();
  const baseTime = now.getTime();
  let seedCounter = baseTime % 100000;

  // 1. Generate 6 hours historical data for 5 nodes
  const nodes: NodeData[] = PANGRANGO_NODES.map((node) => {
    const historical: WeatherSample[] = [];
    for (let i = 5; i >= 0; i--) {
      const t = new Date(baseTime - i * 3600 * 1000).toISOString();
      historical.push({
        timestamp: t,
        precipitation: generateSamplePrecip(seedCounter++),
        wind_speed_10m: Math.round((1.5 + pseudorand(seedCounter++) * 4.5) * 10) / 10,
        relative_humidity_2m: Math.round(70 + pseudorand(seedCounter++) * 25),
        temperature_2m: Math.round((18.0 - node.elevation_m * 0.0065 + pseudorand(seedCounter++) * 2) * 10) / 10,
      });
    }
    return {
      node_name: node.name,
      lat: node.lat,
      lon: node.lon,
      historical,
    };
  });

  // 2. Generate 50 ensemble prediction samples for 6 future hours
  const ensemble_predictions: EnsemblePrediction[] = [];
  for (let s = 0; s < 50; s++) {
    const prec: number[] = [];
    const wind: number[] = [];
    const rh: number[] = [];
    for (let h = 1; h <= 6; h++) {
      prec.push(generateSamplePrecip(seedCounter++));
      wind.push(Math.round((2.0 + pseudorand(seedCounter++) * 5.0) * 10) / 10);
      rh.push(Math.round(65 + pseudorand(seedCounter++) * 30));
    }
    ensemble_predictions.push({
      sample_id: s + 1,
      precipitation: prec,
      wind_speed_10m: wind,
      relative_humidity_2m: rh,
    });
  }

  // Calculate ensemble summary (median, p10, p90 for 6 hours)
  const prec_median: number[] = [];
  const prec_p10: number[] = [];
  const prec_p90: number[] = [];

  for (let h = 0; h < 6; h++) {
    const vals = ensemble_predictions.map((e) => e.precipitation[h]).sort((a, b) => a - b);
    prec_p10.push(vals[Math.floor(vals.length * 0.1)]);
    prec_median.push(vals[Math.floor(vals.length * 0.5)]);
    prec_p90.push(vals[Math.floor(vals.length * 0.9)]);
  }

  // 3. Generate k=3 FAISS analogs
  const faiss_analogs: FAISSAnalog[] = [1, 2, 3].map((k) => ({
    analog_id: k,
    distance: Math.round((0.15 + k * 0.08 + pseudorand(seedCounter++) * 0.05) * 1000) / 1000,
    timestamp: new Date(baseTime - (365 * k + 12) * 24 * 3600 * 1000).toISOString(),
    precipitation_pattern: Array.from({ length: 6 }, () => generateSamplePrecip(seedCounter++)),
  }));

  return {
    timestamp: now.toISOString(),
    target_node: 'MAIN',
    nodes,
    ensemble_predictions,
    faiss_analogs,
    ensemble_summary: {
      precipitation_median: prec_median,
      precipitation_p10: prec_p10,
      precipitation_p90: prec_p90,
    },
  };
}

// Helper fetch with mock fallback & delay
export async function fetchNowcast(timestamp?: string): Promise<NowcastResponse> {
  try {
    const response = await apiClient.get<NowcastResponse>('/nowcast', {
      params: { timestamp },
    });
    return response.data;
  } catch {
    // Backend offline / error fallback to realistic mock
    await new Promise((resolve) => setTimeout(resolve, 300));
    return generateMockNowcast(timestamp);
  }
}

// TanStack Query Hook
export function useNowcastQuery(timestamp?: string): UseQueryResult<NowcastResponse, Error> {
  return useQuery<NowcastResponse, Error>({
    queryKey: ['nowcast', timestamp || 'latest'],
    queryFn: () => fetchNowcast(timestamp),
    staleTime: 60 * 1000,
  });
}
