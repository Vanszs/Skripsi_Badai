import { DashboardLiveSnapshot } from '../types';

const endpoint = import.meta.env.VITE_DASHBOARD_LIVE_URL ?? '/api/v1/dashboard/live';

export async function fetchDashboardLive(signal?: AbortSignal): Promise<DashboardLiveSnapshot> {
  const response = await fetch(endpoint, { headers: { Accept: 'application/json' }, signal });
  if (!response.ok) throw new Error(`Data cuaca tidak tersedia (${response.status})`);
  const payload: unknown = await response.json();
  if (!isDashboardLiveSnapshot(payload)) throw new Error('Format data cuaca tidak sesuai');
  return payload;
}

function isDashboardLiveSnapshot(value: unknown): value is DashboardLiveSnapshot {
  if (!value || typeof value !== 'object') return false;
  const snapshot = value as Partial<DashboardLiveSnapshot>;
  return (snapshot.source === 'live' || snapshot.source === 'sample')
    && typeof snapshot.generatedAt === 'string'
    && Array.isArray(snapshot.mainForecast)
    && Boolean(snapshot.frame?.telemetries)
    && typeof snapshot.frame?.timeWib === 'string';
}
