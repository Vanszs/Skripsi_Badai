import { DashboardLiveSnapshot, NowcastDistribution, NowcastVariableSet } from '../types';

const endpoint = import.meta.env.VITE_DASHBOARD_LIVE_URL ?? '/api/v1/dashboard/live';

export async function fetchDashboardLive(signal?: AbortSignal): Promise<DashboardLiveSnapshot> {
  const response = await fetch(endpoint, { headers: { Accept: 'application/json' }, signal });
  if (!response.ok) throw new Error(`Sumber API tidak tersedia (${response.status})`);
  const payload: unknown = await response.json();
  if (!isDashboardLiveSnapshot(payload)) throw new Error('Format sumber API tidak sesuai');
  return payload;
}

function isNowcast(value: unknown): value is NowcastDistribution {
  if (!value || typeof value !== 'object') return false;
  const item = value as Partial<NowcastDistribution>;
  return item.horizon === 'T+1h' && typeof item.validAtWib === 'string' && Array.isArray(item.samples) && item.samples.length === 50 && typeof item.mean === 'number' && typeof item.p10 === 'number' && typeof item.p25 === 'number' && typeof item.p50 === 'number' && typeof item.p75 === 'number' && typeof item.p90 === 'number' && typeof item.unit === 'string' && typeof item.variable === 'string';
}

function isNowcastSet(value: unknown): value is NowcastVariableSet {
  if (!value || typeof value !== 'object') return false;
  const set = value as Partial<NowcastVariableSet>;
  return isNowcast(set.precipitation) && isNowcast(set.wind_speed_10m) && isNowcast(set.relative_humidity_2m);
}

function isDashboardLiveSnapshot(value: unknown): value is DashboardLiveSnapshot {
  if (!value || typeof value !== 'object') return false;
  const snapshot = value as Partial<DashboardLiveSnapshot>;
  return (snapshot.source === 'live' || snapshot.source === 'sample') && typeof snapshot.generatedAt === 'string' && isNowcastSet(snapshot.nowcast) && Boolean(snapshot.frame?.telemetries) && typeof snapshot.frame?.timeWib === 'string';
}
