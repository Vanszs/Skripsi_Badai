import React, { useState } from 'react';

export interface AtmosphericState {
  datetime: string;
  tp: number;      // Total Precipitation (mm/h)
  t2m: number;     // 2m Temperature (K or °C)
  r850: number;    // Relative Humidity 850hPa (%)
  u10: number;     // 10m Wind U (m/s)
  v10: number;     // 10m Wind V (m/s)
  cape: number;    // CAPE (J/kg)
}

export interface AnalogItem {
  id: string;
  rank: number;
  date: string;
  distanceL2: number;
  state: AtmosphericState;
  rainEvolution: number[]; // e.g. [t+1, t+2, t+3, t+4, t+5, t+6] mm/h
}

export interface HistoricalAnalogsExplorerProps {
  currentQueryState: AtmosphericState;
  analogs?: AnalogItem[];
}

const DEFAULT_CURRENT_QUERY: AtmosphericState = {
  datetime: '2024-03-15 14:00:00',
  tp: 18.5,
  t2m: 298.15,
  r850: 92.4,
  u10: 3.2,
  v10: -4.1,
  cape: 1850.0,
};

const DEFAULT_ANALOGS: AnalogItem[] = [
  {
    id: 'ana-1',
    rank: 1,
    date: '2020-01-01 03:00:00',
    distanceL2: 0.142,
    state: {
      datetime: '2020-01-01 03:00:00',
      tp: 19.2,
      t2m: 297.85,
      r850: 94.1,
      u10: 3.5,
      v10: -4.5,
      cape: 1920.0,
    },
    rainEvolution: [21.0, 28.5, 34.0, 15.2, 8.0, 2.1],
  },
  {
    id: 'ana-2',
    rank: 2,
    date: '2021-02-20 18:00:00',
    distanceL2: 0.287,
    state: {
      datetime: '2021-02-20 18:00:00',
      tp: 16.8,
      t2m: 298.50,
      r850: 90.8,
      u10: 2.8,
      v10: -3.8,
      cape: 1780.0,
    },
    rainEvolution: [18.2, 22.0, 19.5, 11.0, 5.4, 1.2],
  },
  {
    id: 'ana-3',
    rank: 3,
    date: '2023-07-12 15:00:00',
    distanceL2: 0.415,
    state: {
      datetime: '2023-07-12 15:00:00',
      tp: 15.1,
      t2m: 299.10,
      r850: 88.5,
      u10: 4.1,
      v10: -5.0,
      cape: 1650.0,
    },
    rainEvolution: [14.0, 16.8, 12.1, 7.5, 3.0, 0.5],
  },
];

export const HistoricalAnalogsExplorer: React.FC<HistoricalAnalogsExplorerProps> = ({
  currentQueryState = DEFAULT_CURRENT_QUERY,
  analogs = DEFAULT_ANALOGS,
}) => {
  const [selectedAnalogId, setSelectedAnalogId] = useState<string>(analogs[0]?.id || '');

  const selectedAnalog = analogs.find((a) => a.id === selectedAnalogId) || analogs[0];

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '24px', backgroundColor: '#0f172a', color: '#f8fafc', borderRadius: '12px' }}>
      <header style={{ marginBottom: '24px', borderBottom: '1px solid #334155', paddingBottom: '16px' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, margin: '0 0 8px 0', color: '#38bdf8' }}>
          FAISS Historical Analogs Search Explorer (k=3)
        </h2>
        <p style={{ margin: 0, color: '#94a3b8', fontSize: '0.9rem' }}>
          IndexFlatL2 Search over 144-dimensional atmospheric feature embeddings (<code style={{ color: '#cbd5e1' }}>R_t = kNN(c_t, D_train, k=3)</code>)
        </p>
      </header>

      {/* k=3 Top Analog Cards */}
      <section style={{ marginBottom: '28px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px', color: '#e2e8f0' }}>
          Top Historical Extreme Weather Analogs
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {analogs.map((item) => {
            const isSelected = item.id === selectedAnalogId;
            return (
              <div
                key={item.id}
                onClick={() => setSelectedAnalogId(item.id)}
                style={{
                  padding: '16px',
                  borderRadius: '8px',
                  border: isSelected ? '2px solid #38bdf8' : '1px solid #334155',
                  backgroundColor: isSelected ? '#1e293b' : '#1e293b80',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <span style={{ fontWeight: 700, padding: '2px 8px', borderRadius: '4px', backgroundColor: '#0284c7', fontSize: '0.85rem' }}>
                    Rank #{item.rank}
                  </span>
                  <span style={{ fontSize: '0.85rem', color: '#cbd5e1', fontFamily: 'monospace' }}>
                    L2 Dist: {item.distanceL2.toFixed(4)}
                  </span>
                </div>
                <div style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f1f5f9', marginBottom: '6px' }}>
                  📅 {item.date}
                </div>
                <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                  Base Precipitation: <strong style={{ color: '#38bdf8' }}>{item.state.tp} mm/h</strong>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Atmospheric State Comparison & Rain Trajectory */}
      {selectedAnalog && (
        <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '28px' }}>
          {/* Comparison Table */}
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px', border: '1px solid #334155' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '1rem', color: '#38bdf8' }}>
              Atmospheric Condition Comparison (x_q vs x_i)
            </h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #475569', textAlign: 'left' }}>
                  <th style={{ padding: '8px', color: '#94a3b8' }}>Parameter</th>
                  <th style={{ padding: '8px', color: '#38bdf8' }}>Current (x_q)</th>
                  <th style={{ padding: '8px', color: '#f43f5e' }}>Analog #{selectedAnalog.rank} (x_i)</th>
                  <th style={{ padding: '8px', color: '#cbd5e1' }}>Δ (Diff)</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { label: 'Precipitation (mm/h)', key: 'tp', unit: '' },
                  { label: 'Temp 2m (K)', key: 't2m', unit: '' },
                  { label: 'RH 850hPa (%)', key: 'r850', unit: '' },
                  { label: 'Wind U10 (m/s)', key: 'u10', unit: '' },
                  { label: 'Wind V10 (m/s)', key: 'v10', unit: '' },
                  { label: 'CAPE (J/kg)', key: 'cape', unit: '' },
                ].map((row) => {
                  const valQ = (currentQueryState as any)[row.key];
                  const valI = (selectedAnalog.state as any)[row.key];
                  const diff = valQ - valI;
                  return (
                    <tr key={row.key} style={{ borderBottom: '1px solid #334155' }}>
                      <td style={{ padding: '8px', color: '#e2e8f0' }}>{row.label}</td>
                      <td style={{ padding: '8px', fontWeight: 600, color: '#f1f5f9' }}>{valQ}</td>
                      <td style={{ padding: '8px', fontWeight: 600, color: '#f1f5f9' }}>{valI}</td>
                      <td style={{ padding: '8px', color: diff >= 0 ? '#4ade80' : '#f87171' }}>
                        {diff > 0 ? `+${diff.toFixed(2)}` : diff.toFixed(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Post-Event Rain Evolution Trajectory */}
          <div style={{ backgroundColor: '#1e293b', padding: '16px', borderRadius: '8px', border: '1px solid #334155' }}>
            <h4 style={{ margin: '0 0 12px 0', fontSize: '1rem', color: '#38bdf8' }}>
              Rain Evolution Trajectory (t+1h to t+6h)
            </h4>
            <div style={{ height: '140px', display: 'flex', alignItems: 'flex-end', gap: '12px', padding: '16px 8px 8px 8px', borderBottom: '1px solid #475569' }}>
              {selectedAnalog.rainEvolution.map((rain, idx) => {
                const heightPercent = Math.min(100, (rain / 40) * 100);
                return (
                  <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <span style={{ fontSize: '0.75rem', color: '#38bdf8', marginBottom: '4px' }}>{rain}</span>
                    <div
                      style={{
                        width: '100%',
                        height: `${heightPercent}%`,
                        backgroundColor: '#0284c7',
                        borderRadius: '4px 4px 0 0',
                        transition: 'height 0.3s ease',
                      }}
                    />
                    <span style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '4px' }}>t+{idx + 1}h</span>
                  </div>
                );
              })}
            </div>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '12px', marginBottom: 0 }}>
              Histogram memperlihatkan evolusi temporal presipitasi historis pasca jam-t analog.
            </p>
          </div>
        </section>
      )}

      {/* Scientific Rationale for Denoising Diffusion Models */}
      <section style={{ backgroundColor: '#1e293b', padding: '20px', borderRadius: '8px', border: '1px solid #0284c7' }}>
        <h4 style={{ margin: '0 0 8px 0', fontSize: '1.05rem', color: '#38bdf8' }}>
          🧠 Peran Ilmiah Analog Historis dalam Denoising Diffusion Model
        </h4>
        <ul style={{ margin: 0, paddingLeft: '20px', color: '#cbd5e1', fontSize: '0.88rem', lineHeight: '1.6' }}>
          <li>
            <strong>Structural Conditioning Prior (<code style={{ color: '#38bdf8' }}>R_t</code>):</strong> Multi-variate atmospheric embedding yang ditarik via FAISS (<code style={{ color: '#cbd5e1' }}>IndexFlatL2</code>) berfungsi sebagai pengarah (conditioning guidance) pada reverse diffusion step.
          </li>
          <li>
            <strong>Pencegahan Mode Collapse pada Cuaca Ekstrem:</strong> Model difusi murni sering menghasilkan prediksi yang terlalu smooth (blur). Kehadiran analog historis riil menyuplai pola fisik nyata (spatiotemporal dynamics) kejadian konvektif ekstrem masa lalu.
          </li>
          <li>
            <strong>Manifold Constraint:</strong> Menjaga trajectory proses reverse noise-to-sample agar tetap berada dalam ruang manifold fisik atmosferik yang realistis dan tidak melanggar hukum konservasi massa/energi.
          </li>
        </ul>
      </section>
    </div>
  );
};

export default HistoricalAnalogsExplorer;
