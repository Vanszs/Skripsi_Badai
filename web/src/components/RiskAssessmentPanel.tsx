import React from 'react';
import {
  Snowflake,
  Wind,
  AlertTriangle,
  Mountain,
  ShieldCheck,
  ShieldAlert,
  AlertCircle,
  LifeBuoy,
  ChevronRight,
  Thermometer,
} from 'lucide-react';
import { RiskMetrics } from '../types';

interface RiskAssessmentPanelProps {
  riskMetrics: RiskMetrics;
  onOpenDetails: () => void;
}

export type RiskLevel = 'LOW' | 'MODERATE' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'EXTREME';

export interface RiskBadgeConfig {
  label: string;
  badgeBg: string;
  textColor: string;
  borderColor: string;
  glowColor: string;
  icon: React.ReactNode;
}

export const getRiskBadgeConfig = (level: string): RiskBadgeConfig => {
  const normalized = (level || '').toUpperCase();
  switch (normalized) {
    case 'LOW':
      return {
        label: 'LOW',
        badgeBg: 'bg-[#10b981]/20',
        textColor: 'text-[#34d399]',
        borderColor: 'border-[#10b981]/50',
        glowColor: 'shadow-[0_0_12px_rgba(16,185,129,0.35)]',
        icon: <ShieldCheck className="w-3.5 h-3.5 text-[#34d399]" />,
      };
    case 'MODERATE':
    case 'MEDIUM':
      return {
        label: 'MODERATE',
        badgeBg: 'bg-[#f59e0b]/20',
        textColor: 'text-[#fbbf24]',
        borderColor: 'border-[#f59e0b]/50',
        glowColor: 'shadow-[0_0_12px_rgba(245,158,11,0.35)]',
        icon: <AlertCircle className="w-3.5 h-3.5 text-[#fbbf24]" />,
      };
    case 'HIGH':
      return {
        label: 'HIGH',
        badgeBg: 'bg-[#ef4444]/25',
        textColor: 'text-[#f87171]',
        borderColor: 'border-[#ef4444]/60',
        glowColor: 'shadow-[0_0_14px_rgba(239,68,68,0.4)]',
        icon: <AlertTriangle className="w-3.5 h-3.5 text-[#f87171]" />,
      };
    case 'CRITICAL':
    case 'EXTREME':
    default:
      return {
        label: 'EXTREME',
        badgeBg: 'bg-[#dc2626]/30',
        textColor: 'text-[#fb7185]',
        borderColor: 'border-[#ec4899]/70',
        glowColor: 'shadow-[0_0_16px_rgba(236,72,153,0.5)]',
        icon: <ShieldAlert className="w-3.5 h-3.5 text-[#fb7185] animate-pulse" />,
      };
  }
};

const RiskAssessmentPanelComponent: React.FC<RiskAssessmentPanelProps> = ({
  riskMetrics,
  onOpenDetails,
}) => {
  const hypoBadge = getRiskBadgeConfig(riskMetrics.hypothermiaRiskLevel);
  const slopeBadge = getRiskBadgeConfig(riskMetrics.slopeInstabilityRisk);

  // Highest severity calculation for safety guidance
  const isExtreme =
    riskMetrics.hypothermiaRiskLevel === 'CRITICAL' ||
    riskMetrics.hypothermiaRiskLevel === 'EXTREME' ||
    riskMetrics.slopeInstabilityRisk === 'HIGH';

  const isHigh =
    riskMetrics.hypothermiaRiskLevel === 'HIGH' ||
    riskMetrics.slopeInstabilityRisk === 'MEDIUM';

  return (
    <div role="region" aria-label="Panel Penilaian Risiko Gunung Pangrango" className="w-full h-full flex flex-col justify-between space-y-2.5">
      {/* Panel Header */}
      <div className="flex justify-between items-center">
        <h3 className="font-mono-data text-xs font-bold text-[#b9cacb] uppercase tracking-wider flex items-center gap-1.5">
          <AlertTriangle className="w-4 h-4 text-[#ffb4ab]" />
          HAL YANG PERLU DIWASPADAI
        </h3>
        <button
          onClick={onOpenDetails}
          aria-label="Lihat Detail & Matriks Penilaian Risiko Gunung"
          className="text-[10px] font-mono-data text-[#00f0ff] hover:underline cursor-pointer flex items-center gap-0.5 focus:outline-none focus:ring-1 focus:ring-[#00f0ff]"
        >
          Lihat penjelasan <ChevronRight className="w-3 h-3" />
        </button>
      </div>

      {/* Risk Cards Container */}
      <div className="space-y-2 flex-1 flex flex-col justify-center">
        {/* 1. Hypothermia Risk Card with High-Visibility Badge */}
        <div className="bg-[#171f33]/90 border border-white/10 rounded-xl p-2.5 flex items-center justify-between shadow-md hover:border-white/20 transition-all">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#ffb4ab]/15 border border-[#ffb4ab]/30 flex items-center justify-center flex-shrink-0">
              <Snowflake className="w-4 h-4 text-[#ffb4ab]" />
            </div>
            <div>
              <div className="font-mono-data text-[10px] font-bold text-[#849495] uppercase tracking-wider">
                RISIKO KEDINGINAN
              </div>
              <div className="font-body text-xs text-[#dae2fd]">
                Di atas {riskMetrics.hypothermiaElevationMin || 2500} mdpl
              </div>
            </div>
          </div>
          {/* High-Visibility Badge */}
          <div
            className={`px-2.5 py-1 rounded-full border text-xs font-mono-data font-extrabold flex items-center gap-1.5 ${hypoBadge.badgeBg} ${hypoBadge.textColor} ${hypoBadge.borderColor} ${hypoBadge.glowColor}`}
          >
            {hypoBadge.icon}
            <span>{hypoBadge.label}</span>
          </div>
        </div>

        {/* 2. Wind Chill Index Card */}
        <div className="bg-[#171f33]/90 border border-white/10 rounded-xl p-2.5 flex items-center justify-between shadow-md hover:border-[#00f0ff]/30 transition-all">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#00f0ff]/15 border border-[#00f0ff]/30 flex items-center justify-center flex-shrink-0">
              <Wind className="w-4 h-4 text-[#00f0ff]" />
            </div>
            <div>
              <div className="font-mono-data text-[10px] font-bold text-[#849495] uppercase tracking-wider">
                TERASA DINGIN KARENA ANGIN
              </div>
              <div className="font-mono-data text-xs font-bold text-[#dae2fd] flex items-center gap-1.5">
                <Thermometer className="w-3 h-3 text-[#00f0ff]" />
                Terasa seperti {riskMetrics.windChillCelsius}°C
              </div>
            </div>
          </div>
          <span className="text-[10px] font-mono-data text-[#00f0ff] bg-[#00f0ff]/10 px-2 py-0.5 rounded-md border border-[#00f0ff]/20">
            {riskMetrics.windChillCelsius < 0 ? 'Sangat berbahaya' : 'Sangat dingin'}
          </span>
        </div>

        {/* 3. Slope Instability Risk Card */}
        <div className="bg-[#171f33]/90 border border-white/10 rounded-xl p-2.5 flex items-center justify-between shadow-md hover:border-white/20 transition-all">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-[#fed639]/15 border border-[#fed639]/30 flex items-center justify-center flex-shrink-0">
              <Mountain className="w-4 h-4 text-[#fed639]" />
            </div>
            <div>
              <div className="font-mono-data text-[10px] font-bold text-[#849495] uppercase tracking-wider">
                RISIKO JALUR LICIN ATAU LONGSOR
              </div>
              <div className="font-body text-xs text-[#dae2fd]">Area lereng pegunungan</div>
            </div>
          </div>
          {/* High-Visibility Badge */}
          <div
            className={`px-2.5 py-1 rounded-full border text-xs font-mono-data font-extrabold flex items-center gap-1.5 ${slopeBadge.badgeBg} ${slopeBadge.textColor} ${slopeBadge.borderColor} ${slopeBadge.glowColor}`}
          >
            {slopeBadge.icon}
            <span>{slopeBadge.label}</span>
          </div>
        </div>
      </div>

      {/* 4. Mountain Safety Guidance Section */}
      <div className="bg-[#090d16]/90 border border-white/10 rounded-xl p-2.5 text-xs font-body space-y-1.5">
        <div className="flex items-center justify-between border-b border-white/10 pb-1">
          <span className="font-mono-data text-[10px] font-bold text-[#fed639] uppercase tracking-wider flex items-center gap-1">
            <LifeBuoy className="w-3.5 h-3.5 text-[#fed639]" />
            SARAN UNTUK PENDAKIAN
          </span>
          <span className="text-[9px] font-mono-data text-[#849495]">Cek sebelum melanjutkan</span>
        </div>

        <p className="text-[11px] text-[#dae2fd] leading-snug">
          {isExtreme
            ? 'Tunda perjalanan ke puncak. Hujan lebat dan suhu terasa di bawah 0°C dapat meningkatkan risiko kedinginan dan jalur berbahaya.'
            : isHigh
            ? 'Risiko tinggi. Hindari punggungan terbuka, gunakan pakaian hangat dan tahan air, lalu cek pembaruan cuaca sebelum berjalan lagi.'
            : 'Kondisi relatif aman. Tetap ikuti aturan pendakian dan cek pembaruan cuaca secara berkala.'}
        </p>
      </div>
    </div>
  );
};

export const RiskAssessmentPanel = React.memo(RiskAssessmentPanelComponent);
