import { HistoricalAnalog } from '../types';

export const HISTORICAL_ANALOGS: HistoricalAnalog[] = [
  { id: 'SAMPLE-01', rank: 1, datetime: 'Contoh A', patternType: 'Pola sampel', similarityPercent: 94.2, distanceL2: .142, rainRateMmH: 18.5, accumulatedRainMm: 68.4, peakWindKmh: 42, tempCelsius: 16.2, humidityPercent: 95, pressureHpa: 768.4, outcomeSummary: 'Nilai fixture untuk demonstrasi perbandingan antarmuka; bukan catatan kejadian.' },
  { id: 'SAMPLE-02', rank: 2, datetime: 'Contoh B', patternType: 'Pola sampel', similarityPercent: 88.7, distanceL2: .285, rainRateMmH: 12.8, accumulatedRainMm: 52.1, peakWindKmh: 38, tempCelsius: 17.8, humidityPercent: 92, pressureHpa: 770.1, outcomeSummary: 'Nilai fixture untuk demonstrasi perbandingan antarmuka; bukan catatan kejadian.' },
  { id: 'SAMPLE-03', rank: 3, datetime: 'Contoh C', patternType: 'Pola sampel', similarityPercent: 81.5, distanceL2: .419, rainRateMmH: 9.4, accumulatedRainMm: 44.8, peakWindKmh: 35, tempCelsius: 18.5, humidityPercent: 89, pressureHpa: 771.5, outcomeSummary: 'Nilai fixture untuk demonstrasi perbandingan antarmuka; bukan catatan kejadian.' },
];
