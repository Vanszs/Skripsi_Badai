import React, { useEffect, useRef } from 'react';
import { Play, Pause, FastForward, Rewind, Clock } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { TimelineOffset } from '../types';

interface TimelineScrubberProps {
  currentOffset: TimelineOffset;
  onOffsetChange: (offset: TimelineOffset) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  dataStatus: string;
  isLive: boolean;
}

const OFFSETS: TimelineOffset[] = ['T-2h', 'T-1h', 'Now', 'T+1h', 'T+2h'];

const SPRING_TRANSITION = {
  type: 'spring',
  stiffness: 380,
  damping: 28,
};

const TimelineScrubberComponent: React.FC<TimelineScrubberProps> = ({
  currentOffset,
  onOffsetChange,
  isPlaying,
  onTogglePlay,
  dataStatus,
  isLive,
}) => {
  const shouldReduceMotion = useReducedMotion();
  const currentIndex = OFFSETS.indexOf(currentOffset);


  const currentIndexRef = useRef(currentIndex);
  useEffect(() => {
    currentIndexRef.current = currentIndex;
  }, [currentIndex]);

  const onOffsetChangeRef = useRef(onOffsetChange);
  useEffect(() => {
    onOffsetChangeRef.current = onOffsetChange;
  }, [onOffsetChange]);

  // Auto playback interval when playing
  useEffect(() => {
    if (!isPlaying) return;
    const timer = setInterval(() => {
      const nextIndex = (currentIndexRef.current + 1) % OFFSETS.length;
      onOffsetChangeRef.current(OFFSETS[nextIndex]);
    }, 2800);
    return () => clearInterval(timer);
  }, [isPlaying]);

  const progressPercent = (currentIndex / (OFFSETS.length - 1)) * 100;
  const activeTransition = shouldReduceMotion ? { duration: 0 } : SPRING_TRANSITION;

  return (
    <div 
      role="region" 
      aria-label="Kontrol waktu dan pembaruan cuaca" 
      className="glass-panel w-full max-w-4xl rounded-2xl p-3.5 flex flex-col md:flex-row items-center gap-4 border border-slate-700/60 bg-slate-900/90 backdrop-blur-xl shadow-[0_10px_35px_rgba(0,0,0,0.6)]"
    >
      {/* Playback Controls */}
      <div className="flex items-center gap-2 flex-shrink-0" aria-label="Kontrol Pemutaran Animasi">
        <motion.button
          whileHover={{ scale: shouldReduceMotion ? 1 : 1.05 }}
          whileTap={{ scale: shouldReduceMotion ? 1 : 0.95 }}
          disabled={isLive}
          onClick={() => {
            const prevIndex = (currentIndex - 1 + OFFSETS.length) % OFFSETS.length;
            onOffsetChange(OFFSETS[prevIndex]);
          }}
          className="w-11 h-11 rounded-xl bg-slate-800/80 border border-slate-600/50 text-slate-300 flex items-center justify-center hover:text-cyan-400 hover:border-cyan-500/50 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
          title="Langkah Sebelumnya"
          aria-label="Langkah Sebelumnya"
        >
          <Rewind className="w-4.5 h-4.5" />
        </motion.button>

        <motion.button
          whileHover={{ scale: shouldReduceMotion ? 1 : 1.05 }}
          whileTap={{ scale: shouldReduceMotion ? 1 : 0.94 }}
          transition={activeTransition}
          disabled={isLive}
          onClick={onTogglePlay}
          aria-pressed={isPlaying}
          aria-label={isLive ? 'Data terbaru diterima otomatis' : isPlaying ? 'Jeda contoh perubahan cuaca' : 'Putar contoh perubahan cuaca'}
          className={`w-11 h-11 rounded-xl flex items-center justify-center font-bold shadow-lg focus:outline-none focus:ring-2 focus:ring-cyan-400 ${
            isPlaying
              ? 'bg-amber-400 text-slate-950 shadow-amber-400/40 border border-amber-300'
              : 'bg-cyan-400 text-slate-950 shadow-cyan-400/40 hover:bg-cyan-300 border border-cyan-200'
          }`}
          title={isLive ? 'Data terbaru diterima otomatis' : isPlaying ? 'Jeda contoh perubahan cuaca' : 'Putar contoh perubahan cuaca'}
        >
          {isPlaying ? <Pause className="w-5 h-5 fill-current" /> : <Play className="w-5 h-5 fill-current ml-0.5" />}
        </motion.button>

        <motion.button
          whileHover={{ scale: shouldReduceMotion ? 1 : 1.05 }}
          whileTap={{ scale: shouldReduceMotion ? 1 : 0.95 }}
          disabled={isLive}
          onClick={() => {
            const nextIndex = (currentIndex + 1) % OFFSETS.length;
            onOffsetChange(OFFSETS[nextIndex]);
          }}
          className="w-11 h-11 rounded-xl bg-slate-800/80 border border-slate-600/50 text-slate-300 flex items-center justify-center hover:text-cyan-400 hover:border-cyan-500/50 transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-400"
          title="Langkah Selanjutnya"
          aria-label="Langkah Selanjutnya"
        >
          <FastForward className="w-4.5 h-4.5" />
        </motion.button>
      </div>

      {/* High-Contrast Time Horizon & Progress Bar */}
      <div className="flex-1 relative w-full flex flex-col justify-center px-1 py-1">
        {/* Track Line Background */}
        <div className="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
          <motion.div
            className="absolute top-0 left-0 w-full h-full bg-[#1f6056] rounded-full"
            initial={false}
            animate={{ scaleX: progressPercent / 100 }}
            style={{ transformOrigin: 'left center' }}
            transition={activeTransition}
          />
        </div>

        {/* Time Horizon Node Selector */}
        <div 
          role="tablist"
          aria-label={isLive ? 'Data terbaru' : 'Pilih waktu contoh'}
          className="flex justify-between items-center mt-2.5 relative"
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') {
              const prevIndex = (currentIndex - 1 + OFFSETS.length) % OFFSETS.length;
              onOffsetChange(OFFSETS[prevIndex]);
            } else if (e.key === 'ArrowRight') {
              const nextIndex = (currentIndex + 1) % OFFSETS.length;
              onOffsetChange(OFFSETS[nextIndex]);
            }
          }}
        >
          {OFFSETS.map((offset) => {
            const isActive = offset === currentOffset;
            const isNow = offset === 'Now';
            const isFuture = offset.startsWith('T+');

            return (
              <motion.button
                key={offset}
                role="tab"
                aria-selected={isActive}
                aria-label={`Offset waktu ${offset}`}
                whileHover={{ scale: shouldReduceMotion ? 1 : 1.05 }}
                whileTap={{ scale: shouldReduceMotion ? 1 : 0.95 }}
                disabled={isLive}
                onClick={() => onOffsetChange(offset)}
                className="relative flex flex-col items-center justify-center min-h-[44px] min-w-[44px] px-2 py-1 rounded-lg group focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                {/* Active Sliding Background Pill */}
                {isActive && (
                  <motion.div
                    layoutId="active-timeline-pill"
                    className={`absolute inset-0 rounded-lg shadow-md border ${
                      isNow
                        ? 'bg-amber-400/20 border-amber-400/80 shadow-amber-400/20'
                        : isFuture
                        ? 'bg-[#dbeae3] border-[#1f6056]'
                        : 'bg-[#dbeae3] border-[#1f6056]'
                    }`}
                    transition={activeTransition}
                  />
                )}

                {/* Node Indicator Dot */}
                <motion.div
                  animate={{
                    scale: isActive ? (shouldReduceMotion ? 1 : 1.25) : 1,
                  }}
                  transition={activeTransition}
                  className={`w-3.5 h-3.5 rounded-full z-10 border-2 ${
                    isActive
                      ? isNow
                        ? 'bg-amber-400 border-amber-200 shadow-[0_0_10px_rgba(251,191,36,0.9)]'
                        : isFuture
                        ? 'bg-[#1f6056] border-white'
                        : 'bg-[#1f6056] border-white'
                      : 'bg-slate-800 border-slate-600 group-hover:border-slate-400'
                  }`}
                />

                {/* Offset Label */}
                <span
                  className={`font-mono-data text-xs mt-1 z-10 font-semibold transition-colors ${
                    isActive
                      ? isNow
                        ? 'text-amber-300 font-bold'
                        : isFuture
                        ? 'text-[#1f6056] font-bold'
                        : 'text-[#1f6056] font-bold'
                      : 'text-slate-400 group-hover:text-slate-200'
                  }`}
                >
                  {offset}
                </span>
              </motion.button>
            );
          })}
        </div>
      </div>


      {/* Frame Time Info & WIB Badge */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <div className="hidden lg:block max-w-[180px] text-right">
          <div className="text-[11px] font-mono-data text-slate-400 leading-tight truncate">
            {dataStatus}
          </div>
        </div>

        <div className="flex items-center gap-1.5 font-mono-data text-xs font-bold text-cyan-300 px-3 py-1.5 bg-slate-800/90 border border-cyan-500/40 rounded-xl shadow-inner tracking-wider">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span>{isLive ? 'Terbaru' : currentOffset === 'Now' ? 'Sekarang' : currentOffset}</span>
        </div>
      </div>
    </div>
  );
};

export const TimelineScrubber = React.memo(TimelineScrubberComponent);
