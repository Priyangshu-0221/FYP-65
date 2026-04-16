"use client";

import { CheckCircle2 } from "lucide-react";

interface SuccessOverlayProps {
  isVisible: boolean;
}

export function SuccessOverlay({ isVisible }: SuccessOverlayProps) {
  if (!isVisible) {
    return null;
  }

  const confettiDots = Array.from({ length: 12 }, (_, index) => index);

  return (
    <div
      aria-live="assertive"
      aria-busy="false"
      className="fixed inset-0 z-[210] flex items-center justify-center bg-black/92 px-3 backdrop-blur-sm sm:px-4"
    >
      <div className="success-overlay-panel relative flex w-full max-w-[min(92vw,34rem)] flex-col items-center rounded-[1.5rem] border border-white/10 bg-black/95 px-5 py-8 text-center text-white shadow-2xl sm:rounded-[2rem] sm:px-8 sm:py-10">
        <div className="success-ring absolute inset-0 rounded-[2rem]" />
        <div className="success-ring success-ring-delayed absolute inset-0 rounded-[2rem]" />

        <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-[2rem]">
          {confettiDots.map((index) => (
            <span
              key={index}
              className="success-confetti"
              style={{
                left: `${8 + ((index * 13) % 84)}%`,
                top: `${12 + ((index * 17) % 20)}%`,
                animationDelay: `${(index % 6) * 0.12}s`,
              }}
            />
          ))}
        </div>

        <div className="success-checkmark relative flex h-[clamp(5rem,20vw,7rem)] w-[clamp(5rem,20vw,7rem)] items-center justify-center rounded-full border border-[#ffd700] bg-[#ffd700]/10 text-[#ffd700]">
          <CheckCircle2 className="success-check-icon h-[clamp(3rem,12vw,4rem)] w-[clamp(3rem,12vw,4rem)]" />
        </div>

        <h2 className="mt-6 text-[clamp(1.5rem,5vw,2.25rem)] font-bold tracking-tight text-white">
          Recommendations Ready
        </h2>
        <p className="mt-2 max-w-sm text-[clamp(0.85rem,2.2vw,0.95rem)] text-white/72">
          Your best internship matches have been found and loaded successfully.
        </p>
      </div>
    </div>
  );
}
