"use client";

import { useEffect, useRef, useState } from "react";
import { MousePointer2 } from "lucide-react";

const interactiveSelector =
  'a, button, [role="button"], input, select, textarea, summary, label, [data-cursor="interactive"]';

export function GlobalCursor() {
  const ringRef = useRef<HTMLDivElement>(null);
  const dotRef = useRef<HTMLDivElement>(null);
  const positionRef = useRef({ x: -9999, y: -9999 });
  const visibleRef = useRef(false);
  const interactiveRef = useRef(false);
  const pressedRef = useRef(false);
  const [isInteractive, setIsInteractive] = useState(false);
  const [isPressed, setIsPressed] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(hover: hover) and (pointer: fine)");

    if (!media.matches) {
      return undefined;
    }

    const renderCursor = () => {
      const { x, y } = positionRef.current;
      const scale = pressedRef.current
        ? 0.82
        : interactiveRef.current
          ? 1.28
          : 1;
      const dotScale = pressedRef.current
        ? 0.8
        : interactiveRef.current
          ? 1.18
          : 1;
      const opacity = visibleRef.current ? "1" : "0";

      if (ringRef.current) {
        ringRef.current.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) scale(${scale})`;
        ringRef.current.style.opacity = opacity;
      }

      if (dotRef.current) {
        dotRef.current.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%) scale(${dotScale})`;
        dotRef.current.style.opacity = opacity;
      }
    };

    const handlePointerMove = (event: PointerEvent) => {
      positionRef.current = { x: event.clientX, y: event.clientY };
      const nextInteractive =
        event.target instanceof Element &&
        Boolean(event.target.closest(interactiveSelector));

      if (nextInteractive !== interactiveRef.current) {
        interactiveRef.current = nextInteractive;
        setIsInteractive(nextInteractive);
      }

      visibleRef.current = true;
      renderCursor();
    };

    const handlePointerDown = () => {
      pressedRef.current = true;
      setIsPressed(true);
      renderCursor();
    };

    const handlePointerUp = () => {
      pressedRef.current = false;
      setIsPressed(false);
      renderCursor();
    };

    const handleLeave = () => {
      visibleRef.current = false;
      if (ringRef.current) {
        ringRef.current.style.opacity = "0";
      }
      if (dotRef.current) {
        dotRef.current.style.opacity = "0";
      }
    };

    const handleBlur = () => {
      visibleRef.current = false;
    };

    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("pointerup", handlePointerUp);
    document.addEventListener("mouseleave", handleLeave);
    window.addEventListener("blur", handleBlur);

    return () => {
      document.removeEventListener("pointermove", handlePointerMove);
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("pointerup", handlePointerUp);
      document.removeEventListener("mouseleave", handleLeave);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  return (
    <div
      aria-hidden="true"
      className={`custom-cursor${isInteractive ? " custom-cursor--interactive" : ""}${isPressed ? " custom-cursor--pressed" : ""}`}
    >
      <div ref={ringRef} className="custom-cursor__ring">
        <MousePointer2 className="custom-cursor__pointer-icon" />
      </div>
      <div ref={dotRef} className="custom-cursor__dot" />
    </div>
  );
}
