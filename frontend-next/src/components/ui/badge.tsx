import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-all duration-200 ease-out hover:-translate-y-0.5 focus:outline-none focus:ring-2 focus:ring-[#2563eb] focus:ring-offset-2 focus:ring-offset-white",
  {
    variants: {
      variant: {
        default:
          "border-[#2563eb]/20 bg-[#2563eb] text-white hover:bg-[#1d4ed8]",
        secondary:
          "border-[#10b981]/20 bg-[#ecfdf5] text-[#065f46] hover:bg-[#d1fae5]",
        destructive:
          "border-transparent bg-red-600 text-white hover:bg-red-700",
        outline: "border-[#cbd5e1] bg-white text-[#0f172a] hover:bg-[#eff6ff]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends
    React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
