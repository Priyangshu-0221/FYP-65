import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-[#ffd700] focus:ring-offset-2 focus:ring-offset-black",
  {
    variants: {
      variant: {
        default: "border-white/10 bg-[#ffd700] text-black hover:bg-[#ffe14d]",
        secondary:
          "border-white/10 bg-white/[0.06] text-white hover:bg-white/[0.1]",
        destructive:
          "border-transparent bg-red-600 text-white hover:bg-red-700",
        outline: "border-white/15 text-white hover:bg-white/[0.06]",
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
