import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-semibold ring-offset-black transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#ffd700] focus-visible:ring-offset-2 focus-visible:ring-offset-black disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#ffd700] text-black hover:bg-[#e6c200]",
        destructive: "bg-red-600 text-white hover:bg-red-500",
        outline:
          "border border-white/15 bg-white/[0.04] text-white hover:border-white/25 hover:bg-white/[0.08]",
        secondary: "bg-white/[0.08] text-white hover:bg-white/[0.12]",
        ghost: "bg-transparent text-white hover:bg-white/[0.08]",
        link: "text-[#ffd700] underline-offset-4 hover:text-[#ffe978]",
      },
      size: {
        default: "h-9 px-3 py-2 text-sm sm:h-10 sm:px-4",
        sm: "h-8 rounded-md px-2.5 text-xs sm:h-9 sm:px-3",
        lg: "h-10 rounded-md px-5 text-sm sm:h-11 sm:px-8",
        icon: "h-9 w-9 sm:h-10 sm:w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends
    React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, children, ...props }, ref) => {
    const mergedClassName = cn(buttonVariants({ variant, size, className }));

    if (asChild) {
      const onlyChild = React.Children.only(children);
      if (React.isValidElement(onlyChild)) {
        const childProps = onlyChild.props as { className?: string };
        return React.cloneElement(
          onlyChild as React.ReactElement<{ className?: string }>,
          {
            className: cn(mergedClassName, childProps.className),
          },
        );
      }
      return null;
    }

    return (
      <button className={mergedClassName} ref={ref} {...props}>
        {children}
      </button>
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
