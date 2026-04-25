import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-semibold ring-offset-white transition-all duration-200 ease-out hover:-translate-y-0.5 hover:shadow-lg active:translate-y-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#2563eb] focus-visible:ring-offset-2 focus-visible:ring-offset-white disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#2563eb] text-white hover:bg-[#1d4ed8]",
        destructive: "bg-red-600 text-white hover:bg-red-500",
        outline:
          "border border-[#10b981]/35 bg-white text-[#0f172a] hover:border-[#10b981]/55 hover:bg-[#ecfdf5]",
        secondary: "bg-[#10b981] text-white hover:bg-[#059669]",
        ghost: "bg-transparent text-[#0f172a] hover:bg-[#eff6ff]",
        link: "text-[#2563eb] underline-offset-4 hover:text-[#059669]",
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
