import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-semibold ring-offset-white transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1d3b72] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "bg-[#1d3b72] text-white hover:bg-[#27549f]",
        destructive: "bg-red-600 text-white hover:bg-red-700",
        outline:
          "border border-[#c8d4e9] bg-white text-[#1d3b72] hover:bg-[#f1f5fb]",
        secondary: "bg-[#eef3fb] text-[#1d3b72] hover:bg-[#e4ecf9]",
        ghost: "hover:bg-[#eef3fb] hover:text-[#1d3b72]",
        link: "text-[#1d3b72] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3 text-xs",
        lg: "h-11 rounded-md px-8",
        icon: "h-10 w-10",
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
