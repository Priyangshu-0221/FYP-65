import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "@/app/globals.css";
import "react-toastify/dist/ReactToastify.css";
import { ToastContainer } from "react-toastify";
import { Header } from "@/components/layout/Header";
import { GlobalCursor } from "@/components/layout/GlobalCursor";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "AI Powered Career Guidance",
  description:
    "Resume analyser and internship recommendation system for university students",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} bg-white text-slate-900`}>
        <div className="relative isolate min-h-screen overflow-hidden">
          <div aria-hidden="true" className="brand-watermark">
            <div className="brand-watermark__logo" />
            <div className="brand-watermark__tint" />
            <div className="brand-watermark__glass" />
          </div>

          <div className="relative z-10 flex min-h-screen flex-col">
            <Header />
            <div className="mx-auto flex w-full max-w-[1400px] flex-1 flex-col px-3 pb-8 pt-5 sm:px-4 md:px-8 lg:pb-12 lg:pt-8">
              <main className="flex-1">{children}</main>
            </div>
          </div>
          <GlobalCursor />
        </div>
        <ToastContainer
          position="top-right"
          autoClose={3500}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="light"
        />
      </body>
    </html>
  );
}
