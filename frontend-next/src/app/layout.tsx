import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Toaster } from "sonner";
import "@/app/globals.css";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";

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
      <body className={inter.className}>
        <div className="flex min-h-screen flex-col">
          <Header />
          <div className="mx-auto flex w-full max-w-[1400px] flex-1">
            <Sidebar />
            <main className="flex-1">{children}</main>
          </div>
        </div>
        <Toaster />
      </body>
    </html>
  );
}
