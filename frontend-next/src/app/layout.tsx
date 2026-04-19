import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "@/app/globals.css";
import "react-toastify/dist/ReactToastify.css";
import { ToastContainer } from "react-toastify";
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
        <div className="flex min-h-screen flex-col bg-black text-white">
          <Header />
          {/* <Sidebar /> */}
          <div className="mx-auto flex w-full max-w-[1400px] flex-1 flex-col px-3 pb-8 pt-5 sm:px-4 md:px-8 lg:pb-12 lg:pt-8">
            <main className="flex-1">{children}</main>
          </div>
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
          theme="dark"
        />
      </body>
    </html>
  );
}
