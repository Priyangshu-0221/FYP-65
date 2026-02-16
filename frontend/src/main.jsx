import React from "react";
import ReactDOM from "react-dom/client";
import { ChakraProvider } from "@chakra-ui/react";
import App from "./App.jsx";
import "./index.css";

// Add error boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "20px", color: "red" }}>
          <h1>Something went wrong!</h1>
          <pre>{this.state.error?.toString()}</pre>
          <p>Check the console for more details.</p>
        </div>
      );
    }

    return this.props.children;
  }
}

try {
  console.log("Starting React app...");
  const root = document.getElementById("root");
  console.log("Root element:", root);
  
  ReactDOM.createRoot(root).render(
    <React.StrictMode>
      <ErrorBoundary>
        <ChakraProvider>
          <App />
        </ChakraProvider>
      </ErrorBoundary>
    </React.StrictMode>
  );
  console.log("React app rendered successfully!");
} catch (error) {
  console.error("Failed to render React app:", error);
  document.body.innerHTML = `
    <div style="padding: 20px; color: red;">
      <h1>Failed to load application</h1>
      <pre>${error.toString()}</pre>
    </div>
  `;
}
