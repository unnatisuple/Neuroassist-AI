import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Layout from './components/layout/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import MRIAnalysis from './pages/MRIAnalysis';
import RiskAssessment from './pages/RiskAssessment';
import Reports from './pages/Reports';
import PatientHistory from './pages/PatientHistory';
import Chatbot from './pages/Chatbot';
import AdminPanel from './pages/AdminPanel';
import PendingVerification from './pages/PendingVerification';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-medical-navy">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400">Loading NeuroAssist AI...</p>
        </div>
      </div>
    );
  }

  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
}

function VerifiedRoute({ children }: { children: React.ReactNode }) {
  const { isVerified } = useAuth();

  if (!isVerified) {
    return <PendingVerification />;
  }

  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      {/* Public routes — Clinician Portal login */}
      <Route path="/login" element={<Login />} />

      {/* Protected routes — require authenticated doctor */}
      <Route
        path="/"
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route
          path="dashboard"
          element={<VerifiedRoute><Dashboard /></VerifiedRoute>}
        />
        <Route
          path="mri-analysis"
          element={<VerifiedRoute><MRIAnalysis /></VerifiedRoute>}
        />
        <Route
          path="risk-assessment"
          element={<VerifiedRoute><RiskAssessment /></VerifiedRoute>}
        />
        <Route
          path="reports"
          element={<VerifiedRoute><Reports /></VerifiedRoute>}
        />
        <Route
          path="patients"
          element={<VerifiedRoute><PatientHistory /></VerifiedRoute>}
        />
        <Route
          path="chatbot"
          element={<VerifiedRoute><Chatbot /></VerifiedRoute>}
        />
        <Route path="admin" element={<AdminPanel />} />
      </Route>

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/login" replace />} />
    </Routes>
  );
}
