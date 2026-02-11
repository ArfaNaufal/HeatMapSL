import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from '@/app/components/AuthContext';
import { ProtectedRoute } from '@/app/components/ProtectedRoute';
import { Layout } from '@/app/components/Layout';
import { Login } from '@/app/components/Login';
import { LiveTracking } from '@/app/components/LiveTracking';
import { HeatmapPrediction } from '@/app/components/HeatmapPrediction';
import { Toaster } from '@/app/components/ui/sonner';
import { AdminRoute } from '@/app/components/AdminRoute';
import { AdminPanel } from '@/app/components/AdminPanel';
import { useAuth } from '@/app/components/AuthContext';

const HomeRedirect = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) return null;

  return user?.role == 2  
    ? <Navigate to="/admin" replace /> 
    : <Navigate to="/live-tracking" replace />;
};

export const API_URL = "/api";

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            {/* Dynamic Home Base */}
            <Route index element={<HomeRedirect />} />

            {/* Admin Only Section */}
            <Route element={<AdminRoute />}>
              <Route path="admin" element={<AdminPanel />} />
            </Route>

            {/* Standard User Routes */}
            <Route path="live-tracking" element={<LiveTracking />} />
            <Route path="heatmap-prediction" element={<HeatmapPrediction />} />
          </Route>
        </Routes>
        <Toaster />
      </BrowserRouter>
    </AuthProvider>
  );
}