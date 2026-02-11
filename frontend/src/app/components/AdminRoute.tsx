import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '@/app/components/AuthContext';

export const AdminRoute = () => {
  const { user, isLoading } = useAuth();

  if (isLoading) return <div>Checking permissions...</div>;

  return user?.role == 2 ? <Outlet /> : <Navigate to="/login" replace />;
};