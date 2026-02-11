import { Link, Outlet, useLocation } from 'react-router-dom';
import { Eye, Activity, LogOut } from 'lucide-react';
import { useAuth } from '@/app/components/AuthContext';
import { Button } from '@/app/components/ui/button';
import { toast } from 'sonner';

export function Layout() {
  const { isCalibrated } = useAuth();
  const location = useLocation();
  const { user, logout } = useAuth();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const gazer = (window as any).webgazer;
  if (gazer) {
    gazer.params.faceMeshSolutionPath  = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh';
  }
  // await gazer.setRegression('ridge');

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <nav className="border-b border-white/10 bg-black/20 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-2">
              <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-lg flex items-center justify-center">
                <Eye className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl text-white">EyeTrack AI</h1>
                <p className="text-xs text-gray-400">Advanced Eye Tracking & Heatmap Analysis</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex space-x-1">
                {user?.role == 2 ?
                  (<Link to="/admin" className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all ${
                        isActive('/admin')
                          ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/50'
                          : 'text-gray-300 hover:bg-white/10'
                      }`}>
                    Admin Panel
                  </Link>)
                 : (
                    <>
                    <Link
                      to="/live-tracking"
                      className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all ${
                        isActive('/live-tracking')
                          ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/50'
                          : 'text-gray-300 hover:bg-white/10'
                      }`}
                    >
                      <Activity className="w-4 h-4" />
                      <span>Live Tracking</span>
                    </Link>
                    {
                      isCalibrated ?
                      (<Link
                        to="/heatmap-prediction"
                        className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all ${
                          isActive('/heatmap-prediction')
                            ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/50'
                            : 'text-gray-300 hover:bg-white/10'
                        }`}
                      >
                        <Eye className="w-4 h-4" />
                        <span>Heatmap Prediction</span>
                      </Link>
                      ) :
                      (<Button
                        onClick={() => toast.error("Silahkan selesaikan kalibrasi di Live Tracking terlebih dahulu!")}
                        className="px-4 py-2 rounded-lg flex items-center space-x-2 text-gray-500 cursor-not-allowed grayscale"
                      >
                        <Eye className="w-4 h-4" />
                        <span>Heatmap Prediction (Locked)</span>
                      </Button>)
                    }
                    </>
                  )
                }
              </div>

              <div className="flex items-center space-x-3 border-l border-white/10 pl-4">
                <div className="text-right">
                  <p className="text-white text-sm">{user?.name}</p>
                  <p className="text-gray-400 text-xs">{user?.email}</p>
                </div>
                <Button
                  onClick={logout}
                  variant="outline"
                  size="sm"
                  className="bg-white/10 border-white/20 text-white hover:bg-red-500/20 hover:border-red-500/50"
                >
                  <LogOut className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Outlet />
      </main>
    </div>
  );
}