import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { API_URL } from '@/app/App';

declare global {
  interface Window {
    webgazer: any;
  }
}

interface User {
  email: string;
  name: string;
  id: number;
  role: number;
}

interface AuthContextType {
  isAuthenticated: boolean;
  isCalibrated: boolean;
  isLoading: boolean;
  setIsCalibrated: (calibrated: boolean) => void;
  user: User | null;
  login: (email: string, password: string) => Promise<User | false>;
  logout: () => void;
}


const AuthContext = createContext<AuthContextType | undefined>(undefined);

const decodeToken = (token: string) => {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );

    return JSON.parse(jsonPayload);
  } catch (e) {
    return null;
  }
};

const shutdownWebGazer = async () => {
  if (typeof window !== 'undefined' && window.webgazer) {
    if (window.webgazer.isReady()) window.webgazer.end();
    
    const video = document.getElementById('webgazerVideoFeed') as HTMLMediaElement;
    if (video && video.srcObject) {
      const stream = video.srcObject as MediaStream;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      video.srcObject = null;
    }

    const webgazerElements = [
      'webgazerVideoFeed',
      'webgazerVideoCanvas',
      'webgazerFaceOverlay',
      'webgazerFaceFeedbackBox'
    ];
    
    webgazerElements.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.remove();
    });

    console.log("WebGazer fully dismantled.");
  }
};

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initializeAuth = async () => {
      const token = localStorage.getItem('access_token');
      
      if (token) {
        try {
          const response = await fetch(`${API_URL}/verify-token`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });

          if (response.ok) {
            const payload = decodeToken(token);
            setIsAuthenticated(true);
            setUser({
              id: payload.id,
              email: payload.sub,
              name: payload.sub.split('@')[0],
              role: payload.role
            });
          } else {
            logout(); 
          }
        } catch (error) {
          console.error("Auth initialization failed", error);
          logout();
        }
      }
      setIsLoading(false)
    };

    initializeAuth();
  }, []);

  const login = async (email: string, password: string): Promise<User | false> => {
    try {
      const response = await fetch(`${API_URL}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        
        localStorage.setItem('access_token', data.access_token);
        
        const payload = decodeToken(data.access_token);
        
        if (payload) {
          const userData = {
            id: payload.id,
            email: payload.sub,
            name: payload.sub.split('@')[0],
            role: payload.role,
          };

          setIsAuthenticated(true);
          setUser(userData);
          setIsCalibrated(false);
          return userData;
        }
      }
      return false;
    } catch (error) {
      console.error("Auth Error:", error);
      return false;
    }
  };

  const logout = async () => {
    if (typeof window !== 'undefined' && window.webgazer) await shutdownWebGazer();
    const tempUser = user;
    setIsAuthenticated(false);
    setIsCalibrated(false);
    setUser(null);
    localStorage.removeItem('access_token');
    if (tempUser?.role == 1) window.location.assign('/login');
  };

  return (
    <AuthContext.Provider value={{ isAuthenticated, isCalibrated, isLoading, setIsCalibrated, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
