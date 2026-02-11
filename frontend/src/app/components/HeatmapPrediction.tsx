import { useState, useEffect, useRef } from 'react';
import { Play, Upload, Trash2, ImageIcon, Tag, Clock, Table as TableIcon, Eye } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Input } from '@/app/components/ui/input';
import { toast } from 'sonner';
import { useAuth } from './AuthContext';
import { SecureImage } from './SecureImage';
import { Navigate, useNavigate } from 'react-router-dom';
import { API_URL } from '@/app/App';

interface HeatmapSession {
  id: number;
  img_name: string;
  created_at: string;
  user_id: number;
  base64_data: string;
}

export function HeatmapPrediction() {
  const { isCalibrated, user } = useAuth();
  const [sessionName, setSessionName] = useState("");
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const navigate = useNavigate();
  
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [isRecording, setIsRecording] = useState(false);

  const [isShowViewHeatmap, setIsShowViewHeatmap] = useState(false);
  const [viewHeatmapSessionId, setViewHeatmapSessionId] = useState<number | null>(null);
  const viewModalRef = useRef<HTMLDivElement>(null);

  const gazeHistory = useRef<{ x: number, y: number }[]>([]);
  const imageRef = useRef<HTMLImageElement>(null);

  const [sessions, setSessions] = useState<HeatmapSession[]>([]);
  const [isTableLoading, setIsTableLoading] = useState(false);

  if (!isCalibrated) {
    setTimeout(() => {
       navigate('/live-tracking');
    }, 5000);
    return (
      <div className="flex items-center justify-center h-[70vh]">
        <div className="text-center">
          <Clock className="w-16 h-16 text-gray-500 mx-auto mb-4 animate-spin-slow" />
          <h2 className="text-2xl text-white font-bold mb-2">Calibration Required</h2>
          <p className="text-gray-400 max-w-md mx-auto">
            Please complete the calibration process in the Live Tracking section before using the Heatmap Prediction feature.
          </p>
        </div>
      </div>
      
    );
  }

  useEffect(() => {
    const wg = (window as any).webgazer;
    if (isCalibrated && wg.isReady()) {
      wg.setGazeListener((data: any) => {
        if (data && isRecording && imageRef.current) {
          const imageEl = imageRef.current;
          const rect = imageEl.getBoundingClientRect();
          
          if (rect.width === 0 || rect.height === 0) return;

          const relX = data.x - rect.left;
          const relY = data.y - rect.top;

          const scaleX = imageEl.naturalWidth / rect.width;
          const scaleY = imageEl.naturalHeight / rect.height;

          const actualX = Math.round(relX * scaleX);
          const actualY = Math.round(relY * scaleY);

          if (actualX >= 0 && actualX <= imageEl.naturalWidth && 
              actualY >= 0 && actualY <= imageEl.naturalHeight) {
            gazeHistory.current.push({ x: actualX, y: actualY });
          }
        }
      });
    }
  }, [isCalibrated, isRecording]);

  const fetchSessions = async () => {
    setIsTableLoading(true);
    try {
      const response = await fetch(`${API_URL}/heatmap/get_by_user/${user?.id}`, {
        headers: { 
          'Authorization': `Bearer ${localStorage.getItem('access_token')}` 
        }
      });
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (err) {
      toast.error("Could not load previous sessions.");
    } finally {
      setIsTableLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const checkSession = async (img_name:string) => {
    try {
      const response = await fetch(`${API_URL}/heatmap/check/${user?.id}/${img_name}`, {
        headers: { 
          'Authorization': `Bearer ${localStorage.getItem('access_token')}` 
        }
      });
      if (response.ok) {
        const data = await response.json();
        return data.status
      }
    } catch (err) {
      return false
    }
  };

  const startAnalysisSession = async () => {
    if ((await checkSession(sessionName))) return toast.error("Session name already used.");
    if (!sessionName) return toast.error("Please name the session first.");
    if (!isCalibrated) return toast.error("Calibration required!");

    setIsAnalyzing(true);
    setCountdown(3);

    const prepInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(prepInterval);
          startActualRecording();
          return 10;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const startActualRecording = () => {
    const wg = (window as any).webgazer;
    gazeHistory.current = [];
    setIsRecording(true);
    wg.resume();
    wg.showPredictionPoints(true);

    const recordInterval = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(recordInterval);
          finishAndSave();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const finishAndSave = async () => {
    const wg = (window as any).webgazer;
    setIsRecording(false);
    setIsAnalyzing(false);
    wg.pause();
    wg.showPredictionPoints(false);

    if (!imageRef.current) {
    console.error("Image reference lost before save.");
      return;
    }

    const { naturalWidth, naturalHeight } = imageRef.current;


    const payload = {
      name: sessionName,
      user_id: user?.id,
      width: naturalWidth,
      height: naturalHeight,
      points: gazeHistory.current,
      base64_image: uploadedImage 
    };

    toast.promise(
      fetch(`${API_URL}/heatmap/upload`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify(payload)
      }),
      {
        loading: 'Saving gaze data...',
        success: ()=>{
          fetchSessions();
          return 'Session saved successfully!'
        },
        error: 'Failed to save session data.'
      }
    );
  };

  const handleDeleteSession = async (sessionId: number) => {
    toast.promise(
      fetch(`${API_URL}/heatmap/delete/${sessionId}`, {
        method: 'DELETE',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        }
      }),
      {
        loading: 'Deleting session...',
        success: ()=>{
          fetchSessions();
          sessionId === viewHeatmapSessionId && setIsShowViewHeatmap(false);
          setViewHeatmapSessionId(null);
          return 'Session deleted successfully!'
        },
        error: 'Failed to delete session.'
      }
    )
  }

  useEffect(() => {
    if (isShowViewHeatmap) {
      document.body.style.overflow = 'hidden';
      viewModalRef.current?.focus();
    } else {
      document.body.style.overflow = 'unset';
    }
  }, [isShowViewHeatmap]);

  return (
    <>
      <> {/* Use a Fragment as the absolute root */}
        {/* FULLSCREEN ANALYSIS OVERLAY */}
        {isAnalyzing && (
          <div className="fixed inset-0 z-[10000] bg-black flex flex-col items-center justify-center">
            
            {/* PHASE 1: PREPARATION BLUR/OVERLAY */}
            {!isRecording && (
              <div className="absolute inset-0 z-[10001] bg-black/90 flex flex-col items-center justify-center text-white backdrop-blur-sm">
                <h2 className="text-5xl font-bold mb-6 text-cyan-400">Prepare to Focus</h2>
                <div className="text-8xl font-mono border-8 border-cyan-500 w-32 h-32 flex items-center justify-center rounded-full animate-pulse">
                  {countdown}
                </div>
              </div>
            )}

            {/* PHASE 2: THE IMAGE (FULLSCREEN) */}
            <div className="relative w-screen h-screen flex items-center justify-center bg-black">
              <img 
                ref={imageRef} 
                src={uploadedImage!} 
                alt="Stimuli" 
                className="w-full h-full object-contain"
              />

              {/* THE COUNTDOWN INDICATOR (Bottom of Container) */}
              <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center space-y-2 pointer-events-none">
                <div className="bg-black/60 backdrop-blur-xl border border-white/20 px-8 py-3 rounded-full flex items-center space-x-4 shadow-2xl">
                  <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                  <span className="text-white font-mono text-3xl tracking-tighter">
                    {isRecording ? `Recording: ${countdown}s` : `Wait: ${countdown}s`}
                  </span>
                </div>
                {isRecording && (
                  <div className="w-64 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-cyan-500 transition-all duration-1000 ease-linear"
                      style={{ width: `${(countdown / 10) * 100}%` }}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {isShowViewHeatmap && (
          <div className="fixed inset-0 z-[10000] bg-black flex flex-col items-center justify-center">
            <div className="relative w-screen h-screen flex items-center justify-center bg-black"
            onKeyDown={(e) => {
              if (e.key === 'Escape') {
                setIsShowViewHeatmap(false);
                setViewHeatmapSessionId(null);
              }
            }}
            tabIndex={0}
            ref={viewModalRef}
            >
              <SecureImage 
                sessionId={viewHeatmapSessionId!} 
                className="w-full h-full object-cover opacity-80" 
              />
            </div>
            <div className="absolute top-6 right-6">
              <Button onClick={() => {
                setIsShowViewHeatmap(false);
                setViewHeatmapSessionId(null);
                }} variant="ghost" className="text-white border border-black bg-red-500/15 hover:bg-red-500/25 rounded-lg px-4 py-2"><Trash2 className="w-4 h-4 mr-2" /> Close</Button>
            </div>
          </div>
        )}
      </>

      <div className="space-y-6">
        {/* DASHBOARD UI */}
        <div className="flex items-center justify-between">
          <h2 className="text-3xl text-white font-bold">Heatmap Prediction</h2>
          {uploadedImage && <Button onClick={() => setUploadedImage(null)} variant="ghost" className="text-red-400"><Trash2 className="w-4 h-4 mr-2" /> Reset Image</Button>}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 bg-white/5 border-white/10 p-10 flex flex-col items-center justify-center min-h-[400px]">
            {!uploadedImage ? (
              <div className="text-center">
                <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Upload className="w-10 h-10 text-cyan-500" />
                </div>
                <h3 className="text-xl text-white font-bold mb-2">Upload Stimuli</h3>
                <p className="text-gray-500 mb-8 max-w-xs mx-auto">Upload the image or UI mockup you want to analyze.</p>
                <label className="bg-cyan-500 hover:bg-cyan-600 text-white px-8 py-3 rounded-xl cursor-pointer transition-all shadow-lg shadow-cyan-500/20">
                  Choose File
                  <input type="file" className="hidden" onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) {

                      if (!file.type.startsWith('image/')) {
                        toast.error("Invalid file type. Please upload an image (PNG, JPG, etc.).");
                        e.target.value = "";
                        return;
                      }

                      if (file.size > 10 * 1024 * 1024) {
                        toast.error("File is too large. Please upload an image under 10MB.");
                        return;
                      }

                      const reader = new FileReader();
                      reader.onload = () => setUploadedImage(reader.result as string);
                      reader.readAsDataURL(file);
                    }
                  }} 
                  accept="image/png, image/jpeg, image/jpg, image/webp"
                  />
                </label>
              </div>
            ) : (
              <div className="w-full space-y-6">
                <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10">
                  <img src={uploadedImage} className="w-full h-full object-cover opacity-50 grayscale" alt="Preview" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <ImageIcon className="w-12 h-12 text-white/20" />
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="relative">
                    <Tag className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <Input 
                      placeholder="Session Name (e.g. Amazon Hero Banner)" 
                      className="pl-12 bg-white/5 border-white/10 h-14 text-white rounded-xl"
                      value={sessionName}
                      onChange={(e) => setSessionName(e.target.value)}
                    />
                  </div>
                  <Button onClick={startAnalysisSession} size="lg" className="w-full h-14 bg-cyan-500 hover:bg-cyan-600 text-white text-lg font-bold rounded-xl">
                    <Play className="w-5 h-5 mr-2" /> Launch Analysis
                  </Button>
                </div>
              </div>
            )}
          </Card>

          {/* Info Sidebar */}
          <Card className="bg-white/5 border-white/10 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl text-white font-bold flex items-center">
                <TableIcon className="w-5 h-5 mr-2 text-cyan-400" /> Heatmap Management
              </h3>
              {/* <Button variant="outline" className="border-white/10 text-gray-400 hover:text-white">
                <Download className="w-4 h-4 mr-2" /> Export CSV
              </Button> */}
            </div>
          
            <div className="overflow-x-auto rounded-xl border border-white/5 bg-slate-900/50">
              <table className="w-full text-left text-sm text-gray-400">
                <thead className="bg-white/5 text-white uppercase text-[10px] tracking-wider font-bold">
                  <tr>
                    <th className="px-6 py-4">Session Name</th>
                    <th className="px-6 py-4">Date Created</th>
                    <th className="px-6 py-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {isTableLoading ? (
                    <tr>
                      <td colSpan={4} className="py-10 text-center text-gray-500 animate-pulse">
                        Fetching analysis history...
                      </td>
                    </tr>
                  ) : sessions.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="py-10 text-center text-gray-500">
                        No sessions found. Launch an analysis to see data here.
                      </td>
                    </tr>
                  ) : (
                    sessions.map((session) => (
                      <tr key={session.id} className="hover:bg-white/[0.02] transition-colors group">
                        <td className="px-6 py-4">
                          <div className="flex items-center space-x-3">
                            <div className="w-12 h-8 bg-slate-800 rounded border border-white/10 overflow-hidden">
                              <SecureImage 
                                  sessionId={session.id} 
                                  className="w-full h-full object-cover opacity-80" 
                                />
                            </div>
                            <span className="text-white font-medium">{session.img_name}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 italic">
                          {new Date(session.created_at).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 text-right space-x-2">
                          {/* We pass the session ID to these functions to "re-render" the specific heatmap */}
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            onClick={() => {
                              setIsShowViewHeatmap(true);
                              setViewHeatmapSessionId(session.id);
                            }}
                            className="hover:text-cyan-400"
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="sm" 
                            onClick={() => handleDeleteSession(session.id)}
                            className="hover:text-red-400"
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}