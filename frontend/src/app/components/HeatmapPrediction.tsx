import { useState, useEffect, useRef } from 'react';
import { Play, Upload, Trash2, ImageIcon, Tag, Clock, Table as TableIcon, Eye, LoaderIcon } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue, SelectSeparator as Separator } from '@/app/components/ui/select';
import { Input } from '@/app/components/ui/input';
import { toast } from 'sonner';
import { useAuth } from './AuthContext';
import { SecureHeatmap, SecureImageModel } from './SecureImage';
import { useNavigate } from 'react-router-dom';
import { API_URL } from '@/app/App';

interface HeatmapSession {
  id: number;
  img_name: string;
  created_at: string;
}

export function HeatmapPrediction() {
  const { isCalibrated, user } = useAuth();
  const [sessionName, setSessionName] = useState("");
  const [selectValue, setSelectValue] = useState<string>("none");
  const [availableImages, setAvailableImages] = useState<{
    id: number; model_name: string
}[]>([]);
  const [isLoadingImages, setIsLoadingImages] = useState(false);
  const [selectImageId, setSelectImageId] = useState<number | null>(null);
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

  const fetchAvailableImages = async () => {
    setIsLoadingImages(true);
    try {
      const response = await fetch(`${API_URL}/model/all`, { // Adjust to your actual endpoint
        headers: { 
          'Authorization': `Bearer ${localStorage.getItem('access_token')}` 
        }
      });
      if (response.ok) {
        const data = await response.json();
        setAvailableImages(data);
      }
    } catch (err) {
      toast.error("Failed to load stimulus library.");
    } finally {
      setIsLoadingImages(false);
    }
  };

  useEffect(() => {
    fetchAvailableImages();
    fetchSessions();
  }, []);

  const handleSelectStimulus = (value: string) => {
    setSelectValue(value);
    if (value === "none") {
      setSelectImageId(null);
      return;
    }
    const model_id = Number(value);
    setSelectImageId(model_id); 

  };

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
      model_id: selectImageId,
      width: naturalWidth,
      height: naturalHeight,
      points: gazeHistory.current
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
          fetchAvailableImages();
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
              <SecureImageModel
                modelId={selectImageId!}
                ref={imageRef}
                className="w-full h-full object-cover opacity-50"
              ></SecureImageModel>

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
              <SecureHeatmap 
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
        {/* Select Model section*/}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 bg-white/5 border-white/10 p-10 flex flex-col items-center justify-center min-h-[400px]">
            <div className="w-full space-y-6">
              <div className="relative aspect-video rounded-xl overflow-hidden border border-white/10">
                {!selectImageId ? (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6">
                        <LoaderIcon className="w-10 h-10 text-cyan-500" />
                      </div>
                      <h3 className="text-xl text-white font-bold mb-2">Select Stimuli</h3>
                      <p className="text-gray-500 mb-8 max-w-xs mx-auto">Select the image you want to analyze.</p>
                    </div>
                  </div>
                )
                : (
                  <>
                    <SecureImageModel
                      modelId={selectImageId!}
                      className="w-full h-full object-cover opacity-50 grayscale"
                    ></SecureImageModel>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <ImageIcon className="w-12 h-12 text-white/20" />
                    </div>
                  </>
              )}
              </div>
              <div className="w-full space-y-4">
                <h3 className="text-lg text-white font-medium">Select Model from Library</h3>
                
                <Select value={selectValue} onValueChange={handleSelectStimulus} disabled={isLoadingImages}>
                  <SelectTrigger className="w-full h-12 bg-white/5 border-white/10 text-white rounded-xl">
                    <SelectValue placeholder={isLoadingImages ? "Loading images..." : "Select an image"} />
                  </SelectTrigger>
                  <SelectContent className="bg-slate-900 border-white/10 text-white">
                    <SelectItem value="none" className="text-white focus:bg-cyan-500/20 ">
                      <div className="p-2 text-sm text-white text-center">None / Clear Selection</div>
                    </SelectItem>
                    
                    <Separator className="my-1 bg-white/10" />
                    {availableImages.map((img) => (
                      <SelectItem key={img.id} value={String(img.id)} className="focus:bg-cyan-500/20 text-white">
                        <div className="p-2 text-sm text-white text-center">{img.model_name}</div>
                      </SelectItem>
                    ))}
                    {availableImages.length === 0 && !isLoadingImages && (
                      <div className="p-2 text-sm text-gray-500 text-center">No images found in library</div>
                    )}
                  </SelectContent>
                </Select>

              </div>
              {selectImageId && (
                <div className="space-y-4">
                  <div className="space-y-4">
                    <h3 className="text-lg text-white font-medium">Input Session Name</h3>
                    <div className="relative">
                      <Tag className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                      <Input 
                        placeholder="Session Name (e.g. Amazon Hero Banner)" 
                        className="pl-12 bg-white/5 border-white/10 h-12   text-white rounded-xl"
                        value={sessionName}
                        onChange={(e) => setSessionName(e.target.value)}
                      />
                    </div>
                  </div>
                  <Button onClick={startAnalysisSession} size="lg" className="w-full h-14 bg-cyan-500 hover:bg-cyan-600 text-white text-lg font-bold rounded-xl">
                    <Play className="w-5 h-5 mr-2" /> Launch Analysis
                  </Button>
                </div>
              )}
            </div>
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
                              <SecureHeatmap 
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