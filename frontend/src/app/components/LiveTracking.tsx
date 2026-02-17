import { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, CheckCircle2 } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { toast } from 'sonner'
import { useAuth } from './AuthContext';

const CALIBRATION_POINTS = [
  { x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 },
  { x: 10, y: 50 }, { x: 50, y: 50 }, { x: 90, y: 50 },
  { x: 10, y: 90 }, { x: 50, y: 90 }, { x: 90, y: 90 }
];

export function LiveTracking() {
  const { isCalibrated, setIsCalibrated } = useAuth();
  const [isTracking, setIsTracking] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [clickCounts, setClickCounts] = useState<Record<number, number>>({});
  const [gazePoint, setGazePoint] = useState({ x: 50, y: 50 });
  const [allPointsDone, setAllPointsDone] = useState(false);
  const CLICKS_REQUIRED = 9;

  const initWebGazer = async () => {
    const wg = (window as any).webgazer;
    if (!wg.isReady()) {
      await wg.setGazeListener((data: any) => {
        if (data) {
          const xPct = (data.x / window.innerWidth) * 100;
          const yPct = (data.y / window.innerHeight) * 100;
          setGazePoint({ x: xPct, y: yPct });
        }
      }).begin();
      wg.showVideoPreview(true).showPredictionPoints(true);
    }
  };

  useEffect(() => {
    if (allPointsDone) {
      const wg = (window as any).webgazer;
      setIsCalibrating(false);
      setIsCalibrated(true);
      setAllPointsDone(false);
      
      if (wg) {
        wg.showPredictionPoints(false);
        wg.showVideoPreview(false);
        wg.showVideo(false);
        wg.showFaceFeedbackBox(false);
        wg.showFaceOverlay(false);
        wg.pause();
      }
      toast.success("Calibration Successful!");
    }
  }, [allPointsDone, setIsCalibrated]);

  const startCalibration = async () => {
    const wg = (window as any).webgazer;
    if (isCalibrated) await wg.resume();
    else await initWebGazer();
    
    setIsCalibrating(true);
    setIsCalibrated(false);
    setClickCounts({});

    const webgazerVideoContainer = document.getElementById('webgazerVideoContainer') as HTMLElement; 
    if (webgazerVideoContainer.parentElement !== document.body) {
        document.body.appendChild(webgazerVideoContainer);
        webgazerVideoContainer.style.position = 'fixed';
    }
    
    wg.showPredictionPoints(true);
    wg.showVideoPreview(true);
    wg.showVideo(true);
    wg.showFaceFeedbackBox(true);
    wg.showFaceOverlay(true);
  };

  const handleCalibrationClick = (index: number) => {
    const wg = (window as any).webgazer;
    const point = CALIBRATION_POINTS[index];
    
    wg.recordScreenPosition(
      window.innerWidth * (point.x / 100),
      window.innerHeight * (point.y / 100),
      'click'
    );

    setClickCounts(prev => {
      const newCounts = { ...prev, [index]: (prev[index] || 0) + 1 };
      
      const completedPoints = Object.values(newCounts).filter(count => count >= CLICKS_REQUIRED).length;

      if (completedPoints === CALIBRATION_POINTS.length) {
        setAllPointsDone(true);

      }
      
      return newCounts;
    });
  };

  const toggleTracking = async () => {
    const wg = (window as any).webgazer;
    const webgazerVideoContainer = document.getElementById('webgazerVideoContainer') as HTMLElement; 
    if (!isTracking) {
      await initWebGazer();
      
      if (webgazerVideoContainer.parentElement === document.body) {
        const cameraContainer = document.getElementById('cameraContainer') as HTMLElement;
        if (cameraContainer)
          cameraContainer.appendChild(webgazerVideoContainer);
          webgazerVideoContainer.style.position = 'relative';
      }

      wg.resume();
      setIsTracking(true);
      wg.showPredictionPoints(true);
      wg.showVideoPreview(true);
      wg.showVideo(true);
      wg.showFaceFeedbackBox(true);
      wg.showFaceOverlay(true);

    } else {
      if (webgazerVideoContainer.parentElement !== document.body) {
          document.body.appendChild(webgazerVideoContainer);
          webgazerVideoContainer.style.position = 'fixed';
      }

      wg.pause();
      setIsTracking(false);
      wg.showPredictionPoints(false);
      wg.showVideoPreview(false);
      wg.showVideo(false);
      wg.showFaceFeedbackBox(false);
      wg.showFaceOverlay(false);
    }
  };

  useEffect(() => {
    return () => {
      const wg = (window as any).webgazer;
      if (wg) {
        wg.pause();
        wg.showVideoPreview(false);
        wg.showPredictionPoints(false);
        const webgazerVideoContainer = document.getElementById('webgazerVideoContainer') as HTMLElement; 
        if (webgazerVideoContainer && webgazerVideoContainer.parentElement !== document.body){
          document.body.appendChild(webgazerVideoContainer);
          webgazerVideoContainer.style.position = 'fixed';
        }
      }
    };
  }, []);

  return (
    <div className="space-y-6 relative">
      {isCalibrating && (
        <div className="fixed inset-0 z-[9999] bg-transparent cursor-crosshair">
          {CALIBRATION_POINTS.map((point, index) => {
            const clicks = clickCounts[index] || 0;
            const isDone = clicks >= CLICKS_REQUIRED;

            const peripheralDoneCount = CALIBRATION_POINTS.reduce((acc, _, i) => {
              if (i !== 4 && (clickCounts[i] || 0) >= CLICKS_REQUIRED) {
                return acc + 1;
              }
              return acc;
            }, 0);

            const isMiddleDot = index === 4;
            const shouldShow = !isMiddleDot || peripheralDoneCount === 8;

            if (!shouldShow) return null;

            return (
              <button
                key={index}
                onClick={(e) => {
                  e.stopPropagation();
                  handleCalibrationClick(index);
                }}
                style={{
                  left: `${point.x}%`,
                  top: `${point.y}%`,
                  position: 'absolute',
                  transform: 'translate(-50%, -50%)',
                }}
                className={`w-6 h-6 rounded-full border-2 border-white flex items-center justify-center text-[10px] font-bold ${
                  isDone 
                    ? 'bg-green-500 opacity-60' 
                    : 'bg-cyan-500 shadow-[0_0_15px_rgba(34,211,238,0.8)] animate-pulse scale-110'
                }`}
              >
                {!isDone && (CLICKS_REQUIRED - clicks)}
              </button>
            );
          })}
          
          {/* Dynamic Instruction Message */}
          <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
            <p className="text-white bg-black/60 px-6 py-3 rounded-full backdrop-blur-md border border-white/10">
              Click each dot {CLICKS_REQUIRED} times while looking at it
            </p>
          </div>
      </div>
      )}

      {/* --- ORIGINAL UI LAYOUT --- */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl text-white font-bold">Live Eye Tracking</h2>
          <p className="text-gray-400 mt-1">Real-time gaze tracking and calibration</p>
        </div>
        <div className="flex space-x-2">
          <Button
            onClick={startCalibration}
            variant="outline"
            className="bg-white/10 border-white/20 text-white hover:bg-white/20"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Calibrate
          </Button>
          <Button
            onClick={toggleTracking}
            disabled={!isCalibrated}
            className={`${
              isTracking ? 'bg-red-500 hover:bg-red-600' : 'bg-cyan-500 hover:bg-cyan-600'
            } text-white`}
          >
            {isTracking ? <><Pause className="w-4 h-4 mr-2" /> Stop Tracking</> : <><Play className="w-4 h-4 mr-2" /> Start Tracking</>}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 bg-white/5 border-white/10 p-6 overflow-hidden">
          <div className="relative aspect-video bg-slate-900/50 rounded-lg overflow-hidden border border-white/5">
             <div className="absolute inset-0 flex items-center justify-center text-white/20" id="cameraContainer">
                {!isTracking && !isCalibrating && <p>Camera Preview Area</p>}
             </div>
          </div>
        </Card>

        <div className="space-y-4">
          <Card className="bg-white/5 border-white/10 p-4">
            <h3 className="text-white mb-4 font-medium">Calibration Status</h3>
            <div className="flex items-center space-x-2">
              {isCalibrated ? (
                <><CheckCircle2 className="w-5 h-5 text-green-400" /><span className="text-green-400">Calibrated</span></>
              ) : (
                <><div className="w-5 h-5 rounded-full border-2 border-gray-400" /><span className="text-gray-400">Not Calibrated</span></>
              )}
            </div>
          </Card>

          <Card className="bg-white/5 border-white/10 p-4">
            <h3 className="text-white mb-4 font-medium">Gaze Coordinates</h3>
            <div className="space-y-2">
              <div className="flex justify-between font-mono">
                <span className="text-gray-400">X:</span>
                <span className="text-cyan-400">{gazePoint.x.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between font-mono">
                <span className="text-gray-400">Y:</span>
                <span className="text-cyan-400">{gazePoint.y.toFixed(1)}%</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}