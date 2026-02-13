import { useState, useEffect, useRef } from 'react';
import { Upload, Trash2, ImageIcon, Tag, Users, Image, Map, Terminal, Eye, RefreshCw, Activity, Plus } from 'lucide-react';
import { Button } from '@/app/components/ui/button';
import { Card } from '@/app/components/ui/card';
import { Input } from '@/app/components/ui/input';
import { toast } from 'sonner';
import { SecureHeatmap, SecureImageModel } from './SecureImage';
import { API_URL } from '@/app/App';

export function AdminPanel() {
    const [activeTab, setActiveTab] = useState<'stats' | 'users' | 'models' | 'heatmaps' | 'logs'>('stats');
    const [isShowViewHeatmap, setIsShowViewHeatmap] = useState(false);
    const [isShowViewModel, setIsShowViewModel] = useState(false);
    const [isShowAddUserModal, setIsShowAddUserModal] = useState(false);
    const [modelName, setModelName] = useState("");
    const [uploadedImage, setUploadedImage] = useState<string | null>(null);
    const [viewHeatmapSessionId, setViewHeatmapSessionId] = useState<number | null>(null);
    const [viewModelId, setViewModelId] = useState<number | null>(null);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const viewHeatmapModalRef = useRef<HTMLDivElement>(null);
    const viewModelModalRef = useRef<HTMLDivElement>(null);
    const [data, setData] = useState<any>(null);
    const [logs, setLogs] = useState("");
    const [isLoading, setIsLoading] = useState(true);

    const fetchAdminData = async () => {
        setIsLoading(true);
        const headers = { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` };
        try {
        const [dbRes, logRes] = await Promise.all([
            fetch(`${API_URL}/admin/dashboard-data`, { headers }),
            fetch(`${API_URL}/admin/logs`, { headers })
        ]);
        if (dbRes.ok) setData(await dbRes.json());
        if (logRes.ok) setLogs(await logRes.text());
        } catch (err) {
        toast.error("Failed to sync with server.");
        } finally {
        setIsLoading(false);
        }
    };

    const checkSession = async (img_name:string) => {
        try {
          const response = await fetch(`${API_URL}/model/check/${img_name}`, {
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
            fetchAdminData();
            sessionId === viewHeatmapSessionId && setIsShowViewHeatmap(false);
            setViewHeatmapSessionId(null);
            return 'Session deleted successfully!'
            },
            error: 'Failed to delete session.'
        }
        )
    }

    const handleAddUser = async (email: string, password: string, role: string) => {
        
    }

    const savedImage = async () => {
        if (!uploadedImage) return toast.error("No image uploaded.");

        const payload = {
            model_name: modelName,
            base64_image: uploadedImage
        }

        toast.promise(
            fetch(`${API_URL}/model/upload`, {
                method: 'POST',
                headers: { 
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${localStorage.getItem('access_token')}`
                },
                body: JSON.stringify(payload)
            }),
            {
                loading: 'Saving image model...',
                success: ()=>{
                fetchAdminData();
                return 'Model saved successfully!'
                },
                error: 'Failed to save model data.'
            }
        );
    }

    const handleDeleteModel = async (modelId: number) => {}
    const handleUploadModel = async () => {
        if ((await checkSession(modelName))) return toast.error("Model name already used.");
        if (!modelName) return toast.error("Please name the model first.");

        await savedImage()
    }
    useEffect(() => {
        if (isShowViewHeatmap) {
        document.body.style.overflow = 'hidden';
        viewHeatmapModalRef.current?.focus();
        } else {
        document.body.style.overflow = 'unset';
        }
    }, [isShowViewHeatmap]);

    useEffect(() => {
        if (isShowViewModel) {
        document.body.style.overflow = 'hidden';
        viewModelModalRef.current?.focus();
        } else {
        document.body.style.overflow = 'unset';
        }
    }, [isShowViewModel]);

    useEffect(() => {
        if (logContainerRef.current)
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }, [logs, activeTab]);

    useEffect(() => { fetchAdminData(); }, []);

    return (
        <>
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
                    ref={viewHeatmapModalRef}
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

            {isShowViewModel && (
            <div className="fixed inset-0 z-[10000] bg-black flex flex-col items-center justify-center">
                <div className="relative w-screen h-screen flex items-center justify-center bg-black"
                    onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                        setIsShowViewModel(false);
                        setViewModelId(null);
                    }
                    }}
                    tabIndex={0}
                    ref={viewModelModalRef}
                    >
                    <SecureImageModel
                        modelId={viewModelId!} 
                        className="w-full h-full object-cover opacity-80" 
                />
                </div>
                <div className="absolute top-6 right-6">
                    <Button onClick={() => {
                        setIsShowViewModel(false);
                        setViewModelId(null);
                    }} variant="ghost" className="text-white border border-black bg-red-500/15 hover:bg-red-500/25 rounded-lg px-4 py-2"><Trash2 className="w-4 h-4 mr-2" /> Close</Button>
                </div>
            </div>
            )}

            <div className="space-y-8 animate-in fade-in duration-500">
                {/* Header Area */}
                <div className="flex justify-between items-end border-b border-white/5 pb-6">
                    <div>
                        <h2 className="text-4xl font-bold text-white tracking-tight">System Control</h2>
                        <p className="text-gray-500 mt-1">Unified Database Management & Server Monitoring</p>
                    </div>
                    <Button onClick={fetchAdminData} variant="outline" className="bg-white/5 border-white/10 text-cyan-400">
                        <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} /> Sync Data
                    </Button>
                </div>

                {/* Navigation Tabs */}
                <div className="flex space-x-2 bg-white/5 p-1.5 rounded-2xl w-fit border border-white/10">
                    {[
                        { id: 'stats', icon: Activity, label: 'Overview' },
                        { id: 'users', icon: Users, label: 'User DBMS' },
                        { id: 'models', icon: Image, label: 'Model Management' },
                        { id: 'heatmaps', icon: Map, label: 'Heatmap DBMS' },
                        { id: 'logs', icon: Terminal, label: 'Live Logs' }
                    ].map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as any)}
                            className={`flex items-center px-6 py-2.5 rounded-xl transition-all font-medium ${
                            activeTab === tab.id ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/20' : 'text-gray-400 hover:text-white'
                            }`}
                        >
                            <tab.icon className="w-4 h-4 mr-2" /> {tab.label}
                        </button>
                    ))}
                </div>

                {/* --- TAB CONTENT --- */}

                {activeTab === 'stats' && data && (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <Card className="bg-white/5 border-white/10 p-8">
                            <p className="text-xs font-bold text-gray-500 uppercase tracking-widest">Global Users</p>
                            <p className="text-6xl font-mono mt-4 text-white">{data.stats.users}</p>
                        </Card>
                        <Card className="bg-white/5 border-white/10 p-8">
                            <p className="text-xs font-bold text-gray-500 uppercase tracking-widest">Total Analyses</p>
                            <p className="text-6xl font-mono mt-4 text-cyan-400">{data.stats.heatmaps}</p>
                        </Card>
                        <Card className="bg-white/5 border-white/10 p-8">
                            <p className="text-xs font-bold text-gray-500 uppercase tracking-widest">Total Models</p>
                            <p className="text-6xl font-mono mt-4 text-cyan-400">{data.stats.models}</p>
                        </Card>
                    </div>
                )}

                {activeTab === 'users' && (
                    <>
                        <Card className="bg-white/5 border-white/10 overflow-hidden">
                            <table className="w-full text-left">
                                <thead className="bg-white/5 text-gray-400 text-[10px] uppercase font-bold">
                                <tr>
                                    <th className="px-6 py-4">Identity</th>
                                    <th className="px-6 py-4">Role</th>
                                    <th className="px-6 py-4">Sessions</th>
                                    <th className="px-14 py-4 text-right">Action</th>
                                </tr>
                                </thead>
                                <tbody className="divide-y divide-white/5">
                                {data?.users.map((u: any) => (
                                    <tr key={u.id} className="hover:bg-white/[0.02] group">
                                    <td className="px-6 py-4">
                                        <div className="text-white font-medium">{u.email}</div>
                                        <div className="text-[10px] text-gray-500 font-mono">ID: {u.id}</div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <span className="px-2 py-1 rounded-md bg-cyan-500/10 text-cyan-400 text-xs border border-cyan-500/20">
                                        {u.role}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 font-mono text-gray-400">{u.sessions}</td>
                                    <td className="px-6 py-4 text-right">
                                        <Button variant="ghost" className="text-red-400 opacity-45 group-hover:opacity-100"><Trash2 className="w-4 h-4" /></Button>
                                    </td>
                                    </tr>
                                ))}
                                </tbody>
                            </table>
                        </Card>
                    
                        <div className="flex space-x-2 p-1.5 rounded-2xl w-fit">
                            <Button
                                onClick={() => {setIsShowAddUserModal(true)}}
                                variant="outline" 
                                className="bg-white/5 border-white/10 text-cyan-400"
                            >
                                <Plus className="w-4 h-4 mr-2" /> Add User
                            </Button>
                        </div>
                    </>
                )}

                {activeTab === 'models' && (
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <Card className="lg:col-span-2 bg-white/5 border-white/10 p-10 flex flex-col items-center justify-center min-h-[400px]">
                                {!uploadedImage ? (
                                    <div className="text-center">
                                    <div className="w-20 h-20 bg-white/5 rounded-full flex items-center justify-center mx-auto mb-6">
                                        <Upload className="w-10 h-10 text-cyan-500" />
                                    </div>
                                    <h3 className="text-xl text-white font-bold mb-2">Upload Model</h3>
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
                                                placeholder="Model Name (e.g. Amazon Hero Banner)" 
                                                className="pl-12 bg-white/5 border-white/10 h-14 text-white rounded-xl"
                                                value={modelName}
                                                onChange={(e) => setModelName(e.target.value)}
                                            />
                                            </div>
                                            <div className="col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4"> 
                                                <Button onClick={handleUploadModel} size="lg" className="w-full h-14 bg-cyan-500 hover:bg-cyan-600 text-white text-lg font-bold rounded-xl">
                                                    <Upload className="w-5 h-5 mr-2" /> Upload Model
                                                </Button>
                                                <Button onClick={() => setUploadedImage(null)} variant="ghost" className="w-full h-14 bg-red-500 text-white text-lg font-bold rounded-xl hover:bg-red-800/40">
                                                    <Trash2 className="w-4 h-4 mr-2" />Reset Image
                                                </Button>
                                            </div>
                                        </div>
                                        
                                    </div>
                                )}
                            </Card>


                            {/* Info Sidebar */}
                            <Card className="bg-white/5 border-white/10 overflow-hidden">
                                <table className="w-full text-left">
                                    <thead className="bg-white/5 text-gray-400 text-[10px] uppercase font-bold">
                                    <tr>
                                        <th className="px-6 py-4">Model ID</th>
                                        <th className="px-6 py-4">Owner (User)</th>
                                        <th className="px-6 py-4 text-right">Action</th>
                                    </tr>
                                    </thead>
                                    <tbody className="divide-y divide-white/5">
                                    {data?.models.map((h: any) => (
                                        <tr key={h.id} className="hover:bg-white/[0.02]">
                                        <td className="px-6 py-4 flex items-center space-x-4">
                                            <div className="w-12 h-8 bg-black rounded border border-white/10 overflow-hidden">
                                            <SecureImageModel modelId={h.id} className="w-full h-full object-cover" />
                                            </div>
                                            <span className="text-white">{h.model_name}</span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-400 italic text-xs">{h.owner}</td>
                                        <td className="px-6 py-4 text-right space-x-2">
                                            <Button variant="ghost" onClick={()=> {
                                                setViewModelId(h.id);
                                                setIsShowViewModel(true);
                                                }
                                            } size="sm" className="hover:text-cyan-400"><Eye className="w-4 h-4" /></Button>
                                            <Button variant="ghost" size="sm" onClick={()=>handleDeleteModel(h.id)} className="hover:text-red-400"><Trash2 className="w-4 h-4" /></Button>
                                        </td>
                                        </tr>
                                    ))}
                                    </tbody>
                                </table>
                            </Card>
                        </div>
                    </div>
                )}

                {activeTab === 'heatmaps' && (
                    <Card className="bg-white/5 border-white/10 overflow-hidden">
                        <table className="w-full text-left">
                            <thead className="bg-white/5 text-gray-400 text-[10px] uppercase font-bold">
                            <tr>
                                <th className="px-6 py-4">Heatmap Session</th>
                                <th className="px-6 py-4">Owner (User)</th>
                                <th className="px-6 py-4 text-right">Action</th>
                            </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                            {data?.heatmaps.map((h: any) => (
                                <tr key={h.id} className="hover:bg-white/[0.02]">
                                <td className="px-6 py-4 flex items-center space-x-4">
                                    <div className="w-12 h-8 bg-black rounded border border-white/10 overflow-hidden">
                                        <SecureHeatmap sessionId={h.id} className="w-full h-full object-cover" />
                                    </div>
                                    <span className="text-white">{h.name}</span>
                                </td>
                                <td className="px-6 py-4 text-gray-400 italic text-xs">{h.owner}</td>
                                <td className="px-6 py-4 text-right space-x-2">
                                    <Button variant="ghost" onClick={()=> {
                                        setViewHeatmapSessionId(h.id);
                                        setIsShowViewHeatmap(true);
                                        }
                                    } size="sm" className="hover:text-cyan-400"><Eye className="w-4 h-4" /></Button>
                                    <Button variant="ghost" size="sm" onClick={()=>handleDeleteSession(h.id)} className="hover:text-red-400"><Trash2 className="w-4 h-4" /></Button>
                                </td>
                                </tr>
                            ))}
                            </tbody>
                        </table>
                    </Card>

                )}

                {activeTab === 'logs' && (
                    <div className="space-y-4">
                        <div className="bg-black border border-white/10 rounded-2xl p-6 h-[550px] overflow-y-auto font-mono text-sm shadow-inner"
                            ref={logContainerRef}>
                            <pre className="text-green-500/90 leading-relaxed whitespace-pre-wrap">
                            {logs || "// Initializing log stream..."}
                            </pre>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}