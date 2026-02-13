import { useState, useEffect, forwardRef } from 'react';
import { Skeleton } from '@/app/components/ui/skeleton';
import { ImageIcon } from 'lucide-react';
import { API_URL } from '@/app/App';

interface SecureImageProps {
  sessionId: number;
  className?: string;
  alt?: string;
}

export const SecureHeatmap = forwardRef<HTMLImageElement, SecureImageProps>(({ sessionId, className, alt }, ref) => {
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;

    const fetchImage = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${API_URL}/heatmaps/file/${sessionId}`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
          }
        );

        if (!response.ok) throw new Error('Unauthorized or missing');

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        setImgUrl(objectUrl);
      } catch (err) {
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchImage();

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [sessionId]);

  if (loading) return <Skeleton className={className} />;
  if (error) return <div className={`${className} flex items-center justify-center bg-white/5`}><ImageIcon className="text-gray-600 w-4 h-4" /></div>;

  return <img src={imgUrl!} className={className} alt={alt || "Heatmap Preview"} ref={ref} />;
  }
);

export const SecureImageModel = forwardRef<HTMLImageElement, { modelId: number; className?: string; alt?: string; ref?: React.Ref<HTMLImageElement> }>(({ modelId, className, alt }, ref) => {
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;

    const fetchImage = async () => {
      try {
        setLoading(true);
        const response = await fetch(
          `${API_URL}/model/file/${modelId}`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`
            }
          }
        );

        if (!response.ok) throw new Error('Unauthorized or missing');

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        setImgUrl(objectUrl);
      } catch (err) {
        setError(true);
      } finally {
        setLoading(false);
      }
    };

    fetchImage();

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [modelId]);

  if (loading) return <Skeleton className={className} />;
  if (error) return <div className={`${className} flex items-center justify-center bg-white/5`}><ImageIcon className="text-gray-600 w-4 h-4" /></div>;

  return <img src={imgUrl!} className={className} alt={alt || "Model Preview"} ref={ref} />;
  }
);