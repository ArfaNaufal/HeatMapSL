from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional
from datetime import datetime

# 1. For Registration
class UserCreate(BaseModel):
    email: EmailStr = Field(..., min_length=5, max_length=50)
    password: str = Field(..., min_length=5)
    role: int = 1

# 2. For Login
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# 3. For Heatmap Upload
class GazePoint(BaseModel):
    x: int
    y: int

class HeatmapUpload(BaseModel):
    name: str
    model_id: int
    user_id: int
    width: int
    height: int
    points: List[GazePoint]

class HeatmapGet(BaseModel):
    id: int
    user_id: int
    name: str
    created_at: datetime 
    gaze_points_count: int
    model_id: int
    image_path: str
    
    class Config:
        from_attributes = True

# 4. For Model Management
class Image_ModelUpload(BaseModel):
    model_name: str
    base64_image: str

class Image_ModelGet(BaseModel):
    id: int
    model_name: str
    model_path: str
    user_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True