from pathlib import Path
import uuid
import pandas as pd
from config import Config
from database import Database
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from models import UserLogin, UserCreate, HeatmapUpload
from services.auth_service import authService, oauth2_scheme
from contextlib import asynccontextmanager
from services.heatpmap_service import heatmap_service
from admin import router as admin_router
from admin import get_current_admin
import base64
import numpy as np
import cv2

db:Database = Database()

def save_heatmap_to_disk(user_id: int, base64_str: str, session_name: str) -> str:
    """
    Saves a base64 image to the disk in a user-specific folder.
    Returns the relative file path to be stored in the database.
    """

    user_dir = Config.STORAGE_BASE / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{session_name.replace(' ', '_')}.png"
    file_path = user_dir / filename

    img_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(img_data)

    return str(file_path)

def delete_heatmap_from_disk(file_path: str) -> bool:
    """
    Deletes the heatmap image from disk.
    Returns True if deletion was successful, False otherwise.
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        else:
            return False
    except Exception as e:
        Config.log(f"Error deleting file {file_path}: {e}", "FILEDELETEERROR")
        return False

def decode_base64_to_cv2(base64_string):
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    img_bytes = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def create_dummy():
    dummy_admin = db.findUser(Config.DB_USER)
    if dummy_admin.empty:
        admin_user = UserCreate(
            email=Config.DB_USER,
            password=Config.DB_PASSWORD,
            role=2
        )
        hashed_password = authService.hash_password(admin_user.password)
        admin_user.password = hashed_password
        db.addUser(admin_user)

    dummy_user = db.findUser('dummy@heatmap.id')
    if dummy_user.empty:
        normal_user = UserCreate(
            email='dummy@heatmap.id',
            password='dummy',
            role=1
        )
        hashed_password = authService.hash_password(normal_user.password)
        normal_user.password = hashed_password
        db.addUser(normal_user)

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = authService.decode_token(token)
    if payload["role"] != 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please use user account to use heatmap features."
        )
    return payload

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs ON STARTUP
    try:
        print("Initializing Database...")
        create_dummy()
        print("Startup complete!")
    except Exception as e:
        print(f"CRITICAL ERROR ON STARTUP: {e}")
        raise e
    yield

app = FastAPI(title=Config.PROJECT_NAME, lifespan=lifespan)
app.include_router(admin_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/api/login')
def login(userlogin: UserLogin):
    db_user = db.findUser(userlogin.email)
    if db_user.empty or authService.verify_password(userlogin.password, db_user['password'].loc[0]) is False:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token_data = {"sub": str(db_user["email"].loc[0]), "id": int(db_user["id"].loc[0]), "role": int(db_user["role"].loc[0])}
    token = authService.create_access_token(token_data)

    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.post('/api/register')
def register(usercreate: UserCreate, admin: dict = Depends(get_current_admin)):
    hashed_password = authService.hash_password(usercreate.password)
    usercreate.password = hashed_password
    result = db.addUser(usercreate)
    if result.empty:
        raise HTTPException(status_code=400, detail="User registration failed")
    return {"message": "User registered successfully"}

@app.post('/api/heatmap/upload')
def upload_heatmap(data: HeatmapUpload, token: str = Depends(oauth2_scheme)):
    authService.decode_token(token)
    
    try:
        formatted_points = [[p.x, p.y, 1] for p in data.points]
        len_point = len(formatted_points)
        pixel_matrix = decode_base64_to_cv2(data.base64_image)
    
        if pixel_matrix is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        heatmap_img = heatmap_service.create_heatmap(
            gazepoints=formatted_points,
            dispsize=(data.width, data.height),
            background_img=pixel_matrix
        )

        file_path = save_heatmap_to_disk(data.user_id, heatmap_img, data.name)
        
        img_id = db.addHeatmap(data.name, file_path, data.user_id)
        return {"status": "success", "point_count": len_point, "session_id": img_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/api/heatmap/get_by_user/{user_id}')
def get_heatmaps_by_user(user_id: int, token: str = Depends(oauth2_scheme)):
    user_data = authService.decode_token(token)
    user_id = user_data.get("id")
    
    heatmaps = db.findHeatmapByUserID(user_id)[['id', 'img_name', 'created_at']]
    return heatmaps.to_dict(orient='records')

@app.get('/api/heatmap/check/{user_id}/{img_name}')
def check_heatmaps_by_user(user_id: int, img_name:str, token: str = Depends(oauth2_scheme)):
    user_data = authService.decode_token(token)
    user_id = user_data.get("id")
    
    heatmaps: pd.DataFrame = db.findHeatmapByUserID(user_id)[['id', 'img_name', 'created_at']]
    return {"status": img_name in heatmaps["img_name"].values}

@app.delete('/api/heatmap/delete/{session_id}')
def delete_heatmap_by_sessionid(session_id: int, token: str = Depends(oauth2_scheme)):
    user_data = authService.decode_token(token)
    result = db.findHeatmap(session_id)[["image_path", "user_id"]]
    
    if result.size == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image_path, user_id = db.findHeatmap(session_id)[["image_path", "user_id"]].loc[0].values

    if (int(user_id) != user_data["id"]) and user_data["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized access to this data")
    print(image_path, user_id)

    delete_success = delete_heatmap_from_disk(image_path)
    if delete_success :
        deleted_rows = db.delHeatmap(session_id)
        if deleted_rows and deleted_rows > 0:
            return {"status": "deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete heatmap from database")
    else:
        raise HTTPException(status_code=500, detail="Failed to delete heatmap file from disk")

@app.get("/api/heatmaps/file/{session_id}")
async def get_secure_file(session_id: int, current_user = Depends(get_current_user)):
    row = db.findHeatmap(session_id)[["image_path", "user_id"]].loc[0].values

    if row.size == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    file_path, owner_id = row

    if (int(owner_id) != current_user["id"]) and current_user["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized access to this data")

    return FileResponse(file_path)

@app.get('/api/verify-token')
def verify_token(token: str = Depends(oauth2_scheme)):
    payload = authService.decode_token(token)
    return {"payload": payload}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.DB_HOST, port=int(Config.DB_PORT))