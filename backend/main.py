from pathlib import Path
import uuid
import pandas as pd
from config import Config
from database import Database
from fastapi import FastAPI, HTTPException, Depends, Response, status, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from limits import parse
from models import Image_ModelUpload, UserLogin, UserCreate, HeatmapUpload
from services.auth_service import authService, oauth2_scheme
from contextlib import asynccontextmanager
from services.heatpmap_service import heatmap_service
from admin import router as admin_router
from admin import get_current_admin
import base64
import numpy as np
import cv2

db:Database = Database()
limiter = Limiter(key_func=get_remote_address)

def save_model_to_disk(base64_str: str, model_name: str) -> str:
    """
    Saves a base64 model file to the disk in the model folder.
    Returns the relative file path to be stored in the database.
    """

    model_dir = Config.MODEL_PATH
    model_dir.mkdir(parents=True, exist_ok=True)

    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    unique_id = uuid.uuid4().hex
    filename = f"{unique_id}_{model_name.replace(' ', '_')}.png"
    file_path = model_dir / filename

    model_data = base64.b64decode(base64_str)
    with open(file_path, "wb") as f:
        f.write(model_data)

    return str(file_path)

def delete_model_from_disk(file_path: str) -> bool:
    """
    Deletes the model file from disk.
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

def save_heatmap_to_disk(user_id: int, base64_str: str, session_name: str) -> str:
    """
    Saves a base64 image to the disk in a user-specific folder.
    Returns the relative file path to be stored in the database.
    """

    user_dir = Config.HEATMAP_PATH / f"user_{user_id}"
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

async def request_limiter ( request: Request) -> bool:
    limit:str
    auth_header = request.headers.get("Authorization")
    ip = get_remote_address(request)
    if ip == "127.0.0.1":
        limit = "1000/minute"
    elif auth_header:
        try:
            decoded_token = authService.decode_token(auth_header.split(" ")[1])
            if decoded_token["role"] == 2:
                limit = "100/minute"
            else:
                limit = "50/minute"
        except HTTPException as e:
            raise e
    else :
        path = request.url.path
        if path.startswith("/api/register"):
            limit = "2/day"
        else:
            limit = "50/hours"

    try :
        parsed_limit = parse(limit)
        is_allowed_ip = limiter.limiter.hit(parsed_limit, "manual", ip)
        is_allowed_token = limiter.limiter.hit(parsed_limit, "manual", auth_header) if auth_header else True
        if not is_allowed_ip or not is_allowed_token:
            raise RateLimitExceeded(limit=limit)
        return True
    except RateLimitExceeded as e:
        raise e

def check_authorization(request: Request) -> bool:
    rdict = {'status': False}
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.split(" ")[1] != "null":
        try:
            rdict["payload"] = authService.decode_token(auth_header.split(" ")[1])
            rdict["status"] = True
            return rdict
        except HTTPException as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials."
            )
    return rdict

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = authService.decode_token(token)
    if payload["role"] != 1:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please use user account to use user standarize api."
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
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
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
def register(usercreate: UserCreate, authorize = Depends(check_authorization), _ = Depends(request_limiter)):
    if authorize['status']:
        if authorize["payload"]["role"] == 1:
            raise HTTPException(status_code=403, detail="Authenticated users cannot register new accounts")
    elif usercreate.role == 2 and not authorize['status'] and authorize["payload"]["role"] != 2:
        raise HTTPException(status_code=403, detail="Only admins can create admin accounts")
    elif not db.findUser(usercreate.email).empty:
        raise HTTPException(status_code=500, detail="Email already used, please use other email to register new user account.")
    hashed_password = authService.hash_password(usercreate.password)
    usercreate.password = hashed_password

    result = db.addUser(usercreate)
    if not result:
        raise HTTPException(status_code=400, detail="User registration failed")
    
    if not authorize['status']:
        db_user = db.findUser(usercreate.email)
        token_data = {"sub": str(db_user["email"].loc[0]), "id": int(db_user["id"].loc[0]), "role": int(db_user["role"].loc[0])}
        token = authService.create_access_token(token_data)
        return {
            "access_token": token,
            "token_type": "bearer"
        }
    else:
        return {"message": "User registered successfully"}
    
@app.delete('/api/user/delete/{user_id}')
def delete_user(user_id: int, user_data: str = Depends(get_current_admin)):
    result = db.findUserById(user_id)
    if result.empty:
        raise HTTPException(status_code=404, detail="User not found")
    if int(result["role"].iloc[0]) == 2 and user_data["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized to delete admin user")
    deleted_rows = db.delUser(user_id)
    if deleted_rows and deleted_rows > 0:
        return {"message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete user")

@app.post('/api/model/upload')
def upload_model(data: Image_ModelUpload, user_data: str = Depends(get_current_admin)):
    
    try:
        pixel_matrix = decode_base64_to_cv2(data.base64_image)
    
        if pixel_matrix is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        file_path = save_model_to_disk(data.base64_image, data.model_name)
        model_id = db.addImageModel(data.model_name, file_path, user_data["id"])
        return {"status": "success", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/api/model/file/{model_id}')
async def get_model_file(model_id: int, token: str = Depends(oauth2_scheme)):
    authService.decode_token(token)
    row = db.findImageModel(model_id)[["model_path"]]

    if row.empty:
        raise HTTPException(status_code=404, detail="Model not found")
    
    file_path = row.loc[0].values[0]

    return FileResponse(file_path)

@app.get('/api/model/all')
def get_all_models(user_data: str = Depends(oauth2_scheme)):
    models = db.getAllImageModels()[['id', 'model_name', 'created_at', 'user_id']]
    return models.to_dict(orient='records')

@app.delete('/api/model/delete/{model_id}')
def delete_model_by_id(model_id: int, user_data: str = Depends(get_current_admin)):
    result = db.findImageModel(model_id)[["model_path", "user_id"]]
    if result.empty:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_path, user_id = result.loc[0].values

    if (int(user_id) != user_data["id"]) and user_data["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized access to this data")

    delete_success = delete_model_from_disk(model_path)
    if delete_success :
        deleted_rows = db.delImageModel(model_id)
        if deleted_rows and deleted_rows > 0:
            return {"status": "deleted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model from database")
    else:
        raise HTTPException(status_code=500, detail="Failed to delete model file from disk")
    
@app.get('/api/model/check/{img_name}')
def check_heatmaps_by_user(img_name:str, token: str = Depends(oauth2_scheme)):
    authService.decode_token(token)
    
    heatmaps: pd.DataFrame = db.findImageModelByName(img_name)[['id', 'model_name', 'created_at']]
    return {"status": img_name in heatmaps["model_name"].values}

@app.post('/api/heatmap/upload')
def upload_heatmap(data: HeatmapUpload, token: str = Depends(oauth2_scheme)):
    authService.decode_token(token)
    
    try:
        formatted_points = [[p.x, p.y, 1] for p in data.points]
        len_point = len(formatted_points)

        model_path = db.findImageModel(data.model_id)[["model_path"]].loc[0].values[0]
        if not model_path or not Path(model_path).exists():
            raise HTTPException(status_code=404, detail="Model not found")

        model_data = cv2.imread(Path(model_path), cv2.IMREAD_COLOR)

        if model_data is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        heatmap_img = heatmap_service.create_heatmap(
            gazepoints=formatted_points,
            dispsize=(data.width, data.height),
            background_img=model_data
        )

        file_path = save_heatmap_to_disk(data.user_id, heatmap_img, data.name)

        img_id = db.addHeatmap(data.name, file_path, data.model_id, data.user_id)

        return {"status": "success", "point_count": len_point, "session_id": img_id}
    except Exception as e:
        print(e)
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
    
    if result.empty:
        raise HTTPException(status_code=404, detail="Session not found")
    
    image_path, user_id = result.loc[0].values

    if (int(user_id) != user_data["id"]) and user_data["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized access to this data")

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
async def get_heatmap_file(session_id: int, token: str = Depends(oauth2_scheme)):
    user_data = authService.decode_token(token)
    row = db.findHeatmap(session_id)[["image_path", "user_id"]].loc[0].values

    if row.size == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    
    file_path, owner_id = row

    if (int(owner_id) != user_data["id"]) and user_data["role"] != 2:
        raise HTTPException(status_code=403, detail="Unauthorized access to this data")

    return FileResponse(file_path)

@app.get('/api/verify-token')
def verify_token(token: str = Depends(oauth2_scheme)):
    payload = authService.decode_token(token)
    return {"payload": payload}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.DB_HOST, port=int(Config.DB_PORT))