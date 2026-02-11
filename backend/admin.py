from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from database import Database
from services.auth_service import authService, oauth2_scheme

import sqlite3

db_admin:Database = Database()

router = APIRouter(prefix="/api/admin", tags=["admin"])

def get_current_admin(token: str = Depends(oauth2_scheme)):
    payload = authService.decode_token(token)
    if payload["role"] != 2:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can register new users"
        )
    return payload

@router.get("/dashboard-data")
async def get_dashboard_data(admin = Depends(get_current_admin)):
    
    try:
        stats = {
            "users": int(db_admin._returnDataframe(db_admin.queryExecution("SELECT COUNT(*) FROM Users")).iloc[0,0]),
            "heatmaps": int(db_admin._returnDataframe(db_admin.queryExecution("SELECT COUNT(*) FROM Heatmaps")).iloc[0,0])
        }

        users = db_admin._returnDataframe(db_admin.queryExecution("""
            SELECT u.id, u.email, u.role, COUNT(h.id) as sessions 
            FROM Users u LEFT JOIN Heatmaps h ON u.id = h.user_id 
            GROUP BY u.id
        """)).to_dict(orient='records')


        heatmaps = db_admin._returnDataframe(db_admin.queryExecution("""
            SELECT h.id, h.img_name as name, h.created_at, u.email as owner 
            FROM Heatmaps h JOIN Users u ON h.user_id = u.id
        """)).to_dict(orient='records')
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail="Database query failed")
    
    return {"stats": stats, "users": users, "heatmaps": heatmaps}

@router.get("/logs", response_class=PlainTextResponse)
async def get_logs(admin = Depends(get_current_admin)):
    log_path = Path("log.txt")
    if not log_path.exists():
        return HTTPException(status_code=404, detail="Log file not found")
    with open(log_path, "r") as f:
        return "".join(f.readlines()[-300:])

@router.options("/alter_users")
async def alter_users_options():
    pass