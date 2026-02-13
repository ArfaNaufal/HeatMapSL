import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from datetime import datetime as dt
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
load_dotenv(find_dotenv())

class Message ():
    """
        Message class for log in log text file.
    """
    _basePath:str = os.getenv('logPath', './')
    _Send: bool = False
    _Message: str
    _Code: str|None

    def __init__(self, message:str, code:str|None):
        self._Message = message
        self._Code = code if code else '-'
        self._Send = self.addMessage()

    def addMessage(self)->bool:
        f"""
            Adding message to log, that located on {self._basePath}
        """
        try:
            with open (self._basePath+'log.txt', "a") as op:
                op.write(f'({dt.now().strftime("%Y-%m-%d|%I:%M:%S")}) {self}\n')
        except Exception as e:
            return False
        return True
    
    def getSend(self)->bool:
        return self._Send
    
    def __str__(self)->str:
        return f'[{self._Code}] {self._Message}'



class Setting():
    PROJECT_NAME: str = 'EyeGaze_Backend'
    JWT_TOKEN: str = str(os.getenv('JWT_TOKEN', ''))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30))
    HEATMAP_PATH: Path = Path(os.getenv("HEATMAP_PATH", "data/heatmap_storage"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", "data/models"))
    ALGORITHM: str = os.getenv('ALGORITHM', 'HS256')
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PATH: str = os.getenv('DB_PATH', 'database.db')
    DB_PORT: int = int(os.getenv('DB_PORT', 8000))
    DB_USER: str = os.getenv('DB_USER', 'admin')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'admin')
    DB_NAME: str = os.getenv('DB_NAME', 'Backend')
    DEBUG: bool = os.getenv('DEBUG', "False") == "True"


    def __init__ (self):
        if not self.JWT_TOKEN:
            self.log('JWT_TOKEN not found in folder ".env"', 'NOTFOUND')
            raise ValueError("Please input 'JWT_TOKEN' for secret token.")
        
    def log(self, message:str, code:str)->Message|bool:
        msg = Message(message, code)
        return msg if msg.getSend() else False

Config = Setting()