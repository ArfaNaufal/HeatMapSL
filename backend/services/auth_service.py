from fastapi import HTTPException, status
import jwt
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from config import Config



class AuthService():

    def __init__(self):
        self._pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Converts plain text password to a BCrypt hash."""
        return self._pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Checks if the provided password matches the stored hash."""
        print(f"DEBUG: Verifying '{plain_password}' against '{hashed_password}'")
        return self._pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict):
        """Creates a JWT token using the secret from config.py."""
        to_encode = data.copy()
        
        expire = datetime.now(timezone.utc) + timedelta(minutes=Config.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, Config.JWT_TOKEN, algorithm=Config.ALGORITHM)
        return encoded_jwt
    
    def decode_token(self, token: str):
        try:
            payload = jwt.decode(
                token, 
                Config.JWT_TOKEN, 
                algorithms=[Config.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.PyJWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token:\n" + e.__str__()
            )
    

    
authService = AuthService()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")