from typing import Any
from config import Config
import sqlite3 as sql
import pandas as pd
from models import UserCreate
from pathlib import Path

class Database():
    _connect: sql.Connection

    def __init__(self):
        try:
            if (Path(Config.DB_PATH).exists() == False):
                Path(Config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
            self._connect = sql.connect(Config.DB_PATH)
        except Exception as e:
            raise e
        
        self._connect.execute("PRAGMA foreign_keys = ON;")

        self._connect.execute("""
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL CHECK (email LIKE '%_@_%._%'),
                password TEXT NOT NULL,
                role INTEGER NOT NULL DEFAULT 1
            )
        """)

        self._connect.execute("""
            CREATE TABLE IF NOT EXISTS Heatmaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                img_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES Users (id) ON DELETE CASCADE
            )
        """)

        self._connect.commit()
        self._connect.close()

    def _returnDataframe(self, cursor:sql.Cursor )-> pd.DataFrame:
        data:list[Any] = cursor.fetchall()
        colNames:list[Any] = [ desc[0] for desc in cursor.description]
        return pd.DataFrame(data, columns=colNames)

    def getConnection(self)->sql.Connection:
        return sql.connect(Config.DB_PATH)
    
    def queryExecution(self, query:str, var:tuple[Any]=[])->sql.Cursor:
        cursor: sql.Cursor
        with self.getConnection() as con:
            try:
                cursor: sql.Cursor = con.execute(query, var) if len(var)>0 else con.execute(query)
                con.commit()
                for i in range(0,len(var)):
                    strvar = str(var[i])
                    query = query.replace('?', strvar if len(strvar) < 128 else strvar[:32] + "..." + strvar[-32:], 1)
                Config.log(f"Sucessfully execute query ({query})", "QUERYSUCCESS")
            except sql.Error as e:
                con.rollback()
                con.close()
                raise e
        return cursor
    
    def addUser(self, user: UserCreate)->int|None:
        query = "INSERT INTO Users (email, password, role) VALUES (?, ?, ?)"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (user.email.__str__(), user.password, user.role,))
            return cursor.rowcount
        except sql.Error as e:
            Config.log(f"There is error when trying to add User ({user.email.__str__()}, {user.password}, {user.role})", "QUERYERROR")
        return None

    def delUser(self, email:str)->int|None:
        query = "DELETE FROM Users WHERE email LIKE ?"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (email,))
            return cursor.rowcount
        except sql.Error as e:
            Config.log(f"There is error when trying to delete User ({email})", "QUERYERROR")
        return None
    
    def findUser(self, email: str)->pd.DataFrame:
        query = "SELECT * FROM Users WHERE email LIKE ?"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (email,))
            data:pd.DataFrame = self._returnDataframe(cursor)
            return data
        except sql.Error as e:
            Config.log(f"There is error when trying to find User ({email})", "QUERYERROR")
        return None
    
    
    def addHeatmap(self, img_name:str, image_path:str, user_id:int )->int|None:
        query = "INSERT INTO Heatmaps (img_name, image_path, user_id) VALUES (?, ?, ?)"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (img_name, image_path, user_id))
            return cursor.lastrowid
        except sql.Error as e:
            Config.log(f"There is error when trying to add Heatmap ({img_name})", "QUERYERROR")
        return None

    def delHeatmap(self, id:int)->int|None:
        query = "DELETE FROM Heatmaps WHERE id = ?"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (id,))
            return cursor.rowcount
        except sql.Error as e:
            Config.log(f"There is error when trying to delete Heatmap ({id})", "QUERYERROR")
        return None
    
    def findHeatmap(self, id: int)->pd.DataFrame:
        query = "SELECT * FROM Heatmaps WHERE id = ?"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (id,))
            data:pd.DataFrame = self._returnDataframe(cursor)
            return data
        except sql.Error as e:
            Config.log(f"There is error when trying to find Heatmap ({id})", "QUERYERROR")
        return None
    
    def findHeatmapByUserID(self, id: int)->pd.DataFrame:
        query = "SELECT * FROM Heatmaps WHERE user_id = ?"
        try:
            cursor:sql.Cursor = self.queryExecution(query, (id,))
            data:pd.DataFrame = self._returnDataframe(cursor)
            return data
        except sql.Error as e:
            Config.log(f"There is error when trying to find Heatmap ({id})", "QUERYERROR")
        return None
    
    
