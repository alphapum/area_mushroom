from sqlalchemy import create_engine ,types
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import psycopg2

from predict import predict

engine = create_engine('postgresql://postgres:V89infinity@localhost:5432/farm_mushroom')
conn = engine.connect()
# def connect_db():
#     engine = create_engine('postgresql://postgres:V89infinity@localhost:5432/farm_mushroom')
#     conn = engine.connect()
#     return conn , engine

def base64_to_image(base64_string):
    """แปลงสตริง Base64 เป็นอ็อบเจ็กต์ Image ของ PIL."""
    try:
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes))
        white = f"{predict(image):.2f}" 
        return white
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแปลง Base64: {e}")
        return None
    
def select_db():
    table_name = 'image_data'
    try:
        # engine = connect_db()
        
        # conn.autocommit = True  # สำคัญ: ตั้งค่า autocommit เป็น True เพื่อให้คำสั่ง CREATE DATABASE มีผลทันที
        # cur = conn.cursor()    
        sql_query = "SELECT * FROM "+table_name
        df = pd.read_sql(sql_query, engine)
        
        df['Number'] = df.index + 1
        
        return df
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแทรกข้อมูล: {e}")
    finally:
        engine.dispose()    

def update_db():
    table_name = 'image_data'
    try:
        # conn, engine = connect_db()
        
        sql_query = "UPDATE image_data SET white = %s WHERE idx = %s"
        pd.read_sql(sql_query, engine)
        
        
        
        return print(f"อัพข้อมูลสำเร็จ: {e}")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการแทรกข้อมูล: {e}")
    finally:
        engine.dispose()    
        
def predicx():
    df = select_db()
    df2 = df[df['idx'] >= 180]
    
    df2['white'] = df2['img'].apply( base64_to_image)
    
if __name__ == "__main__":
    predicx()
    