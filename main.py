from fastapi import FastAPI
import torch
import torch.nn as nn

from fastapi import FastAPI, File, UploadFile
import boto3
import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
app = FastAPI()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET = os.getenv("S3_BUCKET_NAME")


app = FastAPI()


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
model.eval()


#메인페이지
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

#인공지능 모듈 실행(pytorch 테스트 위함-실제 스케치인식과 관련없음)
@app.get("/predict")
def predict(a: float, b: float, c: float):
    """입력값 3개를 받아 모델로 예측"""
    x = torch.tensor([[a, b, c]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    return {"input": [a, b, c], "prediction": y}

#이미지 업로드 기능
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    s3_key = f"images/{uuid4()}.{file_ext}"

    s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})
    file_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
    return {"url": file_url}