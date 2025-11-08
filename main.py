from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import boto3
import os
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()
app = FastAPI()

# AWS S3 연결
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET = os.getenv("S3_BUCKET_NAME")

# 프론트(Netlify) 연결
origins = [
    "https://myapp.netlify.app",  # Netlify 배포 주소
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PyTorch 테스트용 (단순 선형 모델) =====
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
model.eval()


# ===== 기본 엔드포인트 =====
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


# ===== 인공지능 모듈 테스트용 =====
@app.get("/predict")
def predict(a: float, b: float, c: float):
    """입력값 3개를 받아 모델로 예측"""
    x = torch.tensor([[a, b, c]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x).item()
    return {"input": [a, b, c], "prediction": y}


# ===== 이미지 업로드 기능 =====
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    s3_key = f"images/{uuid4()}.{file_ext}"

    s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})
    file_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
    return {"url": file_url}


# ===== 점수 반환 (테스트용) =====
@app.get("/returnScore")
def return_score():
    return {"점수": [1, 2, 3, 4], "평가": ['a', 'b', 'c', 'd']}


# ===========================================================
#  새로 추가된 접근성 검사 엔드포인트 (/evaluate)
# ===========================================================

def check_spacing(elements, min_distance=8):
    violations = []
    n = len(elements)
    for i in range(n):
        for j in range(i + 1, n):
            A, B = elements[i], elements[j]
            dx = max(0, max(A['bbox']['x'], B['bbox']['x']) - min(A['bbox']['x'] + A['bbox']['w'], B['bbox']['x'] + B['bbox']['w']))
            dy = max(0, max(A['bbox']['y'], B['bbox']['y']) - min(A['bbox']['y'] + A['bbox']['h'], B['bbox']['y'] + B['bbox']['h']))
            dist = min(dx, dy)
            if dist < min_distance:
                violations.append({
                    "e1": A["id"], "e2": B["id"], "type": "spacing_violation"
                })
    return violations


def check_touch_target(elements, min_size=44):
    violations = []
    for e in elements:
        if e["type"] in ["button", "checkbox", "radio"]:
            if e["bbox"]["w"] < min_size or e["bbox"]["h"] < min_size:
                violations.append({
                    "id": e["id"],
                    "type": "touch_target_too_small"
                })
    return violations


def check_label(elements):
    violations = []
    for e in elements:
        if e["type"] != "text_field":
            continue
        has_label = False
        for l in elements:
            if l["type"] == "label":
                gap = e["bbox"]["x"] - (l["bbox"]["x"] + l["bbox"]["w"])
                if 0 < gap < 50 and abs(l["bbox"]["y"] - e["bbox"]["y"]) < 30:
                    has_label = True
                    break
        if not has_label:
            violations.append({
                "id": e["id"],
                "type": "missing_label"
            })
    return violations


@app.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()
    elements = data.get("elements", [])

    # --- 각 검사 함수 실행 ---
    spacing_v = check_spacing(elements)
    touch_v = check_touch_target(elements)
    label_v = check_label(elements)

    # --- per_element 구조화 ---
    per_element = {e["id"]: [] for e in elements}
    pairwise = []

    # touch_target, label 위반
    for v in touch_v + label_v:
        per_element[v["id"]].append(v["type"])

    # spacing 위반
    for v in spacing_v:
        pairwise.append({
            "e1": v["e1"],
            "e2": v["e2"],
            "violations": [v["type"]]
        })

    # --- summary 계산 ---
    all_viols = (
        [vv for vlist in per_element.values() for vv in vlist] +
        [v["type"] for v in spacing_v]
    )
    summary = {
        "total_elements": len(elements),
        "total_violations": len(all_viols),
        "violation_types": {t: all_viols.count(t) for t in set(all_viols)}
    }

    return {
        "per_element": per_element,
        "pairwise": pairwise,
        "summary": summary
    }
