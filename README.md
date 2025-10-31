# DeepPose: Human Pose Estimation via Deep Neural Networks

PyTorch 구현 (CVPR 2014 논문 재현)

---

##   최종 결과

| Stage | PCP@0.5 | PDJ@0.2 | 학습 시간 |
|-------|---------|---------|-----------|
| Stage 1 | 19.56% | 28.05% | ~2시간 |
| **Stage 2** | **32.01%** | **40.07%** | ~7시간 |
| Stage 3 | 30.04% | 38.07% | ~10시간 |

**최고 성능**: Stage 2 - PDJ@0.2 40.07%

### 논문 대비 성능

| 방법 | PDJ@0.2 | 비고 |
|------|---------|------|
| 원본 논문 (2014) | ~82% | ImageNet pre-training 사용 |
| 본 구현 | **40.07%** | Pre-training 없음 |

논문 대비 **약 49%** 수준 달성 (첫 구현치고 합리적인 결과)

---

##   모델 구조

### 네트워크 아키텍처

```
입력: 220×220×3 RGB 이미지

[단일 Stage - DeepPoseNetwork]
Conv1: 96 filters, 11×11, stride 4 → ReLU → MaxPool
Conv2: 256 filters, 5×5 → ReLU → MaxPool  
Conv3: 384 filters, 3×3 → ReLU
Conv4: 384 filters, 3×3 → ReLU
Conv5: 256 filters, 3×3 → ReLU → MaxPool

FC1: 4096 units → ReLU → Dropout(0.6)
FC2: 4096 units → ReLU → Dropout(0.6)
Output: 20 units (10 joints × 2 coordinates)

총 파라미터: ~58M
```

### Cascade 구조

```
Stage 1: 전체 이미지 → 초기 pose 예측
Stage 2: 전체 이미지 → 개선된 pose (독립적)
Stage 3: 전체 이미지 → 최종 pose (독립적)
```

**  현재 구현의 한계**: 
- 각 Stage가 독립적으로 학습됨
- 원본 논문의 cascade 구조 (이전 stage 예측 활용) 미구현
- 결과적으로 Stage 2가 가장 좋은 성능 (과적합 전)

---

##   데이터셋

### FLIC (Frames Labeled In Cinema)

- **학습 데이터**: 3,987 이미지
- **검증 데이터**: 1,016 이미지
- **관절 수**: 10개 (상체)
  - 어깨, 팔꿈치, 손목, 힙, 눈 (좌/우)

### 데이터 증강

| Stage | 증강 배수 | 총 샘플 수 |
|-------|----------|-----------|
| Stage 1 | 5× | 19,935 |
| Stage 2 | 40× | 159,480 |
| Stage 3 | 40× | 159,480 |

**증강 기법**:
- 랜덤 스케일 (0.95-1.05)
- 랜덤 이동 (±10 픽셀)
- 좌우 반전
- 랜덤 크롭

---

##   하이퍼파라미터

```python
# 이미지
IMG_SIZE = 220

# 모델
NUM_JOINTS = 10
NUM_STAGES = 3
DROPOUT = 0.6

# 학습
BATCH_SIZE = 128
LEARNING_RATE = 0.00005  # 5e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# 에포크
NUM_EPOCHS_STAGE1 = 100
NUM_EPOCHS_REFINE = 100

# Loss
SIGMA = 1.0
```

### Optimizer

- **타입**: Adam
- **Learning Rate**: 5e-5
- **스케줄러**: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 5 epochs
  - Min LR: 1e-6

### Early Stopping

- Patience: 10 epochs
- 기준: Validation loss

---

##   학습 과정

### Stage별 학습 전략

**Stage 1** (초기 pose 추정)
- 샘플 수: 19,935 (5× 증강)
- 수렴: Epoch 41/100
- 시간: ~2시간
- 결과: PDJ 28.05%

**Stage 2** (첫 번째 refinement)  
- 샘플 수: 159,480 (40× 증강)
- 수렴: Epoch 21/100
- 시간: ~7시간
- 결과: PDJ 40.07% (최고)

**Stage 3** (두 번째 refinement)
- 샘플 수: 159,480 (40× 증강)
- 수렴: Epoch 51/100
- 시간: ~10시간
- 결과: PDJ 38.07% (과적합 징후)

**총 학습 시간**: ~19시간 (GPU 1개)

---

##   주요 문제 해결

### 1. NaN Loss 문제

**증상**: 학습 초기 NaN 발생으로 붕괴

**해결책**:
```python
# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. 입력 검증
if torch.isnan(images).any() or torch.isnan(joints).any():
    continue

# 3. Loss 검증
if torch.isnan(loss) or torch.isinf(loss):
    continue

# 4. Learning rate 감소
LEARNING_RATE = 0.00005  # 0.0001에서 감소
```

### 2. Cascade 구조 미구현

**현재 구현**:
```python
# 각 Stage가 독립적으로 학습
Stage 1: image → pose1
Stage 2: image → pose2 (pose1 무시) 
Stage 3: image → pose3 (pose2 무시) 
```

**원본 설계**:
```python
# 이전 예측을 활용한 progressive refinement
Stage 1: image → pose1
Stage 2: crop(image, pose1) → pose2 ✓
Stage 3: crop(image, pose2) → pose3 ✓
```

**영향**: 
- Stage들이 독립적인 pose estimator로 작동
- Stage 2가 최고 성능 (학습과 과적합의 균형점)
- 재학습 시 40시간 추가 소요되어 미구현

### 3. 좌표 정규화

```python
# 학습: [0, 1] 정규화
normalized_coords = coords / 220.0

# 평가: 픽셀 단위로 복원
pred_pixels = pred_normalized * 220
```

---

##   평가 지표

### PDJ (Percentage of Detected Joints)

```
PDJ@0.2 = 오차 < (torso 직경 × 0.2)인 관절 비율
```

- 표준 임계값: 0.2 (torso 직경의 20%)
- 본 구현: **40.07%**

### PCP (Percentage of Correct Parts)

```
PCP@0.5 = 양 끝점이 정확한 사지 비율
```

- 표준 임계값: 0.5 (사지 길이의 50%)
- 본 구현: **32.01%**

---

##   시각화 결과

**성능 분석**:
-   우수 (5/8 샘플): 15-29px 평균 오차
-   보통 (2/8 샘플): 30-50px 오차
-   실패 (1/8 샘플): 80-90px 오차

**실패 케이스**:
- 프레임 내 여러 사람
- 극단적인 포즈
- 낮은 조명
- 가림 현상

---

##   논문과의 차이점

### 논문과 동일 
1. AlexNet 기반 아키텍처
2. 3-stage cascade 구조
3. 좌표 회귀 방식
4. FLIC 데이터셋 사용
5. 데이터 증강 전략

### 논문과 다른 점 

| 항목 | 원본 논문 | 본 구현 |
|------|-----------|---------|
| Pre-training | ImageNet | 없음 |
| Cascade | 실제 refinement | 독립 stage |
| Sub-image cropping | 있음 | 없음 |
| Learning rate | 0.0005 | 0.00005 |
| 최종 PDJ@0.2 | ~82% | **40.07%** |

---

### 환경 설정

```bash
# 의존성 설치
pip install torch torchvision numpy matplotlib scipy tqdm Pillow

# FLIC 데이터셋 다운로드 및 배치
# 경로: /content/drive/MyDrive/DeepPose_Dataset/FLIC/
```

### 학습 실행

```python
# 전체 파이프라인 실행
python deeppose_train.ipynb

# 자동으로:
# 1. FLIC 데이터셋 로드
# 2. Stage 1, 2, 3 순차 학습 (체크포인트 있으면 로드)
# 3. 전체 Stage 평가
# 4. 시각화 생성
```

### 추론

```python
from models import DeepPoseNetwork

# 모델 로드 (Stage 2 - 최고 성능)
model = DeepPoseNetwork(num_joints=10)
checkpoint = torch.load('checkpoints/Stage2_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 예측
with torch.no_grad():
    joints = model(image)  # 정규화된 좌표
    joints_pixel = joints * 220  # 픽셀 단위
```

##   개선 방향

### 단기 (높은 효과)
1. ImageNet Pre-training: +10-15% PDJ 예상
2. 실제 Cascade 구현
3. Sub-image cropping 추가
4. Learning rate 스케줄링 개선

### 장기
5. 더 많은 증강 기법
6. 더 긴 학습 (150-200 epochs)
7. 현대적 백본 (ResNet, HRNet)
8. 전신 pose (17+ joints)




