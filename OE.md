# Outlier Exposure(OE) 기반 Vision Classification OOD Detection 구현 정리 (ICLR 2019)

* 참고 논문: 

---

## 1) 문제 설정

* **In-distribution (ID)**: 학습(및 정상 추론)에서 기대하는 데이터 분포 (D_{\text{in}})
* **Out-of-distribution (OOD)**: 배포 환경에서 들어올 수 있는 비정상/미지 분포 (D_{\text{out}})
* 핵심 가정: 실제 배포에서는 (D_{\text{out}})이 **사전에 정확히 알려지기 어렵다**. 그래서 **보조(outlier) 데이터셋 (D^{OE}_{\text{out}})** 를 활용해 “OOD 힌트”를 학습한다. 

---

## 2) 기본 분류기와 OOD 스코어 (MSP baseline)

* 분류기 (f: \mathcal{X} \rightarrow \mathbb{R}^k) 는 softmax posterior (f(x))를 출력(합이 1인 확률 벡터) 
* **MSP(Maximum Softmax Probability) 기반 OOD 스코어**:
  [
  s_{\text{MSP}}(x) = -\max_{c} f_c(x)
  ]
  (\max f_c(x))가 작을수록(= 자신감이 낮을수록) OOD로 보겠다는 단순 기준 

---

## 3) Outlier Exposure(OE)의 핵심 아이디어(일반형)

* 모델의 원래 목적함수 (L)에, **OE 데이터**에 대한 보조 손실 (L_{OE})를 가중합으로 추가:
  [
  \mathbb{E}*{(x,y)\sim D*{\text{in}}}\Big[L(f(x),y) + \lambda , \mathbb{E}*{x'\sim D^{OE}*{\text{out}}}[L_{OE}(f(x'), f(x), y)]\Big]
  ]

* (D^{OE}_{\text{out}})은 테스트 시 마주칠 OOD 분포와 **서로 겹치지 않게(disjoint)** 구성하는 것을 전제로 설명한다. 

---

## 4) Vision 분류(MSP 기반)에서의 OE 손실 설계

논문에서 제시한 가장 구현이 쉬운 형태:

### (1) ID 분류 손실

* 표준 cross-entropy:
  [
  L_{\text{ID}} = \mathbb{E}*{(x,y)\sim D*{\text{in}}}[-\log f_y(x)]
  ]

### (2) OE 손실 (Uniform으로 “눌러주기”)

* OOD(=OE) 샘플 (x')에 대해 posterior가 **균일분포 (U)** 에 가깝게 나오도록 유도:
  [
  L_{\text{OE}} = \mathbb{E}*{x'\sim D^{OE}*{\text{out}}}\big[H(U; f(x'))\big]
  ]
* 최종:
  [
  L_{\text{total}} = L_{\text{ID}} + \lambda L_{\text{OE}}
  ]


### (3) 권장 (\lambda)

* Vision 실험에서는 (\lambda=0.5)를 사용 

---

## 5) 학습 프로토콜(구현 관점)

“사전학습 후 OE 파인튜닝” (가장 현실적인 파이프라인)

1. (D_{\text{in}})으로 분류기 (f)를 일반 방식으로 학습
2. 학습된 (f)를 **OE를 포함한 손실**로 추가 학습(파인튜닝)

   * 논문 비전 실험: Wide ResNet 학습 후 **OE로 10 epoch 파인튜닝** 
   * CIFAR 계열 세팅 예시(학습 디테일): cosine LR, 파인튜닝 LR 0.001 등 



## 6) 배치 구성(데이터 로더 설계)

실무 구현에서는 아래가 가장 단순합니다.

* 각 iteration마다

  * ID 미니배치 ((x_{\text{in}}, y_{\text{in}})\sim D_{\text{in}})
  * OE 미니배치 (x_{\text{oe}}\sim D^{OE}_{\text{out}}) (라벨 불필요)
* 동일 네트워크 (f)로 forward 후

  * `loss_id = CE(f(x_in), y_in)`
  * `loss_oe = CE(f(x_oe), uniform_target)`  (즉 (H(U; f(x_{oe}))))
  * `loss = loss_id + λ * loss_oe`
* 역전파/업데이트

---

## 7) 추론 시 OOD 스코어링 (2가지 선택지)

### 선택지 1: MSP 스코어 그대로 사용

[
s(x)=-\max_c f_c(x)
]
(베이스라인/호환성 좋음) 

### 선택지 2: OE 학습 목적과 정합적인 스코어(권장)

* 논문에서는 **(-H(U; f(x)))** 가 MSP보다 더 좋은 경우가 많다고 보고 

  * 직관: “최대값만” 보는 MSP 대신, **분포 전체의 균일성**을 반영

---

## 8) 평가 지표(논문 기준)

* OOD를 **positive class**로 두고,

  * **AUROC**
  * **AUPR**
  * **FPR@TPR(N%)**(예: FPR95)
    를 사용 

---

## 9) 최소 의사코드(Pytorch 스타일)

```python
# model: f(x) -> logits (B, K)
# softmax 후 posterior를 사용한다고 가정
# lambda_oe: 0.5 (vision 권장)  # :contentReference[oaicite:15]{index=15}

for (x_in, y_in), x_oe in zip(loader_in, loader_oe):
    logits_in = model(x_in)           # (B, K)
    logits_oe = model(x_oe)           # (B2, K)

    loss_id = cross_entropy(logits_in, y_in)

    # uniform target: 각 샘플에 대해 (1/K,...,1/K)
    logp_oe = log_softmax(logits_oe, dim=1)
    loss_oe = -logp_oe.mean(dim=1).mean()   # H(U; f(x_oe)) 와 동치(구현 트릭)

    loss = loss_id + lambda_oe * loss_oe

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---


제공해주신 논문(Outlier Exposure, ICLR 2019)에 기반하여, **Vision Classification** 태스크를 위한 **Outlier Exposure (OE)** 구현 방법을 정리해 드립니다.

이 방법은 모델이 학습 데이터()에 대해서는 정확한 분류를 수행하고, 알려진 이상치 데이터()에 대해서는 '불확실함(Uniform Distribution)'을 예측하도록 학습시키는 방식입니다.

---

# Outlier Exposure (OE) 구현 가이드 - Vision Classification

## 1. 핵심 손실 함수 (Loss Function)

OE의 핵심은 기존 분류 손실 함수에 **OE Loss** 항을 추가하는 것입니다. 전체 목적 함수는 다음과 같이 정의됩니다.

Vision Multiclass Classification 문제에서 구체적인 수식은 다음과 같습니다.

### 1.1. In-distribution Loss (기존 분류 손실)

모델이 원본 학습 데이터()를 올바르게 분류하도록 하는 표준 Cross-Entropy 손실입니다.

### 1.2. Outlier Exposure Loss (OE 손실)

모델이 이상치 데이터()를 입력받았을 때, 특정 클래스로 예측하지 않고 모든 클래스에 대해 균등한 확률(Uniform Distribution)을 내뱉도록 강제하는항입니다. 이는 Outlier 입력 $x'$에 대한 출력 분포와 Uniform Distribution  간의 Cross-Entropy로 정의됩니다.

여기서 $H(\mathcal{U}; f(x'))$는 균등 분포와 예측 분포 간의 Cross Entropy입니다.

### 1.3. 최종 최적화 목표

Vision 실험에서 권장되는  값은 **0.5**입니다.

---

## 2. 데이터셋 구성 ()

성공적인 OE 학습을 위해서는 보조 데이터셋()의 선택이 중요합니다.

* **데이터 소스:**
실제(realistic) 이미지 데이터셋을 사용하는 것이 가우시안 노이즈나 생성된(GAN) 데이터보다 효과적입니다.


* $\mathcal{D}_{in}$이 CIFAR-10/100, SVHN인 경우: **80 Million Tiny Images** 데이터셋을 주로 사용합니다.


* $\mathcal{D}_{in}$이 Tiny ImageNet, Places365인 경우: **ImageNet-22K** (ImageNet-1K 제외) 데이터셋을 사용합니다.




* **주의사항 (Disjoint):**


)와 겹치지 않아야 합니다. 즉, 학습에 사용한 Outlier 데이터를 테스트용 Outlier로 재사용하지 않도록 주의해야 합니다.



---

## 3. 학습 세부 사항 (Training Details)

논문에서 Vision 태스크(Wide ResNet 등)에 적용한 구체적인 설정입니다.

* **파인 튜닝 (Fine-tuning):**
이미 학습된 모델이 있다면, OE Loss를 추가하여 약 **10 epoch** 동안 파인 튜닝을 진행합니다.


* **처음부터 학습 (Training from Scratch):**
처음부터 OE Loss를 포함하여 학습하면 파인 튜닝보다 더 나은 성능과 Calibration 효과를 얻을 수 있습니다.


* **배치 구성:**
각 학습 단계(iteration)마다 In-distribution 데이터 배치와 OE 데이터 배치를 함께 로드하여 손실을 계산합니다. OE 데이터셋의 크기가 크더라도 $\mathcal{D}_{in}$의 epoch 길이에 맞춰 학습을 진행합니다.



---

## 4. 추론 및 이상치 탐지 (Inference & Scoring)

학습된 모델을 사용하여 테스트 시 입력 이미지가 이상치인지 판단하는 점수(OOD Score)를 계산하는 방법입니다.

* **방법 1: MSP (Maximum Softmax Probability):**
Softmax 출력의 최댓값을 사용합니다. (기본 베이스라인) .




* **방법 2: Uniform 분포와의 거리 (권장):**
논문의 부록 E에 따르면, 단순 MSP보다 **Uniform 분포와의 Cross Entropy** (또는 )를 점수로 사용하는 것이 더 높은 성능을 보일 수 있습니다.





(이 값이 작을수록, 즉 Uniform에 가까울수록 이상치일 확률이 높음)

---

## 5. PyTorch 구현 예시 코드

```python
import torch
import torch.nn.functional as F

# Hyperparameter from the paper (Section 4.3)
lambda_oe = 0.5

def loss_function(model, x_in, y_in, x_out):
    """
    x_in: In-distribution images (batch_size, C, H, W)
    y_in: In-distribution labels (batch_size)
    x_out: Outlier Exposure images (batch_size, C, H, W)
    """
    
    # 1. In-distribution forward pass
    logits_in = model(x_in)
    
    # Standard Cross Entropy Loss for classification
    # source: [157] E_in[-log f_y(x)]
    loss_in = F.cross_entropy(logits_in, y_in)
    
    # 2. Outlier Exposure forward pass
    logits_out = model(x_out)
    
    # Softmax probabilities for outliers
    softmax_out = F.softmax(logits_out, dim=1)
    
    # OE Loss: Cross Entropy to Uniform Distribution
    # The target is 1/num_classes for all classes
    # source: [157] E_out[H(U; f(x'))]
    # mean() takes the expectation over the batch
    loss_oe = - (softmax_out.mean(1) - torch.logsumexp(logits_out, dim=1)).mean()
    # Or simply: - (1/num_classes) * sum(log_softmax) which is equivalent to CE with uniform target
    # Implementation shortcut:
    # loss_oe = -(softmax_out * torch.log_softmax(logits_out, dim=1)).sum(1).mean() -> This is Entropy minimization
    # The paper says "Cross Entropy to Uniform".
    # Since Uniform distribution is constant, minimizing CE is equivalent to maximizing entropy 
    # BUT the paper explicitly mentions matching uniform distribution.
    # Effectively: loss_oe = torch.mean(-torch.sum(torch.full_like(softmax_out, 1/num_classes) * torch.log_softmax(logits_out, dim=1), dim=1))
    
    loss_oe = torch.mean(
        -torch.sum(
            (1.0 / logits_out.size(1)) * F.log_softmax(logits_out, dim=1), 
            dim=1
        )
    )

    # 3. Total Loss
    total_loss = loss_in + lambda_oe * loss_oe
    
    return total_loss

```

### 요약

1. **데이터:** 타겟 데이터()와 겹치지 않는 대규모 이미지 데이터()를 준비합니다.
2. **Loss:** $\mathcal{D}*{in}$에는 CrossEntropy, $\mathcal{D}*{out}^{OE}$에는 "CrossEntropy to Uniform"을 적용합니다.
3. **가중치:** 를 사용합니다.
4. **평가:** 테스트 시 입력에 대한 Softmax 분포가 Uniform에 가까울수록(또는 Max prob가 낮을수록) 이상치로 간주합니다.