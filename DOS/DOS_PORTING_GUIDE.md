## DOS 이식 가이드라인 (이 레포 기준 → 다른 코드베이스로 포팅)

이 문서는 이 레포의 **`train_diverse.py` 구현을 기준으로**, DOS(Diverse Outlier Sampling)를 다른 학습 코드베이스에 **이식 가능한 모듈 경계**로 정리한 가이드입니다.

---

### 1) 먼저 “이식 포인트”부터 정하기 (필수 체크리스트)

- **모델이 penultimate feature를 뽑을 수 있어야 함**
  - 이 레포는 `forward(x, ret_feat=True) -> (logits, feats)` 형태입니다.

```95:107:models/wide_resnet.py
    def forward(self, x, ret_feat=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, self.pool_size)
        out = out.view(out.size(0), -1)
        logit = self.linear(out)
        
        if ret_feat:
            return logit, out
        return logit
```

- **학습 중 ID 배치와 OOD 후보 배치를 “동시에” 받을 수 있어야 함**
  - 구현은 `zip(train_loader_id, train_loader_candidate_ood)`로 묶습니다.
- **OOD 학습 목표를 선택해야 함**
  - 이 레포의 DOS 구현은 기본적으로 **Absent class(K+1번째 클래스)** 학습을 사용합니다(= OOD 샘플 레이블을 `num_classes`로 둠).
  - 만약 다른 코드베이스가 “K+1 클래스를 추가하기 어렵다”면, DOS의 “선택(sampling)”만 가져오고 OOD 손실은 Energy/OE 등 기존 방식으로 연결하는 식으로 변형할 수 있습니다(단, 논문 기본 설정은 absent-class 기반).

---

### 2) DOS가 요구하는 최소 인터페이스(다른 코드베이스에 맞춰 매핑)

- **입력**
  - **ID 배치**: `x_id: Tensor[B, ...], y_id: LongTensor[B]`
  - **OOD 후보 배치**: `x_ood_can: Tensor[B_can, ...]` (라벨 없어도 됨)
  - **모델**: `model(x, ret_feat=True)` 혹은 동일 의미의 훅
- **출력**
  - **선택된 OOD 배치**: `x_ood_sel: Tensor[B_ood, ...]` (보통 `B_ood == B`)
  - **학습 손실**: `L = CE_ID + beta * CE_OOD(absent_class)`

---

### 3) “DOS 선택 로직”만 떼서 이식하는 방법 (핵심)

이 레포에서 OOD 후보 배치에서 선택하는 핵심 블록은 아래와 같습니다:

```143:205:train_diverse.py
        for sample_id, sample_ood in zip(train_loader_id, train_loader_candidate_ood):

            # select ood in batch
            clf.eval()
            data_batch_candidate_ood = sample_ood['data'].cuda()

            with torch.no_grad():
                logits_batch_candidate_ood, feats_batch_candidate_ood = clf(data_batch_candidate_ood, ret_feat=True)

            prob_ood = torch.softmax(logits_batch_candidate_ood, dim=1)
            weights_batch_candidate_ood = np.array(prob_ood[:, -1].tolist())
            idxs_sorted = np.argsort(weights_batch_candidate_ood)

            # normalize
            repr_batch_candidate_ood = np.array(F.normalize(feats_batch_candidate_ood.cpu(), dim=-1))

            # clustering
            k = args.num_cluster
            kmeans = KMeans(n_clusters=args.num_cluster, n_init=args.n_init).fit(repr_batch_candidate_ood)
            clus_candidate_ood = kmeans.labels_
            ...
            data_ood = data_batch_candidate_ood[idxs_sampled]
            ...
            loss = F.cross_entropy(logit[:num_id], target_id)
            loss += args.beta * F.cross_entropy(logit[num_id:], target_ood)
```

이걸 다른 코드베이스에 이식할 때는 아래처럼 **3개 함수/모듈**로 쪼개면 가장 안전합니다.

- **(A) feature/logit 추출 어댑터**
  - `extract_logits_and_feats(model, x) -> (logits, feats)`
  - 모델이 `ret_feat`를 지원하지 않으면, “penultimate layer forward hook”으로 feats를 뽑는 어댑터를 만들면 됩니다.

- **(B) DOS sampler (선택기)**
  - 입력: `(logits_can, feats_can, num_select=B_ood, k=num_cluster)`
  - 처리:
    - `p_abs = softmax(logits_can)[:, -1]` (absent class 확률)
    - **hard OOD 정의**: `p_abs`가 낮을수록(= OOD라고 확신 못할수록) 더 hard
    - `feats_norm = L2_normalize(feats_can)`
    - KMeans로 클러스터 라벨 획득
    - 클러스터마다 `p_abs`가 낮은 샘플부터 `sampled_cluster_size`개 선택
    - 부족분은 “미선택 pool”에서 랜덤으로 채움(이 레포 구현 그대로)

- **(C) 학습 손실 결합**
  - absent-class 방식이면:
    - 분류기 출력 차원 = `K + 1`
    - OOD 타겟 = `K` (0-index 기준)
    - `loss = CE(logits_id, y_id) + beta * CE(logits_ood, y_ood)`

---

### 4) 하이퍼파라미터(이 레포 구현 기준) 이식 시 주의점

- **`num_cluster`**: 보통 ID 배치 크기와 동일(예: 64)
- **`size_candidate_ood`**: epoch마다 OOD 전체에서 “후보 pool”을 랜덤 샘플링(예: 300k)
- **`batch_size_candidate_ood` 계산 방식**
  - 이 레포는 ID iter 수와 맞추기 위해 아래처럼 “비율로” 맞춥니다.
  - 다른 코드베이스에서도 **ID step마다 후보 OOD batch가 반드시 공급**되게(= zip이 끊기지 않게) 맞추는 게 중요합니다.
- **KMeans 성능**
  - 이 구현은 sklearn KMeans + numpy 변환이라 CPU 병목이 날 수 있습니다.
  - 타 코드베이스로 옮길 때는 (필요 시) **Faiss KMeans / mini-batch kmeans / GPU clustering**으로 대체하면 됩니다(알고리즘 핵심은 “정규화 feature 공간에서 diversity 확보”).

---

### 5) “Absent class를 못 쓰는 코드베이스”에서의 최소 이식(변형)

- **가능한 최소 단위**: “클러스터링 기반 diverse + hard OOD 선택”만 이식
- 그 다음 학습은 기존 코드베이스의 OOD 목적함수(Energy/OE/Uni 등)에 `x_ood_sel`을 넣는 방식
  - 단, 이 레포의 DOS 기본 흐름은 absent-class 기반이므로, 논문과 동일 비교가 필요하면 K+1 head를 추가하는 편이 가장 깔끔합니다.

