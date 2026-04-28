# 這個 Project 在做什麼?(白話說明)

寫給完全不懂的人看,所以先講白話再帶名詞。

---

## 一、Project 想回答什麼問題

「我想跑一個 AI 模型(神經網路)做推論(inference),把它擺在雲端的不同地方跑,
**誰快、誰便宜、誰安全?**」

這就是這個 project 在做的事。

更精確一點:

- AI 模型固定 = **ResNet-50**(經典的影像分類模型,給它一張照片,它告訴你是貓還是狗)
- 雲端服務商固定 = **Google Cloud (GCP)**
- 變動的是:**部署方式**(同一個模型放在不同類型的雲端服務上)
- 額外加碼:**啟用「機密運算」(Confidential Computing)後,效能會掉多少?**

---

## 二、為什麼這個問題重要

雲端不是只有「一台機器」這麼簡單。同一個模型可以塞在很多種地方跑,各有優缺點:

| 想像成餐廳 | 雲端對應 |
|---|---|
| 你包下一整家米其林餐廳,主廚每天等你 | 專用 GPU 機器(最快,最貴) |
| 你包下一家普通自助餐 | 專用 CPU 機器 |
| 你訂了一張固定座位,廚房隨叫隨上菜 | 容器服務(Cloud Run) |
| 你 24 小時隨時都能去點餐,但廚師可能還沒到 | 函式服務(Cloud Functions, serverless) |

每個模式 **延遲、吞吐量、價錢、冷啟動**都不一樣。
我們的工作就是把這四種都跑一遍,然後**用同一把尺量**:多快?多便宜?

---

## 三、四個人分四種部署方式

| 同學 | 負責的部署方式 | 比喻 |
|---|---|---|
| Aashir | GPU 機器(`n1-standard-4 + NVIDIA T4`) | 米其林主廚 |
| **Eric(我)** | CPU 機器 + 機密運算機器 | 普通自助餐 vs. 全程上鎖的私人廚房 |
| Shopnil | Cloud Run 容器 | 共享廚房固定座位 |
| Ethan | Cloud Functions(serverless) | 24 小時叫餐機 |

每個人在自己的部署上,跑**完全一樣的測試**:
- 同一個模型(ResNet-50, ONNX 格式)
- 同樣的 200 張請求圖
- 同樣的併發等級(1 / 10 / 50 / 100 個同時請求)
- 同樣的測五次取中位數

最後把四個人的結果擺在一起比較。

---

## 四、我(Eric)負責什麼

我負責的是 **「機密運算的開銷分析」(Confidential Computing Overhead)**。

具體做兩件事:

### 1. 標準 CPU 機器(`triton-cpu-standard`)

一台普通的 GCP 虛擬機(VM),跑模型推論。這同時也是整個 team 的 **CPU 基準**。
(因為 Aashir 跑 GPU、Shopnil 跑 Cloud Run、Ethan 跑 Cloud Functions,沒人跑 CPU,所以我這台也代表 team 的 CPU 數據。)

### 2. 機密運算 CPU 機器(`triton-cpu-confidential`)

**完全一樣的機器規格**,只是多開了一個東西:**AMD SEV**(機密運算)。

兩台機器跑出來的差距,就是「機密運算到底會變慢多少」。

---

## 五、什麼是「機密運算」?為什麼要研究它?

### 普通雲端機器的問題

當你把模型放上雲端,理論上雲服務商(Google、AWS)的工程師、或是入侵 hypervisor 的攻擊者,
**有可能看到記憶體裡的資料**。對一般應用沒差,但是:

- 醫療影像(病人 X 光)
- 金融資料(信用卡分析)
- 生物辨識(指紋、臉部)

這些**不能被別人看見**。法規也會要求加密。

### 機密運算的解法

**AMD SEV** 是 CPU 內建的功能:整台 VM 的記憶體**全程加密**,
就連雲服務商自己都看不到裡面的資料。
就算 hypervisor 被入侵,看到的也是亂碼。

**這對 AI 推論特別有意義**:你的模型跟資料都在加密記憶體裡跑,別人看不到。

### 但天下沒有白吃的午餐

加密 = CPU 要多做事 → **變慢**。
我的任務就是量化「到底慢多少、貴多少」。
讓未來的人決定「要安全還是要速度」時,有真實數據可以參考。

---

## 六、整個系統長怎樣

因為 Columbia 的 GCP 組織政策不允許 VM 有外網 IP,我的架構長這樣:

```
                      [我的筆電]
                          |  IAP 隧道(專用安全通道)
                          v
   ┌──────────────────────────────────────────┐
   │           GCP VPC(內部網路)             │
   │                                          │
   │   harness-runner ──┬──> triton-cpu-standard       │
   │   (跑測試的小機器)│                              │
   │                    └──> triton-cpu-confidential   │
   │                         (有 AMD SEV)              │
   │                                          │
   │            (沒有任何 VM 有外網 IP)        │
   │                                          │
   └────────────────┬─────────────────────────┘
                    │ Cloud NAT(對外連網)
                    v
              網際網路(下載 Docker image、套件)
```

**三台 VM**:
- `triton-cpu-standard` — 跑模型,標準 VM
- `triton-cpu-confidential` — 跑模型,啟用 AMD SEV
- `harness-runner` — 跑測試程式(從這裡發送請求到上面兩台)

為什麼測試程式不直接從筆電跑?
因為要走 **IAP 隧道**,中間多一段轉送,會讓測量到的延遲變不準。
所以多開一台小 VM 在同一個內部網路,直接用 internal IP 打到 Triton。

**Triton Inference Server**:NVIDIA 寫的「模型伺服器」,
你給它模型檔案,它就開個 HTTP API 等你送請求。
四個同學都用同一版 Triton,確保「不是因為框架不同才跑出不同數字」。

---

## 七、benchmark 在量什麼?

每個併發等級(1 / 10 / 50 / 100 個同時請求)跑五次,每次 200 張圖。
量這幾個指標:

| 指標 | 白話 |
|---|---|
| **p50 latency** | 50% 的請求多快回來(中位數) |
| **p95 latency** | 95% 的請求都在多少 ms 之內(尾端延遲) |
| **p99 latency** | 99% 的請求都在多少 ms 之內(最慢那一小撮) |
| **Throughput** | 每秒處理幾張圖 |
| **Error rate** | 失敗率 |
| **Cost per 1000 inferences** | 每 1000 張要花多少錢(後處理算出來) |

對機密運算來說,我特別關心:
- 標準 vs. 機密 的 p50 差幾%
- 標準 vs. 機密 的 throughput 差幾%
- 標準 vs. 機密 的 cost per 1000 差幾%

預期答案:**慢 5–15%、貴 11–22%**。
等實驗跑完才知道實際數字。

---

## 八、檔案結構

```
project2-benchmark/
├── model/                  # 把 ResNet-50 轉成 ONNX 格式的腳本
├── cloudrun/               # Shopnil 的 Cloud Run 部署
├── harness/                # 共用測試工具(四個人都用這個)
│   ├── harness.py          # 主程式:發送請求、量延遲、寫 CSV
│   └── run_all.sh          # 包裝腳本:跑 4 種併發 × 5 次 = 20 個 CSV
├── confidential/           # ← 我的部分
│   ├── deploy_vm.sh        # 開一台 Triton VM(standard 或 confidential)
│   ├── setup_vm.sh         # VM 開機後自動執行(裝 Docker、抓模型、啟動 Triton)
│   ├── deploy_runner.sh    # 開那台跑測試的小 VM
│   ├── firewall.sh         # 開防火牆規則
│   ├── cloud_nat.sh        # 開 NAT(讓沒外網 IP 的 VM 可以對外連網)
│   ├── teardown.sh         # 全部刪掉(實驗結束後省錢用)
│   ├── CLAUDE.md           # 給工程師看的詳細技術筆記
│   ├── README.md           # 給 team 看的成果報告
│   └── explanation.md      # ← 這份檔案
└── results/                # 跑出來的原始 CSV
    ├── cpu/                # 我的標準 VM 跑出來的(20 個 CSV)
    ├── confidential/       # 我的 SEV VM 跑出來的(20 個 CSV)
    ├── gpu/                # Aashir 的
    ├── cloudrun/           # Shopnil 的
    └── cloudfunction/      # Ethan 的
```

---

## 九、流程一句話

1. **架環境**(一次):開防火牆、開 NAT、設好 GCP 專案
2. **開機器**:三台 VM 都用 `--no-address`(沒有外網 IP),用我的 deploy 腳本
3. **等 VM 開機完成**:VM 裡的 startup script 會自己裝 Docker、抓模型、啟動 Triton
4. **上傳測試程式**到 runner,進去 SSH 跑 `run_all.sh`
5. **等 ~1.5 小時**:標準跑完跑機密,共產生 40 個 CSV
6. **下載 CSV** 到筆電,commit 進 team 的 repo
7. **填 README.md** 的「實際結果」表格,算出 TEE overhead
8. **拆機器**(`teardown.sh`)免得繼續燒錢

---

## 十、為什麼這個 project 對課程重要

| 課程主題 | 對應 |
|---|---|
| 雲端運算服務模型 | 我們同時碰到 IaaS(VM)、CaaS(Cloud Run)、FaaS(Cloud Functions) |
| 深度神經網路部署 | 真的把 ResNet-50 跑在不同硬體上,不只是紙上談兵 |
| 效能分析 | 用受控實驗(同模型、同資料、同 framework)做四向比較 |
| 安全 vs. 效能 trade-off | 我這部分量化「機密運算到底要付多少代價」 |

最終的產出是一份報告 + 一個可重現的 benchmarking 框架,
讓未來的人選部署方式時,有實際數據可以參考。

---

## 名詞快速對照(完全不懂者用)

- **VM (Virtual Machine)** = 雲端的一台「虛擬電腦」,你租了用
- **CPU / GPU** = 中央處理器 / 顯示卡。GPU 跑 AI 快很多但很貴
- **Container / Docker** = 把程式打包成一個獨立的「盒子」,放到哪都能跑
- **Triton** = NVIDIA 寫的模型伺服器(這個盒子裡跑的程式)
- **ONNX** = 通用模型格式(像「PDF for AI 模型」)
- **ResNet-50** = 一個 50 層的卷積神經網路,做影像分類
- **inference / 推論** = 把訓練好的模型拿來預測新資料(不是訓練)
- **latency / 延遲** = 從發出請求到收到回應的時間
- **throughput / 吞吐量** = 每秒能處理幾個請求
- **concurrency / 併發** = 同時有幾個人在發請求
- **TEE (Trusted Execution Environment)** = 可信執行環境(機密運算的學術名)
- **AMD SEV** = AMD 的 TEE 技術(透過硬體加密 VM 記憶體)
- **IAP (Identity-Aware Proxy)** = Google 的安全代理,讓你不靠外網 IP 就能 SSH
- **NAT (Network Address Translation)** = 網路位址轉換,讓沒外網 IP 的 VM 也能對外連網
