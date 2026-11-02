# ML 2026 Final：SEM 泡孔大小分割（助教試做版）

這個資料夾給助教**以學生的身分試做一次期末專題**：預測結果可以用 `evaluate.py` 在本機評分，也可以轉成 CSV 上傳到 Kaggle（見第 3 節）。
有任何不清楚、不合理或太難、太簡單的地方，請回報給負責的助教。

> ⚠️ `answers/` 是測試集答案（含 private），**只限助教使用，不可外流給學生**。
> 試做時請把它當作不存在，只在評分時讓 `evaluate.py` 讀取。

---

## 1. 任務目標

輸入一張泡棉材料的 SEM 灰階影像 tile（384×384），對每個像素輸出它的類別：

| 值 | 類別 | 定義 |
|---|---|---|
| 0 | wall（牆） | 孔洞之間的材料 |
| 1 | small（小孔） | 該孔洞（cell）面積 < 6000 像素 |
| 2 | medium（中孔） | 6000 ≤ 面積 < 14000 像素 |
| 3 | large（大孔） | 面積 ≥ 14000 像素 |

也就是說，模型要做兩件事：
1. **把孔洞一顆顆分割出來**：分清楚牆和孔洞，相鄰的孔洞不能黏在一起。
2. **依每顆孔洞的大小分類**：同一顆孔洞的所有像素都屬於同一類。

**面積的定義**：孔洞的面積是**整顆孔洞在原始大圖上的面積**，不是它在 tile 裡看得到的面積。
- 測試 tile 是從較大的原圖切下來的。被 tile 邊切到的孔洞，要從看得到的部分推斷整顆有多大。
- 碰到**原圖**邊緣的孔洞，整顆面積無法確定，標成 ignore，不計分。例外：看得到的部分已經 ≥ 14000 像素的，直接算大孔。
- 孔洞以 4 連通計算，小於 20 像素的碎片不算孔洞。

---

## 2. 資料

```
Final_TAtest/
├── README.md            本說明
├── data/
│   ├── train/
│   │   ├── images/      78 張訓練原圖（灰階，SEM 資訊列已裁掉）
│   │   ├── labels/      對應的 label
│   │   └── preview/     label 疊在影像上的彩色預覽，方便目視
│   ├── unlabeled/       366 張沒有 label 的 SEM 影像（可用於自監督、偽標註等）
│   └── test/images/     113 個測試 tile（t001.png ~ t113.png），384×384
├── answers/             ⚠️ 只限助教：測試集答案
│   ├── labels/          測試 tile 的答案（格式同 train/labels）
│   └── test_split.csv   每個 tile 屬於 public 或 private
├── sample_submission.csv
├── make_submission.py   預測 PNG → submission.csv
├── evaluate.py          評分程式
└── rle.py               submission 使用的 RLE 編碼
```

### 2.1 訓練集（`data/train/`）
- 78 張整張的原圖，寬 1024 或 1280、高 839～960，灰階 PNG。
- **label 的像素值就是類別**：0 = wall、1 = small、2 = medium、3 = large，和要提交的值相同；**255 = ignore**（碰到圖片邊緣、整顆面積不確定的孔洞）。
  - 訓練時應把 ignore 排除在 loss 之外，例如 PyTorch 的 `nn.CrossEntropyLoss(ignore_index=255)`。
  - 值只有 0～3，用一般看圖軟體打開幾乎是全黑，要目視請看 `preview/`。
- 包含開放式（open-cell）和封閉式（closed-cell）兩種泡棉。開放式泡棉的答案畫的是外層的孔洞輪廓，不是內部的小窗口。

### 2.2 測試集（`data/test/images/`）
- 113 個 384×384 的 tile，從 22 張原圖切出（1280 寬的原圖切 2×3 個，1024 寬的切 2×2 個，tile 之間留 30～50 像素間隔）。
- 每個 tile 做過隨機 90° 旋轉或翻轉；編號已打散，同一張原圖的 tile 編號不連續。
- **public 57 個、private 56 個**，混在一起不告訴學生哪個是哪個。Kaggle 上排行榜只顯示 public 的分數，最後成績看 private。
- 測試原圖不在訓練集或 unlabeled 中（已用影像比對確認，包含裁切、縮放、旋轉過的副本）。

### 2.3 unlabeled（`data/unlabeled/`）
- 366 張 SEM 影像，沒有 label。
- 注意：這些是 **RGB 格式、尺寸不一、底部仍帶有 SEM 資訊列**，和訓練圖的格式不同，使用前要自己轉灰階、裁掉資訊列。

---

## 3. 提交方式（轉成 CSV、上傳 Kaggle）

Kaggle 不收 PNG，只收一個 `submission.csv`。流程是：**預測存成 PNG → 用 `make_submission.py` 轉成 CSV → 上傳 Kaggle**。

### 3.1 把預測存成 PNG

每個測試 tile 存一張 PNG，放在同一個資料夾（下面稱 `PRED_DIR`）：
- **檔名和測試圖相同**：`t001.png` ～ `t113.png`，113 張都要有。
- **尺寸 384×384**（和 tile 相同；如果模型有縮放輸入，記得把輸出縮放回 384×384，用最近鄰插值）。
- **單通道、像素值 0～3**（0 = wall、1 = small、2 = medium、3 = large），和訓練 label 的值相同。預測裡不能有 255（ignore 只出現在答案裡）。
- 不要存成 JPG（壓縮會改變像素值）。

```python
import numpy as np
from PIL import Image

# pred: (384, 384) 的整數陣列，值為 0~3
Image.fromarray(pred.astype(np.uint8)).save(f"PRED_DIR/{tile_id}.png")
```

### 3.2 轉成 submission.csv

在這個資料夾下執行：
```bash
python make_submission.py PRED_DIR -o submission.csv
```
程式會檢查每個 tile 都有預測、尺寸正確、像素值合法，再輸出 `submission.csv`（339 列 = 113 個 tile × 3 類）。

如果想在自己的程式裡直接寫 CSV，也可以用 `rle.py`：
```python
from rle import rle_encode
rows = [(f"{tile_id}_{c}", rle_encode(pred == c)) for c in (1, 2, 3)]
```

### 3.3 上傳到 Kaggle

1. 進入競賽頁面 → **Submit Prediction**，上傳 `submission.csv`。
2. 幾秒到幾分鐘後會出現 public 分數。格式有錯時會直接顯示錯誤訊息（例如缺少某個 id、RLE 格式錯誤）。
3. 比賽結束前，到 **My Submissions** 勾選最多 2 份作為最終提交，以 private 分數排名。

上傳前可以先用 `python evaluate.py submission.csv` 在本機確認（見第 5 節），分數會和 Kaggle 上的 public 分數一樣。

### 3.4 CSV 格式說明（用 `make_submission.py` 就不需要自己處理）

每個 tile 三列，分別對應小、中、大孔：

```
id,rle
t001_1,<小孔像素的 RLE>
t001_2,<中孔像素的 RLE>
t001_3,<大孔像素的 RLE>
t002_1,...
```

- **RLE 編碼**：像素依「由左到右、由上到下」編號，從 1 開始；一段連續像素寫成 `起點 長度`，多段用空格接起來。例如 `5 3 20 2` 表示第 5、6、7、20、21 個像素。
- 該類沒有任何像素就寫 `-`（不能留空，Kaggle 會當成缺值拒收）。
- 三個類別都沒有涵蓋的像素視為 wall。
- **同一個像素不能同時屬於兩個類別**，否則評分會報錯。
- `sample_submission.csv` 是格式範例（全部 `-` = 全部預測成 wall，分數 0）。

---

## 4. 評分方式

**每個 tile 的分數**：小、中、大三類的 IoU 平均

$$\text{score}_\text{tile} = \frac{1}{|C|}\sum_{c \in C} \frac{|P_c \cap G_c|}{|P_c \cup G_c|}$$

$P_c$、$G_c$ 是預測和答案中屬於類別 $c$ 的像素，$C$ 是該 tile 的答案或預測中**至少一邊有出現**的類別。

1. **先拿掉 ignore 像素**：答案是 ignore 的位置，預測什麼都不算。
2. **wall 不是計分類別，但會間接影響分數**：把牆預測成孔洞，會讓該類的聯集變大；把孔洞預測成牆，等於漏掉。所以牆還是要畫對。
3. 某一類在答案和預測中都沒有，就跳過；只出現在一邊，該類 IoU = 0。三類都沒有的 tile 算 1 分。
4. **最終分數 = 所有 tile 分數的平均**（每個 tile 權重相同）。public、private 分開計算。

---

## 5. 評分程式的使用方式

需要 Python 3 與 `numpy`、`pillow`。在這個資料夾下執行：

```bash
# 1. 預測 PNG 資料夾直接評分
python evaluate.py PRED_DIR

# 2. 評分要上傳 Kaggle 的 submission.csv（建議上傳前先跑一次，確認格式沒問題）
python evaluate.py submission.csv
```

輸出範例：
```
public  0.5159  (57 tiles)
private 0.5058  (56 tiles)
all     0.5109
pooled IoU per class: wall=0.601  small=0.539  medium=0.489  large=0.584
```
- `public` / `private` / `all`：正式分數（tile 平均）。
- `pooled IoU per class`：把所有 tile 的像素合併後各類的 IoU，只是幫助分析哪一類比較差，不是正式分數。

`evaluate.py` 的分數和 Kaggle 上的評分程式完全相同（已用 baseline 的預測交叉驗證過）；差別只在本機會同時顯示 private，Kaggle 比賽期間只顯示 public。

---

## 6. 參考分數

助教已跑過的 baseline，方法和超參數都只用訓練集切出的 validation 選擇：

| baseline | public | private | all |
|---|---|---|---|
| 傳統方法（Otsu 二值化 + watershed + 依面積分類） | 0.272 | 0.273 | 0.273 |
| U-Net | 0.495 | 0.481 | 0.488 |
| ResNet34 encoder + U-Net decoder | 0.524 | 0.498 | 0.511 |

若切割完全正確、只剩分類，上限約 0.64～0.70。目前最大的瓶頸是相鄰孔洞黏在一起。

---

## 7. 試做時希望大家回報的事

- 任務說明、資料格式、提交格式有沒有看不懂或有歧義的地方。
- 評分程式有沒有錯誤或不合理的結果。
- 難度是否合適：一般學生能不能在期限內做出高於傳統 baseline 的結果？
- 資料本身的問題：label 明顯錯誤的圖、格式不一致的檔案等。
- 自己的分數和用的方法，方便我們訂出分數的級距。
