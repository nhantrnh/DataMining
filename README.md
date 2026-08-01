# Khai thác Dữ liệu lớn (MTH055) — Đồ án môn học

Tái thực nghiệm bài báo **Roverato, A. & Nguyen, D. N. (2024), "Exploration of
the Search Space of Gaussian Graphical Models for Paired Data", Journal of
Machine Learning Research 25(92), 1–41.**

Bài báo nghiên cứu bài toán chọn mô hình đồ thị Gauss cho **dữ liệu ghép cặp**
(paired data): tập biến được chia thành hai nhóm tương ứng nhau (hai bán cầu
não, hai thời điểm đo, hai cá thể song sinh…), và mô hình cần thể hiện được
các ràng buộc đối xứng giữa hai nhóm. Đóng góp chính là **twin lattice** —
một dàn nhỏ hơn dàn bao hàm mô hình thông thường — cùng thuật toán loại bỏ lùi
từng bước **nhất quán** (coherent stepwise backward elimination) duyệt trên dàn
đó.

Đồ án thực hiện theo **Hướng 1 (Trình bày lại, tái thực nghiệm)** của đề bài:
trình bày lại phương pháp và kiểm chứng kết quả bằng một bản cài đặt độc lập.

## Nội dung kho mã nguồn

Mã R gốc của nhóm tác giả nằm ở
<https://github.com/NgocDung-NGUYEN/backwardCGM-PD>. Kho này chứa **bản chuyển
đổi sang Python** của mã đó, cộng thêm hai thực nghiệm mở rộng do nhóm viết.

```
src/backward_cgm_pd/     thư viện lõi (bản port từ R)
  graph.py                 biểu diễn đồ thị tô màu pdCG, phép toán tau
  rcon.py                  khớp mô hình RCON (backend MLE và gRc::rcox IPMS)
  search_tau.py            Thuật toán 1 — loại bỏ lùi trên twin lattice
  search_submodel.py       loại bỏ lùi trên dàn bao hàm mô hình
  pdglasso.py              pdRCON graphical lasso (dùng làm baseline)
  metrics.py               các độ đo phục hồi ePPV/eTPR/eTNR, sPPV/sTPR/sTNR
  article_graphs.py        dựng lại đồ thị thật của hai kịch bản mô phỏng
  io.py, plotting.py, r_random.py

experiments/
  simulation.py            tái thực nghiệm mô phỏng của bài báo (Bảng 2, 3)
  air_quality.py           tái thực nghiệm dữ liệu chất lượng không khí
  fmri.py                  dựng lại hình minh hoạ fMRI
  paper_table.py           số liệu đối chiếu trích từ bài báo
  baseline_simulation.py   [mở rộng] hai baseline pdglasso trên dữ liệu mô phỏng
  stability.py             [mở rộng] đánh giá độ ổn định bằng bootstrap

data/simulated-data/     tám tệp .RData dữ liệu mô phỏng của nhóm tác giả
notebooks/               chín notebook Kaggle đã chạy, kèm log
```

Kho không chứa sẵn kết quả thực nghiệm: mọi con số trong báo cáo đều sinh lại
được bằng đúng các lệnh ở mục [Chạy thực nghiệm](#chạy-thực-nghiệm) bên dưới,
ghi vào thư mục `results/`. Riêng log đầy đủ của các lần chạy đã dùng để viết
báo cáo được lưu kèm trong `notebooks/`.

Hai tệp trong `experiments/` có nhãn `[mở rộng]` là phần **nhóm viết thêm**,
không có trong mã nguồn gốc; lý do và kết quả được trình bày ở Chương 4 của
báo cáo. Toàn bộ phần còn lại là bản chuyển đổi trung thực từ R sang Python,
giữ nguyên quy ước đánh chỉ số từ 1 để kết quả so sánh trực tiếp được với các
tệp `.RData` mà nhóm tác giả công bố.

## Chạy trên Kaggle

Toàn bộ số liệu trong báo cáo được sinh ra từ chín notebook trong thư mục
`notebooks/`, chạy trên Kaggle. Mỗi notebook là một job độc lập cho một cấu
hình, kèm log đầy đủ của lần chạy cuối.

### Cấu trúc mỗi thư mục notebook

- `<tên>.ipynb` — notebook chạy được trên Kaggle, chứa toàn bộ pipeline của một
  cấu hình.
- `<tên>.log` — log console đầy đủ của lần chạy cuối.
- `kernel-metadata.json` — cấu hình Kaggle của notebook (dataset đầu vào, GPU,
  Docker image).
- `<tên>-results/` — kết quả của lần chạy đó: tệp JSON trạng thái, CSV tóm tắt
  và các hình đã vẽ.

### Các bước chạy

**B1.** Import notebook cần chạy vào Kaggle, ví dụ
`notebooks/kaggle-simulation-a-p8/kaggle-simulation-a-p8.ipynb`. Nếu chạy ở
local thì bỏ qua bước này và dùng các lệnh ở mục
[Chạy thực nghiệm](#chạy-thực-nghiệm) bên dưới.

**B2.** Thêm một input dataset duy nhất:
<https://www.kaggle.com/datasets/trungquanghuynh/backwardcgm-pd>

Dataset này chứa bản chuyển đổi Python cùng các tệp `.RData` của nhóm tác giả.
Cả chín notebook đều dùng chung đúng dataset này.

**B3.** Giữ Accelerator ở mức **None** và bật **Internet**, rồi start session.
Thuật toán chạy trên CPU nên GPU không giúp tăng tốc; bước nặng nhất là khớp mô
hình RCON bằng đại số tuyến tính trên ma trận nhỏ.

**B4.** Chạy lần lượt các cell từ trên xuống.

### Lưu ý

- **Chọn notebook theo cấu hình.** Tám notebook `kaggle-simulation-*` ứng với
  hai kịch bản A/B và bốn giá trị `p`. Với `p = 8, 12`, notebook dùng backend
  `grc_ipms` để tái tạo đúng cách khớp mô hình của mã R; với `p = 16, 20`,
  notebook dùng backend `mle` nhanh hơn. Vì vậy **không so sánh trực tiếp thời
  gian chạy giữa hai nhóm cấu hình này** — điểm được nêu rõ trong báo cáo.

- **Checkpoint và resume.** Mỗi notebook ghi trạng thái sau từng replicate và
  chỉ resume đúng checkpoint có cùng `rcon_backend`. Nếu session Kaggle hết giờ
  giữa chừng, hãy **Save Version**, rồi ở lần chạy sau **Add Input** chính output
  của version trước để tiếp tục từ chỗ đã dừng thay vì chạy lại từ đầu.

- **Thời gian chạy.** Cấu hình `p = 20` mất khoảng vài giờ. Nếu chỉ muốn kiểm
  tra pipeline hoạt động, hãy bắt đầu với `kaggle-simulation-a-p8` (khoảng mười
  phút) hoặc giảm số replicate trong cell cấu hình.

- **Chạy ở local.** Notebook dùng các đường dẫn `/kaggle/input` và
  `/kaggle/working`, nên khi chạy ngoài Kaggle cần điều chỉnh lại đường dẫn, hoặc
  đơn giản hơn là gọi thẳng các script trong `experiments/` theo hướng dẫn dưới
  đây.

## Cài đặt để chạy ở local

Yêu cầu Python từ 3.11 trở lên.

```bash
git clone git@github.com:nhantrnh/DataMining.git
cd DataMining
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy thực nghiệm

Mọi lệnh dưới đây chạy được ngay sau bước cài đặt, không cần sửa mã nguồn.
Kết quả ghi vào thư mục `results/`.

### Tái thực nghiệm mô phỏng của bài báo

```bash
python experiments/simulation.py \
    --scenario A --p 8 --replicates 20 --method both \
    --source saved --rcon-backend grc_ipms --parallel 3 \
    --output results/simulation-A-p8.json
```

`--scenario` nhận `A`, `B` hoặc `all`; `--p` nhận `8`, `12`, `16`, `20` hoặc
`all`. `--method tau` duyệt twin lattice, `--method submodel` duyệt dàn bao
hàm mô hình, `--method both` chạy cả hai.

`--rcon-backend grc_ipms` tái tạo đúng cách khớp mô hình của mã R
(`gRc::rcox`, phương pháp IPMS) và cho số liệu so sánh được với bài báo, nhưng
chậm. `--rcon-backend mle` khớp trực tiếp bằng cực đại hoá hợp lý, nhanh hơn
nhiều và được dùng cho `p = 16, 20`. Sự khác biệt này được nêu rõ trong báo
cáo vì nó ảnh hưởng tới thời gian chạy.

Tiến trình được ghi lại sau mỗi replicate; thêm `--resume` để chạy tiếp một
lần chạy bị gián đoạn.

### Dữ liệu chất lượng không khí và fMRI

```bash
python experiments/air_quality.py --output results/air-quality.json
python experiments/fmri.py --output results/fmri.json
```

Hai lệnh này cần các tệp `.RData` trong kho mã nguồn của nhóm tác giả (thư mục
`applications/`). Nhóm không có dữ liệu thô của hai ứng dụng nên chỉ tái dựng
được một phần; giới hạn cụ thể trình bày ở Chương 4 của báo cáo.

### Baseline pdglasso trên dữ liệu mô phỏng (mở rộng)

```bash
python experiments/baseline_simulation.py \
    --scenario A --p 8 --replicates 20 \
    --output results/baseline-A-p8.json
```

Chạy pdRCON graphical lasso trên ma trận hiệp phương sai mẫu và trên ma trận
tương quan mẫu, chấm điểm bằng đúng bộ độ đo dùng cho các thủ tục từng bước.
Bài báo chỉ so sánh `tau` với `submodel` — cả hai đều thuộc phương pháp đề
xuất — nên thực nghiệm mô phỏng gốc không có đối thủ ngoài; phần này bổ sung
điều đó.

### Độ ổn định bằng bootstrap (mở rộng)

```bash
python experiments/stability.py \
    --scenario A --p 8 --replicate 1 --method tau \
    --bootstrap 200 --parallel 6 \
    --output results/stability-A-p8-tau.json
```

Lấy mẫu lại **theo hàng** (giữ nguyên cấu trúc cặp của các cột) rồi chạy lại
thuật toán trên từng mẫu bootstrap, thu tần suất được chọn của từng cạnh và
từng phát biểu đối xứng. Bài báo báo cáo độ phục hồi trung bình trên 20
replicate độc lập, tức là mức chính xác kỳ vọng trên một mẫu mới; đại lượng ở
đây trả lời một câu hỏi khác — một mô hình cụ thể nhạy đến mức nào với chính
mẫu đã sinh ra nó.

Kết quả gồm `instability` (chỉ số bất ổn định trung bình trên mọi cặp đỉnh,
bằng 0 khi mọi mẫu bootstrap cho cùng một tập cạnh, tối đa 0,5) và
`exact_recovery_rate` (tỉ lệ mẫu bootstrap trả về đúng mô hình điểm, kể cả các
lớp màu).

## Ghi chú về tái lập kết quả

- Mọi thực nghiệm đều đặt hạt giống ngẫu nhiên cố định (`--seed`, mặc định
  2024). Thuật toán loại bỏ lùi bản thân nó tất định; ngẫu nhiên chỉ xuất hiện
  ở bước sinh dữ liệu và bước lấy mẫu bootstrap.
- Số liệu trong báo cáo được sinh từ chín notebook Kaggle lưu trong
  `notebooks/`, mỗi notebook kèm log đầy đủ của lần chạy cuối.
- Với `p = 16, 20`, cách khớp mô hình RCON khác với `p = 8, 12`. Thời gian
  chạy giữa hai nhóm cấu hình vì thế không so sánh trực tiếp được.

## Nhóm thực hiện

| MSHV | Họ tên |
| --- | --- |
| 25C11025 | Huỳnh Quang Trung |
| 25C11073 | Phan Nguyễn Anh Vinh |
| 25C15053 | Trương Thành Nhân |
| 25C15069 | Trần Ngọc Vỹ |

## Tài liệu tham khảo

1. A. Roverato and D. N. Nguyen, "Exploration of the search space of Gaussian
   graphical models for paired data," *Journal of Machine Learning Research*,
   vol. 25, no. 92, pp. 1–41, 2024.
2. S. Højsgaard and S. L. Lauritzen, "Graphical Gaussian models with edge and
   vertex symmetries," *Journal of the Royal Statistical Society: Series B*,
   vol. 70, no. 5, pp. 1005–1027, 2008.
3. N. Meinshausen and P. Bühlmann, "Stability selection," *Journal of the
   Royal Statistical Society: Series B*, vol. 72, no. 4, pp. 417–473, 2010.
4. Mã nguồn R gốc: <https://github.com/NgocDung-NGUYEN/backwardCGM-PD>
