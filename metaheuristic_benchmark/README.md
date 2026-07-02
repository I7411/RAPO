# Metaheuristic Benchmark Framework

## 1. Giới thiệu

**Metaheuristic Benchmark Framework** là một dự án Python 3.11 chuyên biệt dùng để nghiên cứu, đánh giá và so sánh các thuật toán tối ưu hóa siêu heuristic (metaheuristic). Khung làm việc tập trung vào bài toán tìm **cực tiểu (minimization)** trên các hàm benchmark liên tục.

Framework được xây dựng theo kiến trúc hướng đối tượng (OOP) thuần túy, hỗ trợ đa dạng thuật toán đơn và thuật toán lai RAPO (Rabbits and Pufferfish Optimization), đồng thời tích hợp hai bộ benchmark chuẩn quốc tế là **CEC2017** và **CEC2022**. Toàn bộ quá trình thực nghiệm được quản lý qua menu tương tác trên terminal.

---

## 2. Mục tiêu project

- Triển khai và chuẩn hóa các thuật toán tối ưu hóa metaheuristic kinh điển và hiện đại.
- Xây dựng và đánh giá **21 biến thể lai RAPO** kết hợp giữa ARO (Artificial Rabbits Optimization) và POA (Pufferfish Optimization Algorithm).
- So sánh công bằng giữa các thuật toán theo số lần đánh giá hàm mục tiêu (`nfe`), không chỉ theo số vòng lặp.
- Tự động xuất kết quả thực nghiệm ra **CSV**, **metadata JSON** và **biểu đồ** (convergence curve, boxplot, bar chart) phục vụ viết báo cáo khoa học.
- Xây dựng nền tảng dễ mở rộng: thêm thuật toán mới, thêm benchmark mới chỉ cần đăng ký vào registry.

---

## 3. Điểm nổi bật

| Tính năng | Trạng thái |
|---|---|
| 6 thuật toán đơn (ARO, POA, GWO, PSO, GA, HHO) + 1 lai cơ bản (GA_PSO) | ✅ Có sẵn |
| 21 biến thể lai RAPO | ✅ Có sẵn |
| 15 benchmark cơ bản (n-dimensional + fixed-dimension) | ✅ Có sẵn |
| CEC2017 (30 hàm, D=10/30/50/100) | ✅ Có sẵn |
| CEC2022 (12 hàm, D=2/10/20) qua `opfunu` | ✅ Có sẵn |
| Cơ chế `Evaluator` đếm NFE chính xác | ✅ Có sẵn |
| Quản lý seed tái lập kết quả (`RandomManager`) | ✅ Có sẵn |
| Xử lý biên bằng `BoundaryHandler.clip()` | ✅ Có sẵn |
| Xuất CSV raw + summary + metadata JSON | ✅ Có sẵn |
| Vẽ convergence curve, boxplot, bar chart (dpi=300) | ✅ Có sẵn |
| Xuất lại biểu đồ từ CSV cũ (menu 5) | ✅ Có sẵn |
| Registry thuật toán/benchmark theo tên | ✅ Có sẵn |
| 10 bộ test tự động trong `tests/` | ✅ Có sẵn |
| Menu tương tác chọn theo số thứ tự | ✅ Có sẵn |
| Dừng theo `max_evaluations` (evaluation budget) | ⚠️ Hỗ trợ qua config nhưng phụ thuộc backend RAPO |
| Kiểm định thống kê (Wilcoxon/Friedman) | ❌ TODO |
| Xuất bảng LaTeX | ❌ TODO |

---

## 4. Cấu trúc thư mục

```text
metaheuristic_benchmark/
├── algorithms/                         # Các thuật toán tối ưu hóa
│   ├── __init__.py
│   ├── algorithm_registry.py           # Registry đăng ký/truy xuất thuật toán theo tên
│   ├── base_optimizer.py               # Abstract class BaseOptimizer
│   ├── aro.py                          # Artificial Rabbits Optimization
│   ├── ga.py                          # Genetic Algorithm
│   ├── ga_pso.py                       # GA × PSO hybrid cơ bản
│   ├── gwo.py                         # Grey Wolf Optimizer
│   ├── hho.py                         # Harris Hawks Optimization
│   ├── poa.py                         # Pufferfish Optimization Algorithm
│   ├── pso.py                         # Particle Swarm Optimization
│   └── hybrids/                       # 21 biến thể lai RAPO
│       ├── rapo_chuyenphathichnghi.py  # RAPO-ESW
│       ├── rapo_cungpha_aro_poa.py     # RAPO-SOC-AP
│       ├── rapo_cungpha_poa_aro.py     # RAPO-SOC-PA
│       ├── rapo_dao.py                 # RAPO-IM
│       ├── rapo_ensemble_dachienluoc.py # RAPO-MSE
│       ├── rapo_epr_aro_poa.py         # RAPO-EPR-AP
│       ├── rapo_epr_poa_aro.py         # RAPO-EPR-PA
│       ├── rapo_hauchinh_aro_poa.py    # RAPO-POST-AP
│       ├── rapo_hauchinh_poa_aro.py    # RAPO-POST-PA
│       ├── rapo_quanthe.py             # RAPO-PH
│       ├── rapo_songsong.py            # RAPO-PAR
│       ├── rapo_tinhhoa_aro_poa.py     # RAPO-EG-AP
│       ├── rapo_tinhhoa_poa_aro.py     # RAPO-EG-PA
│       ├── rapo_toantu.py              # RAPO-OLH
│       ├── rapo_toantuthichnghi.py     # RAPO-AOS
│       ├── rapo_tritre_aro_poa.py      # RAPO-STG-AP
│       ├── rapo_tritre_poa_aro.py      # RAPO-STG-PA
│       ├── rapo_tuantu.py              # RAPO-SEQ
│       ├── rapo_tuantu_poa_aro.py      # (biến thể ngược, không đăng ký registry)
│       ├── rapo_xacsuatchontoantu.py   # RAPO-PR
│       ├── rapo_xpr_aro_poa.py         # RAPO-XPR-AP
│       └── rapo_xpr_poa_aro.py         # RAPO-XPR-PA
│
├── benchmarks/                         # Các hàm benchmark
│   ├── __init__.py
│   ├── benchmark_registry.py           # Registry benchmark theo tên
│   ├── basic_functions.py              # Sphere
│   ├── unimodal.py                     # Rosenbrock, Zakharov, Bent_Cigar, Sum_Squares
│   ├── multimodal.py                   # Rastrigin, Ackley, Griewank, Schwefel, Levy
│   ├── fixed_dimension.py              # Booth, Matyas, Three-Hump_Camel, Beale, Easom (D=2)
│   ├── cec2017/
│   │   ├── __init__.py
│   │   ├── cec2017_functions.py        # Adapter CEC2017 (dùng vendor local)
│   │   └── vendor/                     # Thư viện CEC2017 đặt ở đây
│   └── cec2022/
│       ├── __init__.py
│       ├── cec2022_functions.py        # Adapter CEC2022 (dùng opfunu pip)
│       └── vendor/                     # Thư mục chứa mã nguồn gốc MATLAB/C++ (tham khảo)
│
├── configs/                            # Cấu hình thực nghiệm (JSON)
│   ├── default_config.json             # Tham số mặc định (pop, iter, runs, seed, bounds)
│   ├── algorithm_config.json           # Tham số riêng từng thuật toán (PSO: c1, c2; GA: cr, mr)
│   ├── benchmark_config.json           # Danh sách benchmark được chọn
│   └── experiment_config.json          # Cấu hình preset fast_test và full_comparison
│
├── core/                               # Các class nền tảng OOP
│   ├── __init__.py
│   ├── problem.py                      # Dataclass Problem
│   ├── evaluator.py                    # Evaluator (đếm NFE)
│   ├── result.py                       # Dataclass OptimizationResult
│   ├── random_manager.py               # RandomManager (quản lý seed)
│   └── boundary_handler.py             # BoundaryHandler (clip biên)
│
├── experiments/                        # Các runner thực nghiệm
│   ├── __init__.py
│   ├── run_single.py                   # Chạy 1 thuật toán / 1 benchmark
│   ├── run_comparison.py               # So sánh nhiều thuật toán / 1 benchmark
│   ├── run_all_benchmarks.py           # Chạy tất cả thuật toán / tất cả benchmark
│   ├── statistical_summary.py          # Tính thống kê, xếp hạng
│   └── replot_from_csv.py              # Xuất lại biểu đồ từ CSV cũ
│
├── outputs/                            # Kết quả đầu ra (tự động tạo khi chạy)
│   ├── csv/
│   │   ├── raw/                        # CSV raw từng run
│   │   └── summary/                    # CSV tổng kết thống kê
│   ├── figures/
│   │   ├── convergence/                # PNG convergence curve
│   │   ├── boxplot/                    # PNG boxplot
│   │   └── bar/                        # PNG bar chart
│   ├── logs/                           # Logs (thư mục tự tạo)
│   └── metadata/                       # JSON metadata thực nghiệm
│
├── tests/                              # Unit test và integration test
│   ├── test_algorithm_registry.py
│   ├── test_algorithms.py
│   ├── test_benchmarks.py
│   ├── test_cec2017.py
│   ├── test_cec2022.py
│   ├── test_evaluator.py
│   ├── test_fixed_dimension.py
│   ├── test_output.py
│   ├── test_output_structure.py
│   └── test_summary_metrics.py
│
├── utils/                              # Tiện ích hỗ trợ
│   ├── __init__.py
│   ├── console.py                      # In màn hình (print_info, print_success, print_error...)
│   ├── csv_writer.py                   # Ghi CSV raw và summary
│   ├── file_naming.py                  # Sinh tên file output chuẩn hóa
│   ├── plotter.py                      # Vẽ convergence curve, boxplot, bar chart
│   └── timer.py                        # Đo thời gian (context manager)
│
├── main.py                             # Entry point – menu tương tác terminal (0–10)
├── requirements.txt                    # Thư viện phụ thuộc
├── verify.py                           # Script kiểm tra nhanh môi trường
└── README.md                           # Tài liệu hướng dẫn (file này)
```

---

## 5. Kiến trúc tổng thể

Chương trình được xây dựng theo nguyên tắc **OOP**, **phân tách trách nhiệm rõ ràng** và **registry pattern**.

### Các thành phần cốt lõi (`core/`)

| Class | File | Vai trò |
|---|---|---|
| `Problem` | `problem.py` | Dataclass định nghĩa bài toán: `name`, `function`, `dimension`, `lower_bound`, `upper_bound`, `global_minimum`, `category`, `is_fixed_dimension`, `is_placeholder`, `is_official` |
| `Evaluator` | `evaluator.py` | Bọc quanh `problem.function`, đếm `nfe` mỗi lần gọi, track `global_best_fitness` và `global_best_solution`. Thuật toán phải gọi `evaluator.evaluate(x)` thay vì gọi hàm mục tiêu trực tiếp |
| `OptimizationResult` | `result.py` | Dataclass lưu kết quả 1 run: `best_fitness`, `best_solution`, `convergence_curve`, `runtime_seconds`, `nfe`, `final_error`, `seed`, `run_id`, v.v. |
| `RandomManager` | `random_manager.py` | Thiết lập seed cho `numpy.random` và `random` module. Gọi `RandomManager.set_seed(seed)` trước mỗi run để đảm bảo tái lập kết quả |
| `BoundaryHandler` | `boundary_handler.py` | Xử lý nghiệm vượt biên bằng `clip(solution, lb, ub)` – dùng `numpy.clip` |

### Registry (`algorithms/`, `benchmarks/`)

| Module | Vai trò |
|---|---|
| `algorithm_registry.py` | Đăng ký tất cả thuật toán theo tên (string key → class). Hàm chính: `register_algorithm()`, `get_algorithm()`, `list_algorithms()`, `list_algorithms_by_type()`, `create_rapo_wrapper()` |
| `benchmark_registry.py` | Đăng ký 15 benchmark cơ bản. Hàm `get_benchmark(name, dim)` tự động route sang CEC2017 hoặc CEC2022 nếu tên tương ứng |

### Thuật toán (`algorithms/`)

- **Thuật toán đơn** (ARO, POA, GWO, PSO, GA, HHO, GA_PSO): kế thừa trực tiếp `BaseOptimizer` (abstract class trong `algorithms/base_optimizer.py`), implement `optimize(problem, config, seed) → OptimizationResult`.
- **RAPO hybrids**: viết dạng **standalone** (không kế thừa `BaseOptimizer`), nhận `objective_func` và `dimension` trực tiếp. Được bọc bởi `create_rapo_wrapper()` trong `algorithm_registry.py` để chuyển đổi về interface `BaseOptimizer` chuẩn.

### Runners (`experiments/`)

| Module | Chức năng |
|---|---|
| `run_single.py` | Chạy 1 thuật toán / 1 benchmark, `runs` lần độc lập. Xuất CSV raw, PNG convergence, JSON metadata |
| `run_comparison.py` | Gọi `run_single` cho từng thuật toán, tính summary, xuất CSV summary, PNG boxplot, JSON metadata |
| `run_all_benchmarks.py` | Gọi `run_comparison` cho từng benchmark, tính overall summary, xuất CSV tổng, PNG bar chart |
| `statistical_summary.py` | Tính các chỉ số thống kê và xếp hạng từ danh sách `OptimizationResult` |
| `replot_from_csv.py` | Đọc lại file CSV summary có sẵn và vẽ lại bar chart cho metric tùy chọn |

### Tiện ích (`utils/`)

| Module | Chức năng |
|---|---|
| `file_naming.py` | Sinh tên file output theo chuẩn `prefix__algorithm__hybrid__benchmark__Ddim__timestamp.ext` |
| `csv_writer.py` | Ghi CSV raw (chi tiết từng run) và CSV summary (thống kê tổng hợp) |
| `plotter.py` | Vẽ convergence curve (log scale), boxplot (log scale), bar chart (log scale nếu là fitness/error metric) |
| `console.py` | In terminal có màu: `print_info`, `print_success`, `print_error`, `print_warning` |
| `timer.py` | Đo thời gian thực thi bằng context manager `measure_time()` |

---

## 6. Luồng hoạt động của chương trình

```
python main.py
    │
    ├── Tạo thư mục outputs/ (nếu chưa có)
    └── Hiển thị menu (0–10)
        │
        └── [Ví dụ: chọn 2 – So sánh nhiều thuật toán]
            │
            ├── select_multiple_algorithms()
            │     └── Hiển thị danh sách, nhận input (số hoặc tên, hỗ trợ 99=tất cả)
            │
            ├── select_benchmark()
            │     └── Hiển thị danh sách 15 benchmark cơ bản, nhận input
            │
            ├── Nhập dimension
            │
            └── run_comparison(algs, bench, dim)
                  │
                  ├── [Với mỗi alg]:
                  │     └── run_single(alg, bench, dim)
                  │           │
                  │           ├── load_config("default_config.json")
                  │           ├── load_config("algorithm_config.json")
                  │           ├── get_benchmark(bench, dim) → Problem
                  │           ├── get_algorithm(alg) → OptimizerClass
                  │           │
                  │           └── [Với mỗi run (1..runs)]:
                  │                 seed = seed_start + run_id
                  │                 optimizer.optimize(problem, config, seed) → OptimizationResult
                  │                   │
                  │                   ├── RandomManager.set_seed(seed)
                  │                   ├── Khởi tạo quần thể trong [lb, ub]
                  │                   ├── [Mỗi vòng lặp]:
                  │                   │     ├── Cập nhật vị trí các cá thể
                  │                   │     ├── BoundaryHandler.clip(x, lb, ub)
                  │                   │     └── evaluator.evaluate(x) → fitness (nfe++)
                  │                   └── Trả về OptimizationResult
                  │
                  ├── calculate_summary(all_results) → thống kê, xếp hạng
                  │
                  └── Lưu output:
                        ├── outputs/csv/raw/raw__ALG__HYBRID__BENCH__Ddim__TS.csv
                        ├── outputs/csv/summary/summary__all_algorithms__BENCH__Ddim__TS.csv
                        ├── outputs/figures/convergence/convergence__ALG__HYBRID__BENCH__Ddim__TS.png
                        ├── outputs/figures/boxplot/boxplot__all_algorithms__BENCH__Ddim__TS.png
                        └── outputs/metadata/metadata__ALG__HYBRID__BENCH__Ddim__TS.json
```

---

## 7. Danh sách thuật toán

### 7.1. Thuật toán đơn (Base Algorithms)

| STT | Tên đăng ký | File | Mô tả |
|---|---|---|---|
| 1 | `ARO` | `aro.py` | Artificial Rabbits Optimization |
| 2 | `POA` | `poa.py` | Pufferfish Optimization Algorithm |
| 3 | `GWO` | `gwo.py` | Grey Wolf Optimizer |
| 4 | `PSO` | `pso.py` | Particle Swarm Optimization (c1=2.0, c2=2.0, w: 0.9→0.4) |
| 5 | `GA` | `ga.py` | Genetic Algorithm (crossover_rate=0.8, mutation_rate=0.1) |
| 6 | `HHO` | `hho.py` | Harris Hawks Optimization |
| 7 | `GA_PSO` | `ga_pso.py` | Lai cơ bản GA × PSO |

### 7.2. Thuật toán lai RAPO

Tổng cộng **21 biến thể** được đăng ký trong `algorithm_registry.py`.

> Xem chi tiết trong mục 8 bên dưới.

**Tổng số thuật toán đang đăng ký: 28** (7 base + 21 RAPO)

---

## 8. Chi tiết nhóm thuật toán lai RAPO

Tất cả biến thể RAPO kết hợp giữa **ARO** (Artificial Rabbits Optimization) và **POA** (Pufferfish Optimization Algorithm). Mỗi biến thể thể hiện một cách kết hợp khác nhau về chiến lược tìm kiếm.

### 8.1. Quy ước viết tắt thuật toán lai RAPO

Các mã viết tắt bên dưới chỉ dùng cho tài liệu, bảng kết quả và biểu đồ. Trong source code và registry, chương trình vẫn sử dụng tên đăng ký đầy đủ như `RAPO_Sequential`, `RAPO_EnergySwitch`,...

**Quy tắc viết tắt thống nhất:**
- **R** = RAPO
- **SEQ** = Sequential
- **ESW** = Energy Switch
- **SOC** = Sequential Operator Chain
- **IM** = Island Model
- **MSE** = Multi-Strategy Ensemble
- **EPR** = Exploration Phase Replacement
- **XPR** = Exploitation Phase Replacement
- **POST** = Post Optimization
- **PH** = Population Hybrid
- **PAR** = Parallel Hybrid
- **EG** = Elite Guided
- **OLH** = Operator-Level Hybrid
- **AOS** = Adaptive Operator Selection
- **PR** = Probabilistic Roulette
- **STG** = Stagnation Triggered
- **AP** = ARO → POA
- **PA** = POA → ARO

**Ghi chú hướng lai:**
- **AP** nghĩa là hướng lai ARO → POA.
- **PA** nghĩa là hướng lai POA → ARO.
- Các biến thể không có hướng rõ ràng thì không cần hậu tố AP/PA.

**Cách đọc tên thuật toán lai:**
- `RAPO-EPR-AP`: RAPO = RAPO, EPR = Exploration Phase Replacement, AP = ARO → POA
- `RAPO-STG-PA`: RAPO = RAPO, STG = Stagnation Triggered, PA = POA → ARO

**Bảng tra cứu nhanh mã viết tắt:**

| Mã viết tắt | Tên đầy đủ | Kiểu lai | Hướng lai | Ghi chú |
|---|---|---|---|---|
| **RAPO-SEQ** | `RAPO_Sequential` | Lai tuần tự | ARO → POA mặc định | Dễ giải thích, phù hợp baseline hybrid |
| **RAPO-ESW** | `RAPO_EnergySwitch` | Chuyển pha năng lượng | Không cố định | Dựa trên energy factor |
| **RAPO-SOC-AP** | `RAPO_SequentialOperatorChain_ARO_POA` | Chuỗi toán tử cùng pha | ARO → POA | Mỗi cá thể áp dụng ARO rồi POA |
| **RAPO-SOC-PA** | `RAPO_ReverseSequentialOperatorChain_POA_ARO` | Chuỗi toán tử cùng pha | POA → ARO | POA trước, ARO sau |
| **RAPO-IM** | `RAPO_IslandModel` | Island Model | Quần thể chia đôi | Trao đổi qua migration |
| **RAPO-MSE** | `RAPO_MultiStrategyEnsemble` | Ensemble đa chiến lược | Không cố định | Kết hợp nhiều chiến lược |
| **RAPO-EPR-AP** | `RAPO_ExplorationReplacement_ARO_POA` | Thay pha khám phá | ARO → POA | ARO làm khung, POA thay pha exploration |
| **RAPO-EPR-PA** | `RAPO_ExplorationReplacement_POA_ARO` | Thay pha khám phá | POA → ARO | POA làm khung, ARO thay pha exploration |
| **RAPO-XPR-AP** | `RAPO_ExploitationReplacement_ARO_POA` | Thay pha khai thác | ARO → POA | ARO làm khung, POA thay pha exploitation |
| **RAPO-XPR-PA** | `RAPO_ExploitationReplacement_POA_ARO` | Thay pha khai thác | POA → ARO | POA làm khung, ARO thay pha exploitation |
| **RAPO-POST-AP**| `RAPO_PostOptimization_ARO_POA` | Hậu tối ưu hóa | ARO → POA | ARO chạy trước, POA tinh chỉnh sau |
| **RAPO-POST-PA**| `RAPO_PostOptimization_POA_ARO` | Hậu tối ưu hóa | POA → ARO | POA chạy trước, ARO tinh chỉnh sau |
| **RAPO-PH** | `RAPO_PopulationHybrid` | Lai quần thể | Quần thể chia đôi | Trao đổi elite định kỳ |
| **RAPO-PAR** | `RAPO_ParallelHybrid` | Lai song song | Cả 2 cùng sinh ứng viên | Chọn greedy theo fitness |
| **RAPO-EG-AP** | `RAPO_EliteGuided_ARO_POA` | Elite-guided | ARO → POA | ARO là base, POA hướng elite |
| **RAPO-EG-PA** | `RAPO_EliteGuided_POA_ARO` | Elite-guided | POA → ARO | POA là base, ARO hướng elite |
| **RAPO-OLH** | `RAPO_OperatorLevelHybrid` | Lai cấp toán tử | Trộn công thức | Lai ở mức độ toán tử |
| **RAPO-AOS** | `RAPO_OperatorSelection` | Chọn toán tử thích nghi | Xác suất cập nhật | Adaptive Operator Selection |
| **RAPO-PR** | `RAPO_ProbabilisticRoulette` | Roulette xác suất | Xác suất cố định | Không adaptive |
| **RAPO-STG-AP** | `RAPO_StagnationTriggered_ARO_POA` | Kích hoạt khi trì trệ | ARO → POA | Khi kẹt tự động kích hoạt POA |
| **RAPO-STG-PA** | `RAPO_StagnationTriggered_POA_ARO` | Kích hoạt khi trì trệ | POA → ARO | Khi kẹt tự động kích hoạt ARO |

> Ghi chú: File `rapo_tuantu_poa_aro.py` là biến thể tuần tự ngược POA → ARO, hiện chưa được đăng ký vào registry nên không đưa vào bảng.

### 8.2. Danh sách biến thể RAPO chi tiết

| STT | Tên đăng ký | Mã viết tắt | Class | File | Kiểu lai | Mô tả ngắn | Ghi chú NFE |
|---|---|---|---|---|---|---|---|
| 1 | `RAPO_Sequential` | **RAPO-SEQ** | `RAPOSequential` | `rapo_tuantu.py` | Lai tuần tự thời gian | 50% iter đầu dùng ARO (khám phá), 50% iter sau dùng POA (khai thác) | NFE = pop × iter |
| 2 | `RAPO_EnergySwitch` | **RAPO-ESW** | `RAPOEnergySwitch` | `rapo_chuyenphathichnghi.py` | Lai chuyển pha năng lượng | Chuyển đổi thích nghi giữa ARO và POA theo hệ số năng lượng giảm dần | NFE = pop × iter |
| 3 | `RAPO_SequentialOperatorChain_ARO_POA` | **RAPO-SOC-AP** | `RAPOSequentialOperatorChainAROPOA` | `rapo_cungpha_aro_poa.py` | Chuỗi toán tử cùng pha (ARO→POA) | Cùng pha, mỗi cá thể áp dụng liên tiếp toán tử ARO rồi POA | NFE = 2 × pop × iter |
| 4 | `RAPO_ReverseSequentialOperatorChain_POA_ARO` | **RAPO-SOC-PA** | `RAPOReverseSequentialOperatorChainPOAARO` | `rapo_cungpha_poa_aro.py` | Chuỗi toán tử cùng pha (POA→ARO) | Đảo chiều: POA trước, ARO sau | NFE = 2 × pop × iter |
| 5 | `RAPO_IslandModel` | **RAPO-IM** | `RAPOIslandModel` | `rapo_dao.py` | Island Model | Quần thể chia thành các đảo ARO/POA riêng biệt, migration định kỳ | NFE = pop × iter |
| 6 | `RAPO_MultiStrategyEnsemble` | **RAPO-MSE** | `RAPOMultiStrategyEnsemble` | `rapo_ensemble_dachienluoc.py` | Ensemble đa chiến lược | Kết hợp nhiều chiến lược ARO/POA đồng thời, greedy selection | NFE có thể cao hơn |
| 7 | `RAPO_ExplorationReplacement_ARO_POA` | **RAPO-EPR-AP** | `RAPOExplorationReplacementAROPOA` | `rapo_epr_aro_poa.py` | Thay thế pha khám phá (ARO-base, dùng POA khám phá) | Trong pha exploration, thay toán tử ARO bằng toán tử POA | NFE = pop × iter |
| 8 | `RAPO_ExplorationReplacement_POA_ARO` | **RAPO-EPR-PA** | `RAPOExplorationReplacementPOAARO` | `rapo_epr_poa_aro.py` | Thay thế pha khám phá (POA-base, dùng ARO khám phá) | Đảo chiều: POA là base, ARO thay pha khám phá | NFE = pop × iter |
| 9 | `RAPO_ExploitationReplacement_ARO_POA` | **RAPO-XPR-AP** | `RAPOExploitationReplacementAROPOA` | `rapo_xpr_aro_poa.py` | Thay thế pha khai thác (ARO-base, dùng POA khai thác) | Trong pha exploitation, thay toán tử ARO bằng toán tử POA | NFE = pop × iter |
| 10 | `RAPO_ExploitationReplacement_POA_ARO` | **RAPO-XPR-PA** | `RAPOExploitationReplacementPOAARO` | `rapo_xpr_poa_aro.py` | Thay thế pha khai thác (POA-base, dùng ARO khai thác) | Đảo chiều: POA là base, ARO thay pha khai thác | NFE = pop × iter |
| 11 | `RAPO_PostOptimization_ARO_POA` | **RAPO-POST-AP** | `RAPOPostOptimizationAROPOA` | `rapo_hauchinh_aro_poa.py` | Hậu tối ưu hóa (ARO→POA) | ARO chạy trước để tìm vùng tốt, POA tinh chỉnh sau | NFE = pop × iter |
| 12 | `RAPO_PostOptimization_POA_ARO` | **RAPO-POST-PA** | `RAPOPostOptimizationPOAARO` | `rapo_hauchinh_poa_aro.py` | Hậu tối ưu hóa (POA→ARO) | POA chạy trước, ARO tinh chỉnh sau | NFE = pop × iter |
| 13 | `RAPO_PopulationHybrid` | **RAPO-PH** | `RAPOPopulationHybrid` | `rapo_quanthe.py` | Lai quần thể chia đôi | 50% quần thể chạy ARO, 50% chạy POA, trao đổi elite định kỳ | NFE = pop × iter |
| 14 | `RAPO_ParallelHybrid` | **RAPO-PAR** | `RAPOParallelHybrid` | `rapo_songsong.py` | Lai song song | Mỗi iter, cả ARO và POA cùng sinh ứng viên, chọn greedy theo fitness | NFE = 2 × pop × iter |
| 15 | `RAPO_EliteGuided_ARO_POA` | **RAPO-EG-AP** | `RAPOEliteGuidedAROPOA` | `rapo_tinhhoa_aro_poa.py` | Elite-guided (ARO-base, POA hướng elite) | ARO là base, POA được dẫn hướng bởi tập elite của ARO | NFE = pop × iter |
| 16 | `RAPO_EliteGuided_POA_ARO` | **RAPO-EG-PA** | `RAPOEliteGuidedPOAARO` | `rapo_tinhhoa_poa_aro.py` | Elite-guided (POA-base, ARO hướng elite) | Đảo chiều | NFE = pop × iter |
| 17 | `RAPO_OperatorLevelHybrid` | **RAPO-OLH** | `RAPOOperatorLevelHybrid` | `rapo_toantu.py` | Lai cấp toán tử | Lai ở mức độ toán tử cập nhật vị trí, trộn trực tiếp các công thức ARO và POA | NFE = pop × iter |
| 18 | `RAPO_OperatorSelection` | **RAPO-AOS** | `RAPOOperatorSelection` | `rapo_toantuthichnghi.py` | Chọn toán tử thích nghi (AOS) | Xác suất chọn ARO/POA cập nhật theo hiệu quả thực tế | NFE = pop × iter |
| 19 | `RAPO_ProbabilisticRoulette` | **RAPO-PR** | `RAPOProbabilisticRoulette` | `rapo_xacsuatchontoantu.py` | Roulette xác suất | Mỗi cá thể chọn ARO hoặc POA theo xác suất cố định (roulette), không adaptive | NFE = pop × iter |
| 20 | `RAPO_StagnationTriggered_ARO_POA` | **RAPO-STG-AP** | `RAPOStagnationTriggeredAROPOA` | `rapo_tritre_aro_poa.py` | Kích hoạt khi trì trệ (ARO→POA) | ARO là base, khi bị kẹt (stagnation) tự động kích hoạt POA để thoát | NFE = pop × iter |
| 21 | `RAPO_StagnationTriggered_POA_ARO` | **RAPO-STG-PA** | `RAPOStagnationTriggeredPOAARO` | `rapo_tritre_poa_aro.py` | Kích hoạt khi trì trệ (POA→ARO) | Đảo chiều | NFE = pop × iter |

> **Lưu ý NFE:** POA có thể sinh 2 ứng viên cho mỗi cá thể mỗi vòng lặp, do đó các biến thể dùng toán tử POA có thể tiêu thụ NFE gấp đôi so với ARO trong cùng cấu hình. Cơ chế `Evaluator` đếm chính xác từng lần gọi hàm mục tiêu, đảm bảo tính minh bạch khi so sánh.

---

## 9. Danh sách benchmark

### 9.1. Benchmark n-dimensional (linh hoạt số chiều)

| Tên | File | Giá trị tối ưu | Bounds | Ghi chú |
|---|---|---|---|---|
| `Sphere` | `basic_functions.py` | 0 | [-100, 100] | Unimodal đơn giản |
| `Rosenbrock` | `unimodal.py` | 0 | [-30, 30] | Thung lũng hẹp |
| `Zakharov` | `unimodal.py` | 0 | [-5, 10] | Unimodal |
| `Bent_Cigar` | `unimodal.py` | 0 | [-100, 100] | Unimodal ill-conditioned |
| `Sum_Squares` | `unimodal.py` | 0 | [-10, 10] | Unimodal |
| `Rastrigin` | `multimodal.py` | 0 | [-5.12, 5.12] | Multimodal nổi tiếng |
| `Ackley` | `multimodal.py` | 0 | [-32, 32] | Multimodal |
| `Griewank` | `multimodal.py` | 0 | [-600, 600] | Multimodal |
| `Schwefel` | `multimodal.py` | 0 | [-500, 500] | Multimodal (deceptive) |
| `Levy` | `multimodal.py` | 0 | [-10, 10] | Multimodal |

### 9.2. Benchmark Fixed-Dimension (chỉ D=2)

| Tên | File | Giá trị tối ưu | Bounds |
|---|---|---|---|
| `Booth` | `fixed_dimension.py` | 0 | [-10, 10] |
| `Matyas` | `fixed_dimension.py` | 0 | [-10, 10] |
| `Three-Hump_Camel` | `fixed_dimension.py` | 0 | [-5, 5] |
| `Beale` | `fixed_dimension.py` | 0 | [-4.5, 4.5] |
| `Easom` | `fixed_dimension.py` | -1.0 | [-100, 100] |

> Nếu chọn benchmark fixed-dimension với D≠2, chương trình sẽ raise `ValueError` rõ ràng.

### 9.3. CEC2017

Xem mục 10.

### 9.4. CEC2022

Xem mục 11.

---

## 10. CEC2017

- **Menu**: Chọn số `9` trong main menu
- **Số hàm**: 30 hàm (F1–F30)
- **Tên chuẩn**: `CEC2017_F1`, `CEC2017_F2`, ..., `CEC2017_F30`
- **Dimension hỗ trợ**: `10`, `30`, `50`, `100`
- **Bounds**: [-100, 100] cho mọi dimension
- **Global minimum**: `func_id × 100` (F1 = 100, F2 = 200, ..., F30 = 3000)
- **Thư viện**: Dùng package vendor local tại `benchmarks/cec2017/vendor/`
- **F2**: F2 thường được bỏ qua khi chạy toàn bộ (kém ổn định), option `99` sẽ tự động loại F2.

### Adapter hỗ trợ đầu vào 1D/2D

Hàm objective của CEC2017 trong framework hỗ trợ:
- **Vector 1D** `(D,)` → trả về `float` (đánh giá 1 nghiệm)
- **Ma trận 2D** `(pop_size, D)` → trả về `np.ndarray` shape `(pop_size,)` (đánh giá cả quần thể)

### Cách chạy CEC2017

```
python main.py → chọn 9
→ Chọn thuật toán (số thứ tự hoặc tên, 99 = tất cả)
→ Nhập ID hàm (1–30) hoặc 99 để chạy toàn bộ
→ Nhập số chiều (10, 30, 50, 100)
```

---

## 11. CEC2022

- **Menu**: Chọn số `10` trong main menu
- **Số hàm**: 12 hàm (F1–F12)
- **Tên chuẩn**: `CEC2022_F1`, `CEC2022_F2`, ..., `CEC2022_F12`
- **Dimension hỗ trợ**: `2`, `10`, `20`
- **Bounds**: [-100, 100] cho mọi dimension
- **Thư viện**: Dùng pip package `opfunu>=1.0.1` (bắt buộc cài đặt)
- **Adapter**: `benchmarks/cec2022/cec2022_functions.py` bọc `opfunu.cec_based.cec2022.F{i}2022`

### Ràng buộc quan trọng

| Hàm | D=2 | D=10 | D=20 |
|---|---|---|---|
| F1–F5 | ✅ | ✅ | ✅ |
| **F6, F7, F8** | ❌ **Không hỗ trợ** | ✅ | ✅ |
| F9–F12 | ✅ | ✅ | ✅ |

> F6, F7, F8 không hỗ trợ D=2 vì bộ dữ liệu shuffle data gốc không định nghĩa cho chiều này. Khi chạy toàn bộ với D=2, framework tự động bỏ qua F6, F7, F8 và thông báo rõ ràng.

### Global minimum CEC2022

| Hàm | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 | F10 | F11 | F12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Giá trị** | 300 | 400 | 600 | 800 | 900 | 1800 | 2000 | 2200 | 2300 | 2400 | 2600 | 2700 |

### Adapter hỗ trợ đầu vào 1D/2D

Tương tự CEC2017:
- **Vector 1D** `(D,)` → trả về `float`
- **Ma trận 2D** `(pop_size, D)` → trả về `np.ndarray` shape `(pop_size,)` (dùng `np.apply_along_axis`)

### Cách chạy CEC2022

```
python main.py → chọn 10
→ Chọn thuật toán (số thứ tự hoặc tên, 99 = tất cả)
→ Nhập ID hàm (1–12) hoặc 99 để chạy toàn bộ
→ Nhập số chiều (2, 10, 20)
```

---

## 12. Cấu hình thực nghiệm

### `configs/default_config.json`

Tham số mặc định được áp dụng khi không có config cụ thể:

```json
{
  "population_size": 10,
  "max_iterations": 10,
  "dimension": 10,
  "runs": 10,
  "seed_start": 42,
  "lower_bound": -10.0,
  "upper_bound": 10.0,
  "minimization": true
}
```

> **Lưu ý**: Đây là cấu hình mặc định rất nhỏ (dùng để test nhanh). Trong nghiên cứu thực tế, nên dùng `population_size=50`, `max_iterations=500`, `runs=30`.

### `configs/algorithm_config.json`

Tham số riêng từng thuật toán:
- `PSO`: `c1=2.0`, `c2=2.0`, `w_max=0.9`, `w_min=0.4`
- `GA`: `crossover_rate=0.8`, `mutation_rate=0.1`
- `RAPO`: `sequential_switch_ratio=0.5`, `energy_switch_threshold=1.0`, `population_hybrid_ratio=0.5`, `migration_interval=10`

### `configs/experiment_config.json`

Preset thực nghiệm (dùng cho menu `3`):

| Preset | Algorithms | Benchmarks | Runs | D | Pop | Iter |
|---|---|---|---|---|---|---|
| `fast_test` | ARO, POA | Sphere, Rastrigin | 5 | 10 | 20 | 30 |
| `full_comparison` | ARO, POA, RAPO_Sequential (`R-SEQ`), RAPO_EnergySwitch (`R-ESW`), GWO, PSO | Sphere, Rosenbrock, Rastrigin, Ackley, Griewank | 30 | 30 | 50 | 500 |

> Menu `3` đọc preset `full_comparison` từ `experiment_config.json`.

### Thứ tự ưu tiên cấu hình

```
default_config.json  →  algorithm_config.json  →  config_override (từ experiment_config hoặc trực tiếp)
```

### Seeding

Seeds = `[seed_start + 1, seed_start + 2, ..., seed_start + runs]`  
Với `seed_start=42` và `runs=10` → seeds = `[43, 44, ..., 52]`

### Chỉnh sửa cấu hình tại runtime

Chọn menu `6` để xem và sửa `default_config.json` trực tiếp trong terminal mà không cần mở file.

---

## 13. Nguyên tắc đánh giá công bằng

### Vì sao không chỉ so sánh theo Max Iterations?

POA có cơ chế sinh ra **2 ứng viên** cho mỗi cá thể trong một số pha của thuật toán. Do đó:
- ARO với `pop=50, iter=100` tiêu thụ khoảng **5000 NFE**
- POA với cùng `pop=50, iter=100` có thể tiêu thụ đến **10000 NFE**

So sánh theo `max_iterations` là **không công bằng** với các biến thể RAPO dùng công thức POA.

### Tiêu chí so sánh đúng

1. **So sánh theo NFE** (ưu tiên): Thiết lập cùng `max_function_evaluations` cho tất cả thuật toán.
2. **Điều kiện cùng nhau**: `population_size`, `dimension`, `bounds`, `seed_list`, `số lần chạy độc lập`.
3. **Số lần chạy tối thiểu**: Từ **25 đến 30 lần** chạy độc lập để kết quả có ý nghĩa thống kê.

### Chỉ số cần báo cáo

| Chỉ số | Ý nghĩa |
|---|---|
| `best` | Fitness tốt nhất trong tất cả các run |
| `mean` | Trung bình fitness qua các run |
| `worst` | Fitness tệ nhất |
| `std` | Độ lệch chuẩn (đánh giá độ ổn định) |
| `median` | Trung vị (ít bị ảnh hưởng bởi outlier) |
| `mean_error` | Trung bình `|fitness - global_minimum|` |
| `best_error` | Lỗi tốt nhất |
| `success_rate` | Tỷ lệ run đạt ngưỡng hội tụ (`success_tolerance=1e-8`) |
| `nfe` | Số lần gọi hàm mục tiêu |
| `runtime_seconds` | Thời gian thực thi |

### Xếp hạng

- Nếu benchmark có `global_minimum` → **ưu tiên xếp hạng theo `mean_error`**
- Nếu không có `global_minimum` → xếp hạng theo `mean`
- Tiebreaker: `best_error` → `avg_runtime_seconds`
- `excluded_from_ranking=True` nếu benchmark là `is_placeholder=True` hoặc `is_official=False`

---

## 14. Output của chương trình

### Cấu trúc thư mục output

```
outputs/
├── csv/
│   ├── raw/         → CSV chi tiết từng run của từng thuật toán
│   └── summary/     → CSV tổng kết thống kê nhiều thuật toán
├── figures/
│   ├── convergence/ → PNG convergence curve (log scale, dpi=300)
│   ├── boxplot/     → PNG boxplot so sánh (log scale, dpi=300)
│   └── bar/         → PNG bar chart theo metric (log scale nếu fitness/error)
├── logs/            → Thư mục log (tự tạo, hiện chỉ tạo cấu trúc)
└── metadata/        → JSON metadata thực nghiệm
```

### Quy tắc đặt tên file

| Loại file | Mẫu tên |
|---|---|
| CSV raw (1 thuật toán) | `raw__ARO__original__Sphere__D30__20260702_140000.csv` |
| CSV raw (RAPO) | `raw__RAPO_Sequential__RAPO__CEC2017_F1__D30__20260702_140000.csv` (Mã viết tắt khi trình bày báo cáo: `R-SEQ`) |
| PNG convergence | `convergence__ARO__original__Sphere__D30__20260702_140000.png` |
| CSV summary | `summary__all_algorithms__Sphere__D30__20260702_140000.csv` |
| PNG boxplot | `boxplot__all_algorithms__Sphere__D30__20260702_140000.png` |
| PNG bar chart | `bar__all_algorithms__all_benchmarks__D30__20260702_140000.png` |
| JSON metadata | `metadata__ARO__original__Sphere__D30__20260702_140000.json` |
| Bar replot | `bar_mean_error_summary__all_algorithms__CEC2022_F1__D10__20260702_140000.png` |

> **Ví dụ về quy ước tên cho báo cáo:**
> - Tên đăng ký trong CSV: `RAPO_ExplorationReplacement_ARO_POA`
> - Mã viết tắt khi trình bày báo cáo/biểu đồ: `R-EPR-AP`

### Nội dung CSV raw

Mỗi dòng tương ứng 1 run:

```
algorithm, hybrid_type, benchmark, dimension, run_id, seed,
best_fitness, best_solution, runtime_seconds,
population_size, max_iterations, lower_bound, upper_bound,
nfe, final_error, timestamp
```

> `best_solution` được lưu dạng JSON string. `convergence_curve` **không** được lưu vào CSV raw để tránh file quá lớn.

### Nội dung CSV summary

```
algorithm, hybrid_type, algorithm_full_name, benchmark, benchmark_category, dimension, runs,
mean, best, worst, std, median,
mean_error, best_error, worst_error, std_error,
nfe_mean, nfe_best, nfe_worst,
avg_runtime_seconds, rank_by_mean, rank_by_mean_error,
success_count, success_rate, excluded_from_ranking, timestamp
```

---

## 15. Cách cài đặt

### Yêu cầu

- Python **3.11+**
- pip

### Các bước cài đặt

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.24.0
matplotlib>=3.7.0
opfunu>=1.0.1
```

> `opfunu` là bắt buộc để chạy CEC2022. CEC2017 dùng thư viện vendor local, không cần pip thêm.

---

## 16. Cách chạy chương trình

```bash
python main.py
```

### Menu tổng quan

```
====================================================
 METAHEURISTIC BENCHMARK FRAMEWORK
====================================================
1. Chạy 1 thuật toán trên 1 benchmark
2. So sánh nhiều thuật toán trên 1 benchmark
3. Chạy tất cả thuật toán trên tất cả benchmark
4. Chạy riêng nhóm thuật toán lai RAPO
5. Xuất lại biểu đồ từ CSV có sẵn
6. Xem và chỉnh sửa cấu hình chạy
7. Liệt kê thuật toán hiện có
8. Liệt kê benchmark hiện có
9. Chạy benchmark CEC2017
10. Chạy benchmark CEC2022
0. Thoát
====================================================
```

### Chi tiết từng chức năng

| Số | Chức năng | Chọn gì | Kết quả |
|---|---|---|---|
| `1` | 1 thuật toán × 1 benchmark cơ bản | Thuật toán (số/tên) → Benchmark (số/tên) → D | CSV raw + PNG convergence + JSON metadata |
| `2` | Nhiều thuật toán × 1 benchmark cơ bản | Nhiều thuật toán (số, dấu phẩy, 99=tất cả) → Benchmark (số/tên) → D | CSV raw mỗi alg + CSV summary + PNG boxplot + JSON metadata |
| `3` | Tất cả theo `experiment_config.json` (full_comparison) | Không nhập | Toàn bộ kết quả theo preset |
| `4` | Chỉ nhóm RAPO × 1 benchmark cơ bản | Benchmark (số/tên) → D | So sánh tất cả RAPO |
| `5` | Vẽ lại biểu đồ từ CSV cũ | Chọn file CSV summary → chọn metric (số, dấu phẩy, 99=tất cả) | PNG bar chart |
| `6` | Xem/sửa cấu hình | Nhập giá trị mới hoặc Enter để giữ nguyên | Lưu lại `default_config.json` |
| `7` | Liệt kê thuật toán | — | Danh sách 28 thuật toán |
| `8` | Liệt kê benchmark cơ bản | — | Danh sách 15 benchmark kèm category/dim/global_min |
| `9` | CEC2017 | Thuật toán → ID hàm (1-30 hoặc 99) → D (10/30/50/100) | Tương tự 1 hoặc 2 tùy số thuật toán chọn |
| `10` | CEC2022 | Thuật toán → ID hàm (1-12 hoặc 99) → D (2/10/20) | Tương tự 1 hoặc 2 tùy số thuật toán chọn |

---

## 17. Ví dụ chạy thực nghiệm

### Ví dụ 1: ARO trên Sphere D30

```
Chọn: 1
→ Thuật toán: 1 (ARO)
→ Benchmark: 1 (Sphere)
→ Số chiều: 30
```

Output:
```
outputs/csv/raw/raw__ARO__original__Sphere__D30__20260702_140000.csv
outputs/figures/convergence/convergence__ARO__original__Sphere__D30__20260702_140000.png
outputs/metadata/metadata__ARO__original__Sphere__D30__20260702_140000.json
```

### Ví dụ 2: So sánh ARO vs POA vs RAPO_Sequential (R-SEQ) trên Rastrigin D30

```
Chọn: 2
→ Thuật toán: 1,2,25 (ARO, POA, RAPO_Sequential)
→ Benchmark: 6 (Rastrigin)
→ Số chiều: 30
```

### Ví dụ 3: Chạy CEC2017 F1 với tất cả thuật toán D30

```
Chọn: 9
→ Thuật toán: 99 (tất cả)
→ Hàm CEC2017: 1
→ Số chiều: 30
```

### Ví dụ 4: Chạy CEC2022 F12 với RAPO_Sequential (R-SEQ) D20

```
Chọn: 10
→ Thuật toán: 25 (RAPO_Sequential)
→ Hàm CEC2022: 12
→ Số chiều: 20
```

### Ví dụ 5: Chạy toàn bộ CEC2022 với ARO và RAPO_EnergySwitch (R-ESW) D10

```
Chọn: 10
→ Thuật toán: 1,8 (ARO, RAPO_EnergySwitch)
→ Hàm CEC2022: 99 (toàn bộ)
→ Số chiều: 10
```

### Ví dụ 6: Vẽ lại biểu đồ từ CSV summary cũ

```
Chọn: 5
→ Chọn file CSV số 1
→ Chọn metric: 99 (tất cả)
```

---

## 18. Cách đọc kết quả CSV / biểu đồ

### Các chỉ số trong CSV summary

| Cột | Ý nghĩa |
|---|---|
| `best_fitness` | Fitness tốt nhất đạt được trong 1 run |
| `best` | Min của `best_fitness` qua tất cả runs |
| `mean` | Trung bình `best_fitness` qua tất cả runs |
| `worst` | Max của `best_fitness` |
| `std` | Độ lệch chuẩn (nhỏ = ổn định) |
| `median` | Trung vị (ít ảnh hưởng bởi outlier hơn mean) |
| `final_error` | `|best_fitness - global_minimum|` của 1 run |
| `mean_error` | Trung bình `final_error` (chỉ số so sánh chính) |
| `best_error` | Min `final_error` |
| `std_error` | Độ lệch chuẩn của error |
| `nfe` | Số lần gọi hàm mục tiêu trong 1 run |
| `nfe_mean` | Trung bình NFE qua các runs |
| `runtime_seconds` | Thời gian thực thi 1 run (giây) |
| `avg_runtime_seconds` | Trung bình runtime |
| `convergence_curve` | Danh sách fitness tốt nhất qua từng vòng lặp (trong RAM, không lưu CSV) |
| `success_rate` | Tỷ lệ run có `final_error <= 1e-8` |
| `rank_by_mean_error` | Xếp hạng theo `mean_error` (1 = tốt nhất) |
| `rank_by_mean` | Xếp hạng theo `mean fitness` |

### Đọc biểu đồ

- **Convergence curve** (đường hội tụ): Trục Y là fitness trung bình qua các run (log scale), trục X là vòng lặp. Thuật toán tốt hơn có đường đi xuống nhanh và thấp hơn.
- **Boxplot**: Mỗi box thể hiện phân phối fitness (hoặc error) qua các run. Box thấp và hẹp = ổn định và tốt. Trục Y dùng log scale.
- **Bar chart**: So sánh trực tiếp một metric cụ thể giữa các thuật toán. Cột thấp hơn = tốt hơn (với error/fitness).

---

## 19. Cách thêm thuật toán mới

### Bước 1: Tạo file thuật toán

```python
# algorithms/my_algorithm.py
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.result import OptimizationResult
from core.random_manager import RandomManager
from core.boundary_handler import BoundaryHandler
from core.evaluator import Evaluator
from utils.timer import measure_time
import numpy as np

class MyAlgorithm(BaseOptimizer):
    name = "MyAlgorithm"
    hybrid_type = ""

    def optimize(self, problem: Problem, config: dict, seed: int = None) -> OptimizationResult:
        if seed is not None:
            RandomManager.set_seed(seed)

        pop_size = config.get("population_size", 50)
        max_iter = config.get("max_iterations", 100)
        lb, ub = problem.lower_bound, problem.upper_bound
        dim = problem.dimension

        evaluator = Evaluator(problem, config.get("minimization", True))

        # Khởi tạo quần thể
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([evaluator.evaluate(pop[i]) for i in range(pop_size)])

        convergence = []
        with measure_time() as t:
            for it in range(max_iter):
                # --- Logic thuật toán của bạn ở đây ---
                for i in range(pop_size):
                    new_x = ...  # cập nhật vị trí
                    new_x = BoundaryHandler.clip(new_x, lb, ub)
                    new_fit = evaluator.evaluate(new_x)  # PHẢI dùng evaluator
                    if new_fit < fitness[i]:
                        pop[i] = new_x
                        fitness[i] = new_fit
                convergence.append(float(np.min(fitness)))

        best_idx = np.argmin(fitness)
        final_error = abs(fitness[best_idx] - problem.global_minimum) if problem.global_minimum is not None else None

        return OptimizationResult(
            algorithm=self.name, hybrid_type=self.hybrid_type,
            benchmark=problem.name, dimension=dim, run_id=config.get("run_id", 1),
            seed=seed, best_fitness=float(fitness[best_idx]),
            best_solution=pop[best_idx].tolist(), convergence_curve=convergence,
            runtime_seconds=t.duration, population_size=pop_size, max_iterations=max_iter,
            lower_bound=lb, upper_bound=ub, nfe=evaluator.nfe, final_error=final_error,
        )
```

### Bước 2: Đăng ký vào registry

```python
# algorithms/algorithm_registry.py – thêm ở cuối file
from algorithms.my_algorithm import MyAlgorithm
register_algorithm("MyAlgorithm", MyAlgorithm)
```

### Bước 3: Thêm test

```python
# tests/test_my_algorithm.py
def test_my_algorithm_runs():
    from benchmarks.benchmark_registry import get_benchmark
    from algorithms.algorithm_registry import get_algorithm
    prob = get_benchmark("Sphere", 10)
    alg = get_algorithm("MyAlgorithm")()
    result = alg.optimize(prob, {"population_size": 5, "max_iterations": 10, "minimization": True}, seed=42)
    assert result.best_fitness >= 0
    assert result.nfe > 0
```

### Bước 4: Cập nhật README

Thêm tên thuật toán vào bảng mục 7.

---

## 20. Cách thêm benchmark mới

### Benchmark đơn giản

```python
# benchmarks/my_benchmarks.py
from core.problem import Problem
import numpy as np

def my_function_problem(dim: int) -> Problem:
    def objective(x):
        return float(np.sum(x ** 4))

    return Problem(
        name="MyFunction",
        function=objective,
        dimension=dim,
        lower_bound=-10.0,
        upper_bound=10.0,
        global_minimum=0.0,
        category="unimodal",
    )
```

### Đăng ký vào registry

```python
# benchmarks/benchmark_registry.py
from benchmarks.my_benchmarks import my_function_problem

_registry["MyFunction"] = my_function_problem
```

### Benchmark ngoài (có chuẩn riêng)

Nếu benchmark có thư viện ngoài (như CEC2022), tạo file adapter riêng trong thư mục con:
1. Tạo `benchmarks/my_benchmark_suite/my_adapter.py`
2. Adapter phải cung cấp `objective(x: 1D_array) → float` và `objective(x: 2D_array) → ndarray`
3. Đăng ký trong `benchmark_registry.py` bằng cách thêm điều kiện `if name.startswith("MY_"):` trong hàm `get_benchmark()`
4. Khai báo đầy đủ: bounds, dimension hợp lệ, global minimum

---

## 21. Kiểm thử chương trình

### Chạy tất cả test

```bash
# Nếu dùng unittest (mặc định, không cần cài thêm)
python -m unittest discover tests/

# Nếu dùng pytest (cài thêm: pip install pytest)
pytest tests/ -v
```

### Danh sách test

| File test | Nội dung kiểm tra |
|---|---|
| `test_algorithm_registry.py` | `list_algorithms()`, `list_algorithms_by_type()`, `get_algorithm()` raise ValueError đúng |
| `test_algorithms.py` | Chạy nhanh từng thuật toán, kết quả không NaN, bounds hợp lệ |
| `test_benchmarks.py` | `Sphere(0)=0`, `Rastrigin(0)=0`, `Ackley(0)≈0` |
| `test_cec2017.py` | Bounds [-100,100], global_min đúng, evaluate 1D→float, 2D→ndarray shape đúng |
| `test_cec2022.py` | Bounds [-100,100], global_min=300 cho F1, 1D/2D eval, F6+D=2 raises ValueError |
| `test_evaluator.py` | NFE tăng đúng sau mỗi lần evaluate, `reset()` hoạt động |
| `test_fixed_dimension.py` | Booth D=2 OK, Booth D=30 raises ValueError, Sphere D=30 OK |
| `test_output.py` | `run_single` tạo được CSV và PNG sau khi chạy |
| `test_output_structure.py` | Các thư mục `outputs/csv/raw/` và `outputs/metadata/` tồn tại sau khi chạy |
| `test_summary_metrics.py` | `calculate_summary()` trả về `success_count`, `success_rate`, `mean_error` đúng |

---

## 22. Lỗi thường gặp và cách xử lý

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: No module named 'opfunu'` | Chưa cài thư viện CEC2022 | Chạy: `pip install opfunu` hoặc `python -m pip install opfunu` |
| `ModuleNotFoundError: No module named 'cec2017'` | Thiếu vendor CEC2017 | Đảm bảo thư mục `benchmarks/cec2017/vendor/` tồn tại và có đầy đủ file |
| `FileNotFoundError: data.pkl` | Thiếu file dữ liệu của vendor CEC2017 | Kiểm tra file `benchmarks/cec2017/vendor/cec2017/data.pkl` tồn tại |
| `ValueError: CEC2017 dimension must be one of: 10, 30, 50, 100` | Chọn sai dimension cho CEC2017 | Chỉ nhập 10, 30, 50, hoặc 100 |
| `ValueError: CEC2022 dimension must be one of: 2, 10, 20` | Chọn sai dimension cho CEC2022 | Chỉ nhập 2, 10, hoặc 20 |
| `ValueError: CEC2022_F6 does not support D=2` | F6/F7/F8 không hỗ trợ D=2 | Dùng D=10 hoặc D=20 cho các hàm này |
| `ValueError: Benchmark 'X' không tồn tại trong registry` | Nhập sai tên benchmark | Dùng menu `8` để xem danh sách benchmark đúng |
| `ValueError: Algorithm 'X' không tồn tại trong registry` | Nhập sai tên thuật toán | Dùng menu `7` để xem danh sách thuật toán đúng |
| `ValueError: Test functions are only defined for D=2 only` | Benchmark fixed-dimension gọi với D≠2 | Chỉ dùng Booth, Matyas, v.v. với D=2 |
| `Không tìm thấy file CSV summary nào` (menu 5) | Chưa có file trong `outputs/csv/summary/` | Chạy ít nhất một thực nghiệm so sánh (menu 2, 3, 4, 9, 10) trước |
| Kết quả khác nhau mỗi lần chạy | Chưa cố định seed | Đảm bảo `seed_start` cố định trong `default_config.json` và không thay đổi |
| Biểu đồ không xuất được | Thiếu matplotlib | Chạy `pip install matplotlib>=3.7.0` |
| So sánh không công bằng | Dùng `max_iterations` thay vì NFE | Xem mục 13 về nguyên tắc đánh giá công bằng |

---

## 23. Hướng mở rộng

| Tính năng | Độ ưu tiên | Ghi chú |
|---|---|---|
| Thêm CEC2020 / CEC2021 / CEC2024 | Cao | `opfunu` hỗ trợ nhiều bộ CEC |
| Kiểm định thống kê (Wilcoxon rank-sum, Friedman test) | Cao | Cần cho báo cáo khoa học nghiêm túc |
| Xuất bảng kết quả dạng LaTeX | Trung bình | Cho báo cáo/luận văn |
| Thêm early stopping theo `max_function_evaluations` | Trung bình | Hiện tại tham số `use_evaluation_budget` đã có trong config nhưng phụ thuộc implementation backend |
| Chế độ batch experiment qua JSON config | Trung bình | Không cần chạy menu tương tác |
| Giao diện Streamlit trực quan | Thấp | Trực quan hóa kết quả trên web |
| Thêm bài toán rời rạc (TDVRP, TSP) | Thấp | Cần refactor `BoundaryHandler` và định nghĩa `Problem` |
| Thêm local search (2-opt, swap) | Thấp | Dành cho bài toán định tuyến |
| Multi-objective optimization | Thấp | Cần thêm Pareto front logic |

---

## 24. Ghi chú nghiên cứu

### Tính ngẫu nhiên và tái lập

- Thuật toán metaheuristic có tính ngẫu nhiên bản chất. **Không bao giờ kết luận thuật toán A tốt hơn B dựa trên 1 lần chạy**.
- Phải chạy **ít nhất 25–30 lần độc lập** với các seed khác nhau.
- `RandomManager` trong framework đảm bảo seed được thiết lập trước mỗi run. Kết quả có thể tái lập hoàn toàn nếu giữ nguyên `seed_start` và `runs`.

### NFE và công bằng

- Luôn báo cáo `nfe_mean` cùng với kết quả fitness.
- Khi paper yêu cầu "same computational budget", hãy dùng `max_function_evaluations` thay vì `max_iterations`.
- Biến thể dùng toán tử POA (sinh 2 ứng viên) sẽ tiêu thụ NFE cao hơn – cần ghi chú rõ trong báo cáo.

### Phân nhóm benchmark cho báo cáo

Tách rõ kết quả theo từng nhóm benchmark:
- **Nhóm 1**: Benchmark cơ bản (Sphere, Rastrigin, Ackley, ...) – kiểm tra hội tụ cơ bản
- **Nhóm 2**: CEC2017 – benchmark chuẩn quốc tế, có tham chiếu với nhiều paper khác
- **Nhóm 3**: CEC2022 – benchmark mới hơn, ít tham chiếu nhưng khó hơn

### Kết luận thống kê

Không kết luận thuật toán tốt hơn nếu chưa có kiểm định thống kê (Wilcoxon signed-rank test, Friedman test). Framework hiện chưa tích hợp sẵn tính năng này (TODO). Có thể dùng `scipy.stats.wilcoxon` bên ngoài sau khi xuất CSV.
