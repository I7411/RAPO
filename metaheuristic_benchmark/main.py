import sys
import json
from pathlib import Path
from experiments.run_single import run_single
from experiments.run_comparison import run_comparison
from experiments.run_all_benchmarks import run_all_benchmarks
from experiments.replot_from_csv import replot_from_csv
from benchmarks.benchmark_registry import list_benchmarks, get_benchmark
from algorithms.algorithm_registry import list_algorithms, list_algorithms_by_type

RAPO_ABBREVIATIONS = {
    "RAPO_Sequential": "RAPO-SEQ",
    "RAPO_EnergySwitch": "RAPO-ESW",
    "RAPO_SequentialOperatorChain_ARO_POA": "RAPO-SOC-AP",
    "RAPO_ReverseSequentialOperatorChain_POA_ARO": "RAPO-SOC-PA",
    "RAPO_IslandModel": "RAPO-IM",
    "RAPO_MultiStrategyEnsemble": "RAPO-MSE",
    "RAPO_ExplorationReplacement_ARO_POA": "RAPO-EPR-AP",
    "RAPO_ExplorationReplacement_POA_ARO": "RAPO-EPR-PA",
    "RAPO_ExploitationReplacement_ARO_POA": "RAPO-XPR-AP",
    "RAPO_ExploitationReplacement_POA_ARO": "RAPO-XPR-PA",
    "RAPO_PostOptimization_ARO_POA": "RAPO-POST-AP",
    "RAPO_PostOptimization_POA_ARO": "RAPO-POST-PA",
    "RAPO_PopulationHybrid": "RAPO-PH",
    "RAPO_ParallelHybrid": "RAPO-PAR",
    "RAPO_EliteGuided_ARO_POA": "RAPO-EG-AP",
    "RAPO_EliteGuided_POA_ARO": "RAPO-EG-PA",
    "RAPO_OperatorLevelHybrid": "RAPO-OLH",
    "RAPO_OperatorSelection": "RAPO-AOS",
    "RAPO_ProbabilisticRoulette": "RAPO-PR",
    "RAPO_StagnationTriggered_ARO_POA": "RAPO-STG-AP",
    "RAPO_StagnationTriggered_POA_ARO": "RAPO-STG-PA",
}

def format_alg_name(alg_name: str) -> str:
    if alg_name in RAPO_ABBREVIATIONS:
        return RAPO_ABBREVIATIONS[alg_name]
    return alg_name

def print_menu():
    print("\n" + "="*52)
    print(" METAHEURISTIC BENCHMARK FRAMEWORK")
    print("="*52)
    print("1. Chạy 1 thuật toán trên 1 benchmark")
    print("2. So sánh nhiều thuật toán trên 1 benchmark")
    print("3. Chạy tất cả thuật toán trên tất cả benchmark")
    print("4. Chạy riêng nhóm thuật toán lai RAPO")
    print("5. Xuất lại biểu đồ từ CSV có sẵn")
    print("6. Xem và chỉnh sửa cấu hình chạy (dim, iter, pop_size, seed, bounds)")
    print("7. Liệt kê thuật toán hiện có")
    print("8. Liệt kê benchmark hiện có")
    print("9. Chạy benchmark CEC2017")
    print("10. Chạy benchmark CEC2022")
    print("0. Thoát")
    print("="*52)

def load_config(file_name: str) -> dict:
    try:
        with open(f"configs/{file_name}", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi đọc {file_name}: {e}")
        return {}

def select_algorithm() -> str:
    algs = list_algorithms()
    print("\nDanh sách thuật toán hiện có:")
    for i, alg in enumerate(algs, 1):
        print(f"{i}. {format_alg_name(alg)}")
    
    while True:
        choice = input(f"Chọn thuật toán (nhập số 1-{len(algs)} hoặc nhập tên): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(algs):
                return algs[idx - 1]
        elif choice in algs:
            return choice
        else:
            for alg in algs:
                if choice == RAPO_ABBREVIATIONS.get(alg, ""):
                    return alg
        print("Lựa chọn không hợp lệ, vui lòng thử lại.")

def select_benchmark() -> str:
    benchs = list_benchmarks()
    print("\nDanh sách benchmark cơ bản hiện có:")
    for i, b in enumerate(benchs, 1):
        print(f"{i}. {b}")
    
    while True:
        choice = input(f"Chọn benchmark (nhập số 1-{len(benchs)} hoặc nhập tên): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(benchs):
                return benchs[idx - 1]
        elif choice in benchs:
            return choice
        print("Lựa chọn không hợp lệ, vui lòng thử lại.")

def select_multiple_algorithms() -> list:
    algs = list_algorithms()
    print("\nDanh sách thuật toán hiện có:")
    for i, alg in enumerate(algs, 1):
        print(f"{i}. {format_alg_name(alg)}")
    print("99. Tất cả các thuật toán")
    
    while True:
        print("\nNhập các số thứ tự hoặc tên thuật toán, cách nhau bằng dấu phẩy (VD: 1,2,5 hoặc ARO,POA). Nhập 99 để chọn tất cả.")
        choice_str = input("Lựa chọn của bạn: ").strip()
        
        if choice_str == '99':
            return algs
            
        selected = []
        invalid = False
        parts = [p.strip() for p in choice_str.split(',') if p.strip()]
        
        if not parts:
            print("Bạn chưa nhập lựa chọn nào.")
            continue
            
        for p in parts:
            if p.isdigit():
                idx = int(p)
                if 1 <= idx <= len(algs):
                    selected.append(algs[idx - 1])
                elif idx == 99:
                    return algs
                else:
                    print(f"Số '{p}' không hợp lệ.")
                    invalid = True
                    break
            else:
                if p in algs:
                    selected.append(p)
                else:
                    found = False
                    for alg in algs:
                        if p == RAPO_ABBREVIATIONS.get(alg, ""):
                            selected.append(alg)
                            found = True
                            break
                    if not found:
                        print(f"Thuật toán '{p}' không tồn tại.")
                        invalid = True
                        break
                    
        if not invalid and selected:
            # Loại bỏ trùng lặp và giữ nguyên thứ tự
            return list(dict.fromkeys(selected))

def main():
    while True:
        print_menu()
        choice = input("Chọn chức năng (0-10): ").strip()
        
        if choice == '0':
            print("Đã thoát chương trình.")
            break
            
        elif choice == '1':
            cfg = load_config("default_config.json")
            default_dim = cfg.get("dimension", 30)
            alg = select_algorithm()
            bench = select_benchmark()
            dim_str = input(f"Nhập số chiều (mặc định {default_dim}): ").strip()
            dim = int(dim_str) if dim_str.isdigit() else default_dim
            run_single(alg, bench, dim)
            
        elif choice == '2':
            cfg = load_config("default_config.json")
            default_dim = cfg.get("dimension", 30)
            algs = select_multiple_algorithms()
            bench = select_benchmark()
            dim_str = input(f"Nhập số chiều (mặc định {default_dim}): ").strip()
            dim = int(dim_str) if dim_str.isdigit() else default_dim
            run_comparison(algs, bench, dim)
            
        elif choice == '3':
            exp_cfg = load_config("experiment_config.json")
            if "experiments" in exp_cfg and "full_comparison" in exp_cfg["experiments"]:
                cfg = exp_cfg["experiments"]["full_comparison"]
                run_all_benchmarks(cfg["algorithms"], cfg["benchmarks"], cfg.get("dimension", 30), cfg)
            else:
                print("Không tìm thấy cấu hình 'full_comparison' trong experiment_config.json")
                
        elif choice == '4':
            cfg = load_config("default_config.json")
            default_dim = cfg.get("dimension", 30)
            rapo_algs = list_algorithms_by_type("rapo")
            bench = select_benchmark()
            dim_str = input(f"Nhập số chiều (mặc định {default_dim}): ").strip()
            dim = int(dim_str) if dim_str.isdigit() else default_dim
            run_comparison(rapo_algs, bench, dim)
            
        elif choice == '5':
            replot_from_csv()
            
        elif choice == '6':
            print("\n--- Cấu hình hiện tại (default_config) ---")
            cfg = load_config("default_config.json")
            print(json.dumps(cfg, indent=2))
            
            edit = input("\nBạn có muốn chỉnh sửa cấu hình không? (y/n): ").strip().lower()
            if edit == 'y':
                print("Nhập giá trị mới (hoặc ấn Enter để giữ nguyên):")
                
                dim_str = input(f"Số chiều [hiện tại: {cfg.get('dimension')}]: ").strip()
                if dim_str: cfg["dimension"] = int(dim_str)
                
                iter_str = input(f"Số vòng lặp tối đa [hiện tại: {cfg.get('max_iterations')}]: ").strip()
                if iter_str: cfg["max_iterations"] = int(iter_str)
                
                pop_str = input(f"Số cá thể [hiện tại: {cfg.get('population_size')}]: ").strip()
                if pop_str: cfg["population_size"] = int(pop_str)
                
                seed_str = input(f"Seed bắt đầu [hiện tại: {cfg.get('seed_start')}]: ").strip()
                if seed_str: cfg["seed_start"] = int(seed_str)
                
                runs_str = input(f"Số lần chạy (runs) [hiện tại: {cfg.get('runs')}]: ").strip()
                if runs_str: cfg["runs"] = int(runs_str)
                
                lb_str = input(f"Cận dưới (hiện tại: {cfg.get('lower_bound')}, bỏ trống để dùng mặc định của hàm): ").strip()
                if lb_str:
                    cfg["lower_bound"] = float(lb_str)
                elif lb_str == "" and "lower_bound" in cfg and input("Bạn muốn xóa cận dưới đang có để dùng mặc định hàm? (y/n): ").strip().lower() == 'y':
                    cfg["lower_bound"] = None
                    
                ub_str = input(f"Cận trên (hiện tại: {cfg.get('upper_bound')}, bỏ trống để dùng mặc định của hàm): ").strip()
                if ub_str:
                    cfg["upper_bound"] = float(ub_str)
                elif ub_str == "" and "upper_bound" in cfg and input("Bạn muốn xóa cận trên đang có để dùng mặc định hàm? (y/n): ").strip().lower() == 'y':
                    cfg["upper_bound"] = None
                
                try:
                    with open("configs/default_config.json", "w") as f:
                        json.dump(cfg, f, indent=2)
                    print("Đã lưu cấu hình mới thành công!")
                except Exception as e:
                    print(f"Lỗi khi lưu cấu hình: {e}")
            
        elif choice == '7':
            print("\nDanh sách thuật toán:")
            for alg in list_algorithms():
                print(f"- {format_alg_name(alg)}")
                
        elif choice == '8':
            print("\nDanh sách benchmark cơ bản:")
            for b in list_benchmarks():
                try:
                    prob = get_benchmark(b, 2)
                    cat = prob.category
                    dim_type = "fixed_dimension | D2 only" if prob.is_fixed_dimension else "n-dimensional"
                    gmin = prob.global_minimum
                    print(f"- {b} | {cat} | {dim_type} | global_minimum = {gmin}")
                except Exception:
                    print(f"- {b}")
                    
        elif choice == '9':
            cfg = load_config("default_config.json")
            default_dim = cfg.get("dimension", 30)
            algs = select_multiple_algorithms()
            
            print("Nhập ID của hàm CEC2017 (1 đến 30). Nhập 99 để chạy toàn bộ 30 hàm (trừ F2).")
            func_choice = input("Lựa chọn (1-30 hoặc 99): ").strip()
            dim_str = input(f"Nhập số chiều (10, 30, 50, 100) [mặc định {default_dim}]: ").strip()
            dim = int(dim_str) if dim_str.isdigit() else default_dim
            
            if func_choice == '99':
                benchmarks = [f"CEC2017_F{i}" for i in range(1, 31) if i != 2]
                print(f"Bắt đầu chạy CEC2017 toàn bộ ({len(benchmarks)} hàm) với {len(algs)} thuật toán ở số chiều D={dim}")
                run_all_benchmarks(algs, benchmarks, dim, cfg)
            else:
                try:
                    func_id = int(func_choice)
                    bench = f"CEC2017_F{func_id}"
                    if len(algs) == 1:
                        run_single(algs[0], bench, dim)
                    else:
                        run_comparison(algs, bench, dim)
                except ValueError:
                    print("Lựa chọn không hợp lệ.")
                    
        elif choice == '10':
            cfg = load_config("default_config.json")
            default_dim = cfg.get("dimension", 10)
            algs = select_multiple_algorithms()
            
            print("Nhập ID của hàm CEC2022 (1 đến 12). Nhập 99 để chạy toàn bộ 12 hàm.")
            func_choice = input("Lựa chọn (1-12 hoặc 99): ").strip()
            dim_str = input(f"Nhập số chiều (2, 10, 20) [mặc định {default_dim}]: ").strip()
            dim = int(dim_str) if dim_str.isdigit() else default_dim
            
            if func_choice == '99':
                if dim == 2:
                    print("Lưu ý: F6, F7, F8 không hỗ trợ D=2 nên sẽ tự động bị bỏ qua.")
                    benchmarks = [f"CEC2022_F{i}" for i in range(1, 13) if i not in (6, 7, 8)]
                else:
                    benchmarks = [f"CEC2022_F{i}" for i in range(1, 13)]
                print(f"Bắt đầu chạy CEC2022 toàn bộ ({len(benchmarks)} hàm) với {len(algs)} thuật toán ở số chiều D={dim}")
                run_all_benchmarks(algs, benchmarks, dim, cfg)
            else:
                try:
                    func_id = int(func_choice)
                    if dim == 2 and func_id in (6, 7, 8):
                        print(f"Lỗi: Hàm CEC2022_F{func_id} không hỗ trợ số chiều D=2.")
                        continue
                        
                    bench = f"CEC2022_F{func_id}"
                    if len(algs) == 1:
                        run_single(algs[0], bench, dim)
                    else:
                        run_comparison(algs, bench, dim)
                except ValueError:
                    print("Lựa chọn không hợp lệ.")
                
        else:
            print("Lựa chọn không hợp lệ, vui lòng nhập số từ 0-10.")

if __name__ == "__main__":
    Path("outputs/csv/raw").mkdir(parents=True, exist_ok=True)
    Path("outputs/csv/summary").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures/convergence").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures/boxplot").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures/bar").mkdir(parents=True, exist_ok=True)
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path("outputs/metadata").mkdir(parents=True, exist_ok=True)
    main()
