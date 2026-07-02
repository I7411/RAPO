import csv
import json
from pathlib import Path
from utils.plotter import plot_bar_comparison
from utils.console import print_info, print_success, print_error

def replot_from_csv():
    summary_dir = Path("outputs/csv/summary")
    if not summary_dir.exists():
        print_error("Thư mục outputs/csv/summary không tồn tại.")
        return
        
    csv_files = list(summary_dir.glob("*.csv"))
    if not csv_files:
        print_error("Không tìm thấy file CSV summary nào.")
        return
        
    print("\nDanh sách các file Summary CSV hiện có:")
    for i, file_path in enumerate(csv_files, 1):
        print(f"{i}. {file_path.name}")
        
    while True:
        choice = input(f"Chọn file CSV (1-{len(csv_files)}) hoặc 0 để hủy: ").strip()
        if choice == '0':
            return
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(csv_files):
                selected_file = csv_files[idx - 1]
                break
        print("Lựa chọn không hợp lệ.")
        
    # Read summary CSV
    summary_data = []
    try:
        with open(selected_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert strings to float for plotting metrics
                for key, val in row.items():
                    if key not in ["algorithm", "hybrid_type", "algorithm_full_name", "benchmark", "benchmark_category", "timestamp"]:
                        try:
                            row[key] = float(val) if val else None
                        except ValueError:
                            pass
                summary_data.append(row)
    except Exception as e:
        print_error(f"Lỗi khi đọc file {selected_file.name}: {e}")
        return
        
    if not summary_data:
        print_error("File CSV trống.")
        return
        
    # Ask for metric
    available_metrics = [
        "mean_error", "best_error", "worst_error", 
        "mean", "best", "worst", "median", 
        "std", "std_error", "avg_runtime_seconds", 
        "rank_by_mean", "rank_by_mean_error", 
        "success_rate"
    ]
    
    print("\nCác metric có thể vẽ biểu đồ:")
    for i, metric in enumerate(available_metrics, 1):
        print(f"{i}. {metric}")
    print("99. Tất cả các metric")
        
    while True:
        m_choice = input(f"Chọn metric (nhập các số cách nhau bằng dấu phẩy hoặc 99 để chọn tất cả): ").strip()
        if m_choice == '99':
            selected_metrics = available_metrics
            break
            
        selected_metrics = []
        invalid = False
        parts = [p.strip() for p in m_choice.split(',') if p.strip()]
        
        if not parts:
            print("Bạn chưa nhập lựa chọn nào.")
            continue
            
        for p in parts:
            if p.isdigit():
                m_idx = int(p)
                if 1 <= m_idx <= len(available_metrics):
                    selected_metrics.append(available_metrics[m_idx - 1])
                elif m_idx == 99:
                    selected_metrics = available_metrics
                    invalid = False
                    break
                else:
                    print(f"Số '{p}' không hợp lệ.")
                    invalid = True
                    break
            else:
                if p in available_metrics:
                    selected_metrics.append(p)
                else:
                    print(f"Metric '{p}' không tồn tại.")
                    invalid = True
                    break
                    
        if not invalid and selected_metrics:
            selected_metrics = list(dict.fromkeys(selected_metrics))
            break
            
    # Generate bar charts
    try:
        for metric in selected_metrics:
            title = f"Comparison of {metric} from {selected_file.stem}"
            out_name = f"bar_{metric}_{selected_file.stem}.png"
            out_path = Path("outputs/figures/bar") / out_name
            
            plot_bar_comparison(summary_data, metric, title, out_path)
            print_success(f"Đã xuất biểu đồ thành công tại: {out_path}")
    except Exception as e:
        print_error(f"Lỗi khi vẽ biểu đồ: {e}")
