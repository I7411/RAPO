import datetime

def generate_filename(prefix: str, algorithm: str, hybrid_type: str, benchmark: str, dimension: int, ext: str) -> str:
    """
    Tạo tên file theo chuẩn: <PREFIX>__<TEN_THUAT_TOAN>__<PHEP_LAI>__<BENCHMARK>__D<DIM>__<TIMESTAMP>
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hybrid = hybrid_type if hybrid_type else "original"
    return f"{prefix}__{algorithm}__{hybrid}__{benchmark}__D{dimension}__{timestamp}.{ext}"

def generate_summary_filename(prefix: str, benchmark: str, dimension: int, ext: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}__all_algorithms__{benchmark}__D{dimension}__{timestamp}.{ext}"

def generate_overall_filename(prefix: str, metric: str, dimension: int, ext: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{metric}__all_algorithms__all_benchmarks__D{dimension}__{timestamp}.{ext}"
