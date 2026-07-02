import inspect
from typing import Dict, Type, List, Any
from algorithms.base_optimizer import BaseOptimizer
from core.problem import Problem
from core.evaluator import Evaluator
from core.result import OptimizationResult

from algorithms.aro import ARO
from algorithms.poa import POA
from algorithms.gwo import GWO
from algorithms.pso import PSO
from algorithms.ga import GA
from algorithms.hho import HHO
from algorithms.ga_pso import GA_PSO

# Nhập đúng tên file và class thực tế trong thư mục hybrids
from algorithms.hybrids.rapo_chuyenphathichnghi import RAPOEnergySwitch
from algorithms.hybrids.rapo_cungpha_aro_poa import RAPOSequentialOperatorChainAROPOA
from algorithms.hybrids.rapo_cungpha_poa_aro import RAPOReverseSequentialOperatorChainPOAARO
from algorithms.hybrids.rapo_dao import RAPOIslandModel
from algorithms.hybrids.rapo_ensemble_dachienluoc import RAPOMultiStrategyEnsemble
from algorithms.hybrids.rapo_epr_aro_poa import RAPOExplorationReplacementAROPOA
from algorithms.hybrids.rapo_epr_poa_aro import RAPOExplorationReplacementPOAARO
from algorithms.hybrids.rapo_hauchinh_aro_poa import RAPOPostOptimizationAROPOA
from algorithms.hybrids.rapo_hauchinh_poa_aro import RAPOPostOptimizationPOAARO
from algorithms.hybrids.rapo_quanthe import RAPOPopulationHybrid
from algorithms.hybrids.rapo_songsong import RAPOParallelHybrid
from algorithms.hybrids.rapo_tinhhoa_aro_poa import RAPOEliteGuidedAROPOA
from algorithms.hybrids.rapo_tinhhoa_poa_aro import RAPOEliteGuidedPOAARO
from algorithms.hybrids.rapo_toantu import RAPOOperatorLevelHybrid
from algorithms.hybrids.rapo_toantuthichnghi import RAPOOperatorSelection
from algorithms.hybrids.rapo_tritre_aro_poa import RAPOStagnationTriggeredAROPOA
from algorithms.hybrids.rapo_tritre_poa_aro import RAPOStagnationTriggeredPOAARO
from algorithms.hybrids.rapo_tuantu import RAPOSequential
from algorithms.hybrids.rapo_xacsuatchontoantu import RAPOProbabilisticRoulette
from algorithms.hybrids.rapo_xpr_aro_poa import RAPOExploitationReplacementAROPOA
from algorithms.hybrids.rapo_xpr_poa_aro import RAPOExploitationReplacementPOAARO

_registry: Dict[str, Type[BaseOptimizer]] = {}

def register_algorithm(name: str, optimizer_class: Type[BaseOptimizer]):
    _registry[name] = optimizer_class

def get_algorithm(name: str) -> Type[BaseOptimizer]:
    if name not in _registry:
        raise ValueError(f"Algorithm '{name}' không tồn tại trong registry.")
    return _registry[name]

def list_algorithms() -> List[str]:
    return list(_registry.keys())

def list_algorithms_by_type(type_name: str) -> List[str]:
    results = []
    for name, cls in _registry.items():
        if type_name in ["base", "baseline"] and cls.hybrid_type == "":
            results.append(name)
        elif type_name == "hybrid" and cls.hybrid_type != "":
            results.append(name)
        elif type_name == "rapo" and "RAPO" in name:
            results.append(name)
    return results

def create_rapo_wrapper(name: str, backend_class: Type) -> Type[BaseOptimizer]:
    """Tạo lớp bọc (wrapper) để tương thích với kiến trúc BaseOptimizer"""
    class WrappedRAPO(BaseOptimizer):
        def __init__(self):
            self.name = name
            self.hybrid_type = "RAPO"
            
        def optimize(self, problem: Problem, config: Dict[str, Any], seed: int = None) -> OptimizationResult:
            pop_size = config.get("population_size", 50)
            max_iter = config.get("max_iterations", 100)
            use_budget = config.get("use_evaluation_budget", False)
            max_evals = config.get("max_function_evaluations", 10000) if use_budget else None
            
            # Khởi tạo backend instance
            sig = inspect.signature(backend_class.__init__)
            kwargs = {}
            if "max_evaluations" in sig.parameters:
                kwargs["max_evaluations"] = max_evals
                
            backend_instance = backend_class(
                population_size=pop_size,
                max_iterations=max_iter,
                seed=seed,
                **kwargs
            )
            
            # Cập nhật bounds nếu class backend hỗ trợ
            if hasattr(backend_instance, "lower_bound"):
                backend_instance.lower_bound = problem.lower_bound
                backend_instance.upper_bound = problem.upper_bound
            elif hasattr(backend_instance, "lower_bound_input"):
                backend_instance.lower_bound_input = problem.lower_bound
                backend_instance.upper_bound_input = problem.upper_bound
                
            evaluator = Evaluator(problem, config.get("minimization", True))
            evaluator.reset()
            
            # Khởi chạy hàm optimize của backend
            sig_opt = inspect.signature(backend_instance.optimize)
            opt_kwargs = {
                "objective_func": evaluator.evaluate,
                "dimension": problem.dimension,
            }
            if "lower_bound" in sig_opt.parameters:
                opt_kwargs["lower_bound"] = problem.lower_bound
                opt_kwargs["upper_bound"] = problem.upper_bound
                
            raw_res = backend_instance.optimize(**opt_kwargs)
            
            # Chuyển đổi kết quả về chuẩn OptimizationResult của hệ thống
            final_error = abs(raw_res.best_fitness - problem.global_minimum) if problem.global_minimum is not None else None
            conv = getattr(raw_res, "convergence_curve", getattr(raw_res, "convergence", []))
            
            return OptimizationResult(
                algorithm=self.name,
                hybrid_type=self.hybrid_type,
                benchmark=problem.name,
                dimension=problem.dimension,
                run_id=config.get("run_id", 1),
                seed=seed,
                best_fitness=raw_res.best_fitness,
                best_solution=raw_res.best_solution.tolist(),
                convergence_curve=list(conv),
                runtime_seconds=raw_res.runtime_seconds,
                population_size=pop_size,
                max_iterations=max_iter,
                lower_bound=problem.lower_bound,
                upper_bound=problem.upper_bound,
                nfe=evaluator.nfe,
                final_error=final_error,
                metadata={"evaluations": evaluator.nfe}
            )
            
    WrappedRAPO.name = name
    WrappedRAPO.hybrid_type = "RAPO"
    return WrappedRAPO

# Tự động đăng ký
register_algorithm("ARO", ARO)
register_algorithm("POA", POA)
register_algorithm("GWO", GWO)
register_algorithm("PSO", PSO)
register_algorithm("GA", GA)
register_algorithm("HHO", HHO)
register_algorithm("GA_PSO", GA_PSO)

# Đăng ký các thuật toán RAPO bằng Wrapper
register_algorithm("RAPO_EnergySwitch", create_rapo_wrapper("RAPO_EnergySwitch", RAPOEnergySwitch))
register_algorithm("RAPO_SequentialOperatorChain_ARO_POA", create_rapo_wrapper("RAPO_SequentialOperatorChain_ARO_POA", RAPOSequentialOperatorChainAROPOA))
register_algorithm("RAPO_ReverseSequentialOperatorChain_POA_ARO", create_rapo_wrapper("RAPO_ReverseSequentialOperatorChain_POA_ARO", RAPOReverseSequentialOperatorChainPOAARO))
register_algorithm("RAPO_IslandModel", create_rapo_wrapper("RAPO_IslandModel", RAPOIslandModel))
register_algorithm("RAPO_MultiStrategyEnsemble", create_rapo_wrapper("RAPO_MultiStrategyEnsemble", RAPOMultiStrategyEnsemble))
register_algorithm("RAPO_ExplorationReplacement_ARO_POA", create_rapo_wrapper("RAPO_ExplorationReplacement_ARO_POA", RAPOExplorationReplacementAROPOA))
register_algorithm("RAPO_ExplorationReplacement_POA_ARO", create_rapo_wrapper("RAPO_ExplorationReplacement_POA_ARO", RAPOExplorationReplacementPOAARO))
register_algorithm("RAPO_PostOptimization_ARO_POA", create_rapo_wrapper("RAPO_PostOptimization_ARO_POA", RAPOPostOptimizationAROPOA))
register_algorithm("RAPO_PostOptimization_POA_ARO", create_rapo_wrapper("RAPO_PostOptimization_POA_ARO", RAPOPostOptimizationPOAARO))
register_algorithm("RAPO_PopulationHybrid", create_rapo_wrapper("RAPO_PopulationHybrid", RAPOPopulationHybrid))
register_algorithm("RAPO_ParallelHybrid", create_rapo_wrapper("RAPO_ParallelHybrid", RAPOParallelHybrid))
register_algorithm("RAPO_EliteGuided_ARO_POA", create_rapo_wrapper("RAPO_EliteGuided_ARO_POA", RAPOEliteGuidedAROPOA))
register_algorithm("RAPO_EliteGuided_POA_ARO", create_rapo_wrapper("RAPO_EliteGuided_POA_ARO", RAPOEliteGuidedPOAARO))
register_algorithm("RAPO_OperatorLevelHybrid", create_rapo_wrapper("RAPO_OperatorLevelHybrid", RAPOOperatorLevelHybrid))
register_algorithm("RAPO_OperatorSelection", create_rapo_wrapper("RAPO_OperatorSelection", RAPOOperatorSelection))
register_algorithm("RAPO_StagnationTriggered_ARO_POA", create_rapo_wrapper("RAPO_StagnationTriggered_ARO_POA", RAPOStagnationTriggeredAROPOA))
register_algorithm("RAPO_StagnationTriggered_POA_ARO", create_rapo_wrapper("RAPO_StagnationTriggered_POA_ARO", RAPOStagnationTriggeredPOAARO))
register_algorithm("RAPO_Sequential", create_rapo_wrapper("RAPO_Sequential", RAPOSequential))
register_algorithm("RAPO_ProbabilisticRoulette", create_rapo_wrapper("RAPO_ProbabilisticRoulette", RAPOProbabilisticRoulette))
register_algorithm("RAPO_ExploitationReplacement_ARO_POA", create_rapo_wrapper("RAPO_ExploitationReplacement_ARO_POA", RAPOExploitationReplacementAROPOA))
register_algorithm("RAPO_ExploitationReplacement_POA_ARO", create_rapo_wrapper("RAPO_ExploitationReplacement_POA_ARO", RAPOExploitationReplacementPOAARO))
