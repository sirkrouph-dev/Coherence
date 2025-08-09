import os
import re
import sys
import ast
import json
import time
import importlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


TARGET_AREAS = [
    "neuron_models",
    "plasticity",
    "neuromodulation",
    "sensory_encoding",
    "simulation_modes",
    "edge_features",
    "performance",
    "visualization_monitoring",
    "api_docs",
]


AREA_KEYWORDS = {
    "neuron_models": [
        "AdEx",
        "Adaptive Exponential",
        "Hodgkin",
        "Leaky Integrate-and-Fire",
        "LIF",
        "Izhikevich",
        "neuron model",
    ],
    "plasticity": [
        "STDP",
        "Short-Term Plasticity",
        "STP",
        "RSTDP",
        "Hebbian",
        "plasticity",
    ],
    "neuromodulation": [
        "Dopamine",
        "Serotonin",
        "Acetylcholine",
        "Norepinephrine",
        "Neuromodulatory",
        "Homeostatic",
        "Reward",
    ],
    "sensory_encoding": [
        "Retinal",
        "Cochlear",
        "Somatosensory",
        "encoder",
        "MultiModal",
    ],
    "simulation_modes": [
        "event-driven",
        "time-driven",
        "simulation",
        "event driven",
        "time step",
    ],
    "edge_features": [
        "Jetson",
        "edge",
        "deployment",
        "power",
        "resource",
    ],
    "performance": [
        "1000x",
        "real-time",
        "efficiency",
        "memory",
        "speed",
        "scalability",
        "throughput",
    ],
    "visualization_monitoring": [
        "visualization",
        "plot",
        "monitoring",
        "raster",
        "heatmap",
    ],
    "api_docs": [
        "API",
        "documentation",
        "tutorial",
        "reference",
    ],
}


@dataclass
class Claim:
    area: str
    text: str
    file: str
    line: int


@dataclass
class Symbol:
    path: str
    module: str
    symbol_type: str  # class/function
    name: str
    doc: str


@dataclass
class MatchResult:
    claim: Claim
    status: str  # Implemented | Partial | Missing
    evidence: List[str]


def read_text_lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []


def detect_area(line: str) -> Optional[str]:
    lower = line.lower()
    for area, kws in AREA_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in lower:
                return area
    return None


def scan_marketing_claims(root: Path) -> List[Claim]:
    claim_files = [
        root / "README.md",
        root / "docs",
        root / "JETSON_DEPLOYMENT.md",
        root / "docs" / "benchmarks.md",
        root / "ARCHITECTURE.md",
    ]
    claims: List[Claim] = []

    def scan_file(path: Path):
        lines = read_text_lines(path)
        for i, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            area = detect_area(line)
            if area:
                claims.append(Claim(area=area, text=line.strip(), file=str(path.relative_to(root)), line=i))

    for target in claim_files:
        if target.is_file():
            scan_file(target)
        elif target.is_dir():
            for p in target.rglob("*.md"):
                scan_file(p)
    return claims


def inventory_symbols(root: Path) -> List[Symbol]:
    symbols: List[Symbol] = []
    ignore_dirs = {"venv", "venv_neuron", "__pycache__", ".git", "docs/build", ".benchmarks", ".pytest_cache"}
    for py in root.rglob("*.py"):
        rel = py.relative_to(root)
        if any(part in ignore_dirs for part in rel.parts):
            continue
        try:
            src = py.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(src)
        except Exception:
            continue

        module = ".".join(rel.with_suffix("").parts)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                symbols.append(Symbol(str(rel), module, "class", node.name, doc.strip().splitlines()[0] if doc else ""))
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                symbols.append(Symbol(str(rel), module, "function", node.name, doc.strip().splitlines()[0] if doc else ""))
    return symbols


def heuristic_match(claim: Claim, sym: Symbol) -> bool:
    text = claim.text.lower()
    name = sym.name.lower()
    doc = sym.doc.lower()
    # Simple heuristics
    keywords = AREA_KEYWORDS.get(claim.area, [])
    if any(kw.lower() in name or kw.lower() in doc for kw in keywords):
        return True
    # Specific mapping examples
    pairs = [
        ("adex", ["adaptiveexponential", "adex", "adaptive_exponential", "adaptiveexponentialintegrateandfire"]),
        ("hodgkin", ["hodgkin", "hh"]),
        ("lif", ["leaky", "lif"]),
    ]
    for key, patterns in pairs:
        if key in text:
            if any(p in name for p in patterns):
                return True
    return False


def cross_check_claims(claims: List[Claim], symbols: List[Symbol]) -> List[MatchResult]:
    results: List[MatchResult] = []
    for claim in claims:
        matched = [s for s in symbols if heuristic_match(claim, s)]
        if matched:
            # Default to Implemented; dynamic tests may refine later
            evidence = [f"{m.symbol_type} {m.name} ({m.path})" for m in matched[:5]]
            results.append(MatchResult(claim, "Implemented", evidence))
        else:
            results.append(MatchResult(claim, "Missing", []))
    return results


def dynamic_smoke_tests(symbols: List[Symbol]) -> Dict[str, Dict[str, Any]]:
    outcomes: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        key = f"{sym.module}:{sym.name}"
        mod_name = sym.module
        try:
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, sym.name)
            instantiated = False
            # Try minimal instantiation for classes with common signatures
            if sym.symbol_type == "class":
                try:
                    # Common constructor patterns in this repo: neuron_id or synapse_id
                    if "neurons" in sym.path.lower():
                        obj(neuron_id=0)
                        instantiated = True
                    elif "synapse" in sym.path.lower():
                        obj(synapse_id=0, pre_neuron_id=0, post_neuron_id=1)
                        instantiated = True
                    else:
                        obj()  # best-effort
                        instantiated = True
                except Exception as e:
                    outcomes[key] = {"import": "ok", "instantiate": "fail", "error": repr(e)}
                else:
                    outcomes[key] = {"import": "ok", "instantiate": "ok"}
            else:
                outcomes[key] = {"import": "ok"}
        except Exception as e:
            outcomes[key] = {"import": "fail", "error": repr(e)}
    return outcomes


def quick_benchmarks() -> Dict[str, Any]:
    """Run small CPU benchmark; detect GPU availability; skip heavy runs."""
    metrics: Dict[str, Any] = {"environment": {}, "cpu_micro": {}, "gpu": {}}
    metrics["environment"] = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }

    # CPU micro benchmark
    try:
        start = time.time()
        from core.network import NeuromorphicNetwork

        net = NeuromorphicNetwork()
        net.add_layer("input", 50, "lif")
        net.add_layer("hidden", 25, "adex")
        net.add_layer("output", 10, "lif")
        net.connect_layers("input", "hidden", connection_probability=0.1)
        net.connect_layers("hidden", "output", connection_probability=0.2)
        results = net.run_simulation(duration=50.0, dt=0.1)
        wall = time.time() - start
        sim_ms = 50.0
        speed_x = (sim_ms / 1000.0) / wall if wall > 0 else 0.0
        metrics["cpu_micro"] = {
            "duration_ms": sim_ms,
            "wall_s": wall,
            "speed_x_real_time": round(speed_x, 2),
            "spikes_recorded": sum(len(v) for v in results.get("layer_spike_times", {}).values()) if isinstance(results, dict) else None,
        }
    except Exception as e:
        metrics["cpu_micro"] = {"error": repr(e)}

    # GPU capability probe
    gpu_info: Dict[str, Any] = {"cupy": False, "torch": False}
    try:
        import cupy

        gpu_info["cupy"] = True
    except Exception:
        pass
    try:
        import torch

        gpu_info["torch"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    metrics["gpu"] = gpu_info
    return metrics


def visualization_check() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        from api.neuromorphic_api import NeuromorphicVisualizer

        viz = NeuromorphicVisualizer()
        # Minimal call smoke without rendering heavy
        info["import"] = "ok"
        info["methods"] = [
            m
            for m in dir(viz)
            if m.startswith("plot_") and callable(getattr(viz, m, None))
        ]
    except Exception as e:
        info["import"] = "fail"
        info["error"] = repr(e)
    return info


def api_docs_review(root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"missing_symbols": [], "present_symbols": []}
    api_ref = root / "docs" / "API_REFERENCE.md"
    if not api_ref.exists():
        return report
    text = api_ref.read_text(encoding="utf-8", errors="ignore")
    # crude symbol mentions like api.add_processing_layer or class names
    symbol_names = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text))
    # only check a subset that looks like classes/functions by capitalization or underscores
    candidates = [s for s in symbol_names if s[0].isupper() or "_" in s]
    for name in sorted(set(candidates)):
        found = False
        for py in root.rglob("*.py"):
            try:
                src = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if re.search(rf"\bclass\s+{re.escape(name)}\b|def\s+{re.escape(name)}\b", src):
                found = True
                break
        (report["present_symbols"] if found else report["missing_symbols"]).append(name)
    return report


def dump_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    out_dir = ROOT / "archive" / "reports" / "latest_claims_assessment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse Marketing Claims
    claims = scan_marketing_claims(ROOT)
    claims_json = [asdict(c) for c in claims]
    dump_json(out_dir / "claims.json", claims_json)

    # 2. Inventory Codebase Implementations
    symbols = inventory_symbols(ROOT)
    impl_index = [asdict(s) for s in symbols]
    dump_json(out_dir / "implementations.json", impl_index)

    # 3. Static Cross-Check
    cross = cross_check_claims(claims, symbols)
    cross_json = [
        {"claim": asdict(c.claim), "status": c.status, "evidence": c.evidence}
        for c in cross
    ]
    dump_json(out_dir / "crosscheck.json", cross_json)

    # 4. Dynamic Import & Smoke Tests
    smoke = dynamic_smoke_tests(symbols[:200])  # limit for speed
    dump_json(out_dir / "smoke_tests.json", smoke)

    # 5. Performance & Resource Benchmarking (quick)
    bench = quick_benchmarks()
    dump_json(out_dir / "benchmarks.json", bench)

    # 8. Visualization & Real-Time Monitoring Check
    viz = visualization_check()
    dump_json(out_dir / "visualization.json", viz)

    # 9. API & Documentation Completeness Review
    api_rev = api_docs_review(ROOT)
    dump_json(out_dir / "api_docs_review.json", api_rev)

    # 10. Aggregate Markdown Report (brief)
    # Summaries
    implemented = sum(1 for c in cross if c.status == "Implemented")
    missing = sum(1 for c in cross if c.status == "Missing")
    partial = sum(1 for c in cross if c.status == "Partial")
    total = len(cross)
    summary = [
        f"Total claims: {total}",
        f"Implemented: {implemented}",
        f"Partial: {partial}",
        f"Missing: {missing}",
        f"CPU micro speed (x real-time): {bench.get('cpu_micro', {}).get('speed_x_real_time')}",
        f"GPU available (CuPy/Torch): {bench.get('gpu', {}).get('cupy')}/{bench.get('gpu', {}).get('torch')}",
        f"Visualizer import: {viz.get('import')}",
    ]
    report_md = [
        "# Claims vs Implementation Report (Latest)",
        "",
        "## Summary",
        "\n".join(summary),
        "",
        "## Notable Missing Claims",
    ]
    for c in cross[:200]:
        if c.status == "Missing":
            report_md.append(f"- [{c.claim.area}] {c.claim.text} ({c.claim.file}:{c.claim.line})")
    (out_dir / "report.md").write_text("\n".join(report_md), encoding="utf-8")

    print(json.dumps({
        "outputs": {
            "dir": str(out_dir.relative_to(ROOT)),
            "claims": "claims.json",
            "implementations": "implementations.json",
            "crosscheck": "crosscheck.json",
            "smoke_tests": "smoke_tests.json",
            "benchmarks": "benchmarks.json",
            "visualization": "visualization.json",
            "api_docs_review": "api_docs_review.json",
            "report": "report.md",
        }
    }, indent=2))


if __name__ == "__main__":
    main()


