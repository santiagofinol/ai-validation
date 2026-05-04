"""
AI Output Validation Test Engine
Runs consistency benchmarks against local LLM (Ollama) and auto-generates analysis.
"""

import ollama
import json
import time
import yaml
import sys
from pathlib import Path

# Import our analyzer
from analyzer import analyze_runs, generate_report


def load_config(path="config.yaml"):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def warmup_model(model_name, prompt="Say hello"):
    """Send a dummy request to load model into memory before benchmarking."""
    print(f" Warming up {model_name}...")
    try:
        ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        print("✅ Warmup complete.")
    except Exception as e:
        print(f"⚠️ Warmup failed (non-critical): {e}")


def run_consistency_bench(cfg):
    """Run N iterations of the same prompt and collect results."""
    model = cfg["model"]
    temp = cfg["temperature"]
    iters = cfg["iterations"]
    prompt = cfg["prompt"]
    output_file = cfg["output_file"]
    analysis_file = cfg["analysis_file"]

    results = []
    print(f" Starting Consistency Benchmark")
    print(f"   Model: {model} | Temperature: {temp} | Iterations: {iters}")
    print(f"   Prompt: {prompt[:60]}...")
    print()

    for i in range(iters):
        start_time = time.time()
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            duration = time.time() - start_time
            output = response["message"]["content"]
            status = "✅"
        except Exception as e:
            duration = time.time() - start_time
            output = f"[ERROR: {e}]"
            status = "❌"

        results.append({
            "iteration": i + 1,
            "latency_sec": round(duration, 2),
            "output": output,
            "status": "ok" if status == "✅" else "error"
        })
        print(f"{status} Run {i+1}/{iters} complete ({results[-1]['latency_sec']}s)")

    # Save raw results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n Raw results saved to {output_file}")

    # Load reference answer if available
    reference_text = ""
    ref_path = cfg.get("reference_file", "reference_answer.txt")
    if Path(ref_path).exists():
        with open(ref_path, "r") as f:
            reference_text = f.read()
        print(f" Loaded reference answer from {ref_path}")

    # Run analysis
    print("\n Running analysis...")
    analysis = analyze_runs(results, reference_text)

    # Save analysis
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=4)
    print(f" Analysis saved to {analysis_file}")

    # Generate report
    report = generate_report(results, analysis, model_name=model)
    report_path = output_file.replace(".json", "_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f" Validation report saved to {report_path}")

    # Print summary
    score = analysis["score_data"]["total_score"]
    print(f"\n{'='*50}")
    print(f"  OVERALL RELIABILITY SCORE: {score}%")
    print(f"{'='*50}")
    if analysis["unverified_standards"]:
        print(f"  ⚠️  {len(analysis['unverified_standards'])} unverified standard(s) detected")
    else:
        print("  ✅ No unverified standards detected")
    print(f"{'='*50}")

    return results, analysis


def run_adversarial_test(cfg):
    """Run adversarial prompts to test robustness."""
    model = cfg["model"]
    temp = cfg["temperature"]
    adversarial_prompts = cfg.get("adversarial_prompts", [])

    if not adversarial_prompts:
        print("No adversarial prompts configured. Skipping.")
        return []

    print(f"\n Running Adversarial Robustness Test ({len(adversarial_prompts)} variants)")
    all_results = []

    for idx, prompt in enumerate(adversarial_prompts, 1):
        print(f"\n  Variant {idx}: {prompt[:50]}...")
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temp}
            )
            output = response["message"]["content"]
            print(f"  ✅ Response received ({len(output)} chars)")
            all_results.append({
                "variant": idx,
                "prompt": prompt,
                "output": output
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Save adversarial results
    adv_file = cfg["output_file"].replace(".json", "_adversarial.json")
    with open(adv_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n Adversarial results saved to {adv_file}")
    return all_results


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(cfg_path)

    if cfg.get("warmup", True):
        warmup_model(cfg["model"], cfg.get("warmup_prompt", "Say hello"))

    results, analysis = run_consistency_bench(cfg)

    if cfg.get("adversarial_prompts"):
        run_adversarial_test(cfg)
