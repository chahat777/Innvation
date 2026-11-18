import argparse
import sys
import os

from src.pipeline_generator import PipelineGenerator
from src.evaluate_models import MultimodalEvaluator

def main():
    parser = argparse.ArgumentParser(
        description="SchemAI-Bench: Automated Visual Task Factory for Frontier Multimodal AI Evaluation"
    )
    parser.add_argument("--generate", action="store_true", help="Generate synthetic schematics and verifiable tasks")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate multimodal models against generated benchmark")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: generate dataset and evaluate benchmark")
    parser.add_argument("--graphs", type=int, default=12, help="Number of synthetic schematics to generate (default: 12)")
    parser.add_argument("--model", type=str, default="Gemini-Multimodal-Frontier", help="Model name identifier for benchmark report")
    
    args = parser.parse_args()

    if not any([args.generate, args.evaluate, args.all]):
        print("💡 No mode selected. Defaulting to running full pipeline (--all)...")
        args.all = True

    if args.generate or args.all:
        print("\n=======================================================")
        print("  STEP 1: SYNTHETIC IMAGE & TASK FACTORY GENERATION")
        print("=======================================================\n")
        generator = PipelineGenerator(base_output_dir="dataset")
        generator.generate_batch(num_graphs=args.graphs, seed_offset=42)

    if args.evaluate or args.all:
        print("\n=======================================================")
        print("  STEP 2: MULTIMODAL MODEL BENCHMARK & VERIFICATION")
        print("=======================================================\n")
        evaluator = MultimodalEvaluator(dataset_dir="dataset")
        evaluator.run_benchmark(model_name=args.model)

    print("\n✨ Done! You can view outputs in the 'dataset/' folder.")

if __name__ == "__main__":
    main()
