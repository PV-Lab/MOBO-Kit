"""
MOBO-Kit Command Line Interface

Provides command-line access to MOBO-Kit functionality.
"""

import argparse
import sys
import os
from typing import Optional

from .main import run_mobo_experiment, generate_initial_experiments


def main():
    """
    Command line interface for MOBO-Kit.
    
    Usage examples:
        mobo-kit run --csv data/my_data.csv
        mobo-kit generate --config configs/my_config.yaml --n-samples 20 --out initial_experiments.csv
    """
    parser = argparse.ArgumentParser(
        description="MOBO-Kit: Multi-objective Bayesian Optimization Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate initial experiments
  mobo-kit generate --config configs/my_config.yaml --n-samples 20 --out initial_experiments.csv

  # Run MOBO with existing data
  mobo-kit run --csv data/processed/configCSV_example.csv

  # Run with custom configuration and output directory
  mobo-kit run --csv data/my_data.csv --config configs/my_config.yaml --out results/my_experiment

  # Run with specific device and seed
  mobo-kit run --csv data/my_data.csv --device cpu --seed 123 --verbose

  # Run with custom batch size for acquisition
  mobo-kit run --csv data/my_data.csv --batch-size 10 --num-restarts 50
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate subcommand
    generate_parser = subparsers.add_parser('generate', help='Generate initial experiments using LHS')
    generate_parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    generate_parser.add_argument('--n-samples', type=int, required=True, help='Number of initial experiments to generate')
    generate_parser.add_argument('--out', required=True, help='Output CSV file path')
    generate_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    generate_parser.add_argument('--max-corr', type=float, help='Maximum absolute correlation between variables')
    generate_parser.add_argument('--max-attempts', type=int, default=100, help='Maximum attempts for LHS generation (default: 100)')
    generate_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    # Run subcommand
    run_parser = subparsers.add_parser('run', help='Run MOBO optimization with existing data')
    run_parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file with experimental data"
    )
    
    # Optional arguments for run command
    run_parser.add_argument(
        "--config",
        help="Path to YAML configuration file (optional, will auto-generate from CSV if not provided)"
    )
    
    run_parser.add_argument(
        "--out",
        default="results/experiment",
        help="Output directory for results (default: results/experiment)"
    )
    
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    run_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for computation (default: auto)"
    )
    
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Advanced options (for future expansion)
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for acquisition (default: 5)"
    )
    
    run_parser.add_argument(
        "--num-restarts",
        type=int,
        default=20,
        help="Number of restarts for acquisition optimization (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == 'generate':
        # Validate inputs for generate command
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        try:
            # Generate initial experiments
            results = generate_initial_experiments(
                config_path=args.config,
                n_samples=args.n_samples,
                save_path=args.out,
                seed=args.seed,
                verbose=args.verbose,
                max_abs_corr=args.max_corr,
                max_attempts=args.max_attempts
            )
            
            if args.verbose:
                print("\n" + "=" * 60)
                print("GENERATION SUMMARY:")
                print(f"  Config: {args.config}")
                print(f"  Samples: {results['n_samples']}")
                print(f"  Variables: {', '.join(results['variables'])}")
                print(f"  Constraints: {'Applied' if results['constraints_applied'] else 'None'}")
                print(f"  Seed: {results['seed']}")
                print(f"  Output: {results['save_path']}")
                print("=" * 60)
            else:
                print(f"Success! Generated {results['n_samples']} experiments in {results['save_path']}")
                
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
            
    elif args.command == 'run':
        # Validate inputs for run command
        if not os.path.exists(args.csv):
            print(f"Error: CSV file not found: {args.csv}")
            sys.exit(1)
        
        if args.config and not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(args.out, exist_ok=True)
        
        try:
            # Run the experiment
            results = run_mobo_experiment(
                csv_path=args.csv,
                save_dir=args.out,
                config_path=args.config,
                seed=args.seed,
                device=args.device,
                verbose=args.verbose,
                batch_size=args.batch_size
            )
            
            # Print summary
            if args.verbose:
                print("\n" + "=" * 60)
                print("EXPERIMENT SUMMARY:")
                print(f"  CSV file: {args.csv}")
                print(f"  Config: {args.config or 'auto-generated'}")
                print(f"  Samples: {results['n_samples']}")
                print(f"  Inputs: {results['n_inputs']}")
                print(f"  Objectives: {results['n_objectives']}")
                print(f"  Objectives: {', '.join(results['objective_names'])}")
                print(f"  Device: {results['device']}")
                print(f"  Seed: {results['seed']}")
                print(f"  Results: {results['save_dir']}")
                if results.get('candidates'):
                    print(f"  Next batch: {args.batch_size} candidates proposed")
                print("=" * 60)
            else:
                success_msg = f"Success! Results saved to {results['save_dir']}"
                if results.get('candidates'):
                    success_msg += f" ({args.batch_size} candidates proposed)"
                print(success_msg)
                
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
