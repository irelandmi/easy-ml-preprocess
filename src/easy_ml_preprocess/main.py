import argparse
from easy_ml_preprocess import health_check, generate_yaml, preprocess

def main():
    parser = argparse.ArgumentParser(
        description="CLI tool with multiple commands"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand: generate_report
    parser_gen_report = subparsers.add_parser("generate_report", help="Generate a report")
    parser_gen_report.add_argument("--input_csv", required=True, type=str, help="File path to the CSV file")
    parser_gen_report.add_argument("--output_yaml", required=True, type=str, help="=Path to the output YAML file to be generated.")
    parser_gen_report.add_argument("--output_report", type=str, help="Output filepath and name for the report")
    parser_gen_report.set_defaults(func=lambda args: health_check.main(args))

    # Subcommand: ping_server
    parser_gen_yaml = subparsers.add_parser("generate_yaml_test", help="Generate a YAML file for preprocessing")
    parser_gen_yaml.add_argument(
        "--input_csv",
        required=True,
        help="Path to the CSV file containing raw features."
    )
    parser_gen_yaml.add_argument(
        "--output_yaml",
        required=True,
        help="Path to the output YAML file to be generated."
    )
    parser_gen_yaml.add_argument(
        "--base_yaml",
        required=False,
        help="Optional path to a base YAML file. If provided, this script will load "
             "that YAML first, then override or add settings based on the CSV."
    )
    parser_gen_yaml.add_argument(
        "--output_type",
        choices=["pandas", "torch"],
        default="pandas",
        help="Desired output data format in the generated config (default: pandas)."
    )
    parser_gen_yaml.set_defaults(func=lambda args: generate_yaml.main(args))

    # Subcommand: generate_report
    parser_preprocess = subparsers.add_parser("preprocess", help="Preprocess csv file")
    parser_preprocess.add_argument("--input_csv", required=True, type=str, help="File path to the CSV file")
    parser_preprocess.add_argument("--yaml_config", required=True, help="Path to the CSV file containing raw features.")
    parser_preprocess.add_argument("--output_csv", type=str, help="Output filepath and name for the data")
    parser_preprocess.set_defaults(func=lambda args: preprocess.main(args))

    # Parse the arguments
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)  # Calls the appropriate function
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
