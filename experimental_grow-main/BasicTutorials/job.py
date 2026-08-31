import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--A",
    type=int
)
parser.add_argument(
    "--B",
    type=int
)
args, _ = parser.parse_known_args()

print(
    f"I'm the job with A={args.A} and B={args.B}"
)