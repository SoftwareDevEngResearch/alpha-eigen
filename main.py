# General imports
import argparse

# Local imports
from alpha_eigen_modules.inputs import kornreich_parsons_symmetric

parser = argparse.ArgumentParser()

parser.add_argument('--problem', type=str, required=True, help='options: kornreich-parsons-symmetric')       # Problem name
parser.add_argument('--length', type=int, required=False)       # Length of slab; if not included, default is used
parser.add_argument('--cells_per_mfp', type=int, required=False)
parser.add_argument('--num_angles', type=int, required=False)
parser.add_argument('--steps', type=str, required=False)         # Number of time-steps

args = parser.parse_args()

# Run selected input
if args.problem == "kornreich-parsons-symmetric":
    if not args.length:
        args.length = 9
    if not args.cells_per_mfp:
        args.cells_per_mfp = 50
    if not args.num_angles:
        args.num_angles = 16
    if not args.steps:
        args.steps = 1
    kornreich_parsons_symmetric(length=args.length, cells_per_mfp=args.cells_per_mfp, num_angles=args.num_angles, time_steps=args.steps)


