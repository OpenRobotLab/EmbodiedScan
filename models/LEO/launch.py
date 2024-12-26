import argparse
from datetime import datetime

import common.launch_utils as lu


def parse_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return argparse.ArgumentTypeError('Unsupported value encountered')

    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument('--mode',
                        default='submitit',
                        type=str,
                        help='Launch mode (submitit | accelerate | python)')
    parser.add_argument('--debug',
                        default=False,
                        type=str2bool,
                        help='Debug mode (True | False)')

    # Slurm settings
    parser.add_argument('--name',
                        default='leo',
                        type=str,
                        help='Name of the job')
    parser.add_argument('--run_file',
                        default='run.py',
                        type=str,
                        help='File position of launcher file')
    parser.add_argument('--job_dir',
                        default='jobs/%j',
                        type=str,
                        help='Directory to save the job logs')
    parser.add_argument('--num_nodes',
                        default=1,
                        type=int,
                        help='Number of nodes to use in SLURM')
    parser.add_argument('--gpu_per_node',
                        default=4,
                        type=int,
                        help='Number of gpus to use in each node')
    parser.add_argument('--cpu_per_task',
                        default=32,
                        type=int,
                        help='Number of cpus to use for each gpu')
    parser.add_argument('--qos',
                        default='lv0b',
                        type=str,
                        help='Qos of the job')
    parser.add_argument('--partition',
                        default='HGX',
                        type=str,
                        help='Partition of the job')
    parser.add_argument('--account',
                        default='research',
                        type=str,
                        help='Account of the job')
    parser.add_argument('--mem_per_gpu',
                        default=100,
                        type=int,
                        help='Memory allocated for each gpu in GB')
    parser.add_argument('--time',
                        default=24,
                        type=int,
                        help='Time allocated for the job in hours')
    parser.add_argument('--port',
                        default=1234,
                        type=int,
                        help='Default port for distributed training')
    parser.add_argument('--nodelist',
                        default='',
                        type=str,
                        help='Default node id for distributed training')

    # Accelerate settings
    parser.add_argument(
        '--mixed_precision',
        default='no',
        type=str,
        help='Mixed precision training, options (no | fp16 | bf16)')

    # Additional Training settings
    parser.add_argument('--config',
                        default='configs/default.yaml',
                        type=str,
                        help='Path to the config file')
    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='Additional options to change configureation')
    return parser.parse_args()


def main():
    args = parse_args()
    getattr(lu, f'{args.mode}_launch')(args)
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]}] - Launched")


if __name__ == '__main__':
    main()
