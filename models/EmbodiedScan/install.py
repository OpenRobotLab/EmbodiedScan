import argparse
import re
import subprocess
import sys


def run_subprocess(command):
    try:
        process = subprocess.Popen(command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)

        # Read output and error in real-time
        for line in process.stdout:
            print(line.strip())
        for line in process.stderr:
            print(line.strip())

        # Wait for the subprocess to finish
        process.wait()

        # Get the return code
        return_code = process.returncode

        if return_code != 0:
            print(f'Command failed with return code {return_code}')

    except subprocess.CalledProcessError as e:
        print(f'Command failed with return code {e.returncode}')
        print('Error output:')
        print(e.output.decode())


def pytorch3d_links():
    try:
        import torch
    except ImportError as e:
        print('Pytorch is not installed.')
        raise e
    cuda_version = torch.version.cuda
    if cuda_version is None:
        print('Pytorch is cpu only.')
        raise NotImplementedError

    pyt_version_str = torch.__version__.split('+')[0].replace('.', '')
    cuda_version_str = torch.version.cuda.replace('.', '')
    version_str = ''.join([
        f'py3{sys.version_info.minor}_cu', cuda_version_str,
        f'_pyt{pyt_version_str}'
    ])
    pytorch3d_links = f'https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html'  # noqa: E501
    return pytorch3d_links


def mmcv_links():
    try:
        import torch
    except ImportError as e:
        print('Pytorch is not installed.')
        raise e
    cuda_version = torch.version.cuda
    if cuda_version is None:
        print('Pytorch is cpu only.')
        raise NotImplementedError

    cuda_version_str = torch.version.cuda.replace('.', '')
    pyt_version = torch.__version__.split('+')[0].split('.')
    pyt_version_mmcv = pyt_version[0] + '.' + pyt_version[1]
    mmcv_links = f'https://download.openmmlab.com/mmcv/dist/cu{cuda_version_str}/torch{pyt_version_mmcv}/index.html'  # noqa: E501
    return mmcv_links


def install_package(line):
    pat = '(' + '|'.join(['>=', '==', '>', '<', '<=', '@']) + ')'
    parts = re.split(pat, line, maxsplit=1)
    package_name = parts[0].strip()
    print('installing', package_name)
    if package_name == 'pytorch3d':
        links = pytorch3d_links()
        run_subprocess(
            [sys.executable, '-m', 'pip', 'install', 'pytorch3d', '-f', links])
    elif package_name == 'mmcv':
        links = mmcv_links()
        print(links)
        run_subprocess(
            [sys.executable, '-m', 'pip', 'install', line, '-f', links])
    elif package_name == 'MinkowskiEngine':
        run_subprocess([sys.executable, '-m', 'pip', 'install', 'ninja'])
        run_subprocess([
            sys.executable, '-m', 'pip', 'install', '-U',
            'git+https://github.com/NVIDIA/MinkowskiEngine', '--no-deps'
        ])  # noqa: E501
    else:
        run_subprocess([sys.executable, '-m', 'pip', 'install', line])


def install_requires(fname):
    with open(fname, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                install_package(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Install Embodiedscan from pre-built package.')
    parser.add_argument('mode', default=None)
    args = parser.parse_args()

    install_requires('requirements/base.txt')
    if args.mode == 'visual' or args.mode == 'all':
        install_requires('requirements/visual.txt')

    if args.mode == 'run' or args.mode == 'all':
        install_requires('requirements/run.txt')

    run_subprocess([sys.executable, '-m', 'pip', 'install', '-e', '.'])
