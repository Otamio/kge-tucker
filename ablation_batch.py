import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')

if __name__ == "__main__":

    args = parser.parse_args()
    dataset = args.dataset
    gpu = args.gpu
    if 'FHC' in dataset or 'FHN' in dataset or 'QHC' in dataset or 'QHN' in dataset:
        for lv in [1, 2, 3, 4, 5]:
            subprocess.run(f"python ablation.py --dataset {dataset}_{lv} --gpu {gpu} --model distmult", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{lv} --gpu {gpu} --model complex", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{lv} --gpu {gpu} --model conve", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{lv} --gpu {gpu} --model tucker", shell=True)
    else:
        for bins in [2, 4, 8, 16, 32]:
            subprocess.run(f"python ablation.py --dataset {dataset}_{bins} --gpu {gpu} --model distmult", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{bins} --gpu {gpu} --model complex", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{bins} --gpu {gpu} --model conve", shell=True)
            subprocess.run(f"python ablation.py --dataset {dataset}_{bins} --gpu {gpu} --model tucker", shell=True)
