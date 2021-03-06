import argparse
import subprocess


parser = argparse.ArgumentParser(
    description="Running Machine"
)
parser.add_argument('--model', default='rotate', help='Please provide a model to run')
parser.add_argument('--dataset', default='fb15k237', help='Please provide a dataset path')
parser.add_argument('--gpu', default='0', help='Please provide a gpu to assign the task')
parser.add_argument('--options', default='', help='Please provide additional instructions if necessary')

if __name__ == "__main__":

    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    gpu = args.gpu
    options = args.options

    if model in {"tucker", "tucker_literal", "tucker_kbln"}:
        if "fb15k237" in dataset:
            command = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {dataset} --model {model} " \
                       "--num_iterations 500 --batch_size 128 --lr 0.0005 --dr 1.0 --edim 200 --rdim 200 " \
                       "--input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1"
        else:
            command = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {dataset} --model {model} " \
                       "--num_iterations 500 --batch_size 128 --lr 0.003 --dr 0.99 --edim 200 --rdim 200 " \
                       "--input_dropout 0.2 --hidden_dropout1 0.2 --hidden_dropout2 0.3 --label_smoothing 0.0"
    elif model in {"distmult", "distmult_literal", "distmult_kbln"}:
        command = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {dataset} --model {model} " \
                  "--num_iterations 200 --eval_step 5 --batch_size 128 --lr 0.003 --dr 0.995 " \
                  "--edim 200 --rdim 200 --input_dropout 0.2 --label_smoothing 0.1"
    elif model in {"complex", "complex_literal", "complex_kbln"}:
        command = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {dataset} --model {model} " \
                  "--num_iterations 200 --eval_step 5 --batch_size 128 --lr 0.003 --dr 0.995 " \
                  "--edim 400 --rdim 400 --input_dropout 0.2 --label_smoothing 0.1"
    elif model in {"conve", "conve_literal", "conve_kbln"}:
        command = f"CUDA_VISIBLE_DEVICES={gpu} python main.py --dataset {dataset} --model {model} " \
                  "--num_iterations 1000 --eval_step 10 --batch_size 128 --lr 0.003 --dr 0.995 " \
                  "--edim 200 --rdim 200 --input_dropout 0.2 --hidden_dropout1 0.3 --feature_map_dropout 0.2 " \
                  "--label_smoothing 0.1 --use_bias"
    else:
        print(model, "is not supported")
        exit()

    if options == "dry-run":
        print(command)
    else:
        subprocess.run(command, shell=True)
