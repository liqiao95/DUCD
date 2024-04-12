import subprocess
import argparse
parser = argparse.ArgumentParser(description='adversarial attack')
parser.add_argument('--attack', type=str, help='folder to save model and training log)')
args = parser.parse_args()
# 定义eps值范围
eps_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]

# 定义norms值范围
norms = [1, 2, -1]

# 定义folders值范围
folders = ['cifar10', 'cifar-s', 'cifar101']
i=0
# 打开文档
all_output = ""
for folder in folders:
    for eps in eps_values:
        for norm in norms:
            if norm == 1:
                command = f"python /root/project/QUCD/universal-black-box-certified-defense/attack_eval.py --device 0 --model resnet18 --path /root/project/universal-black-box-certified-defense/results{eps}/substitute/{folder}/resnet34_substitute_cifar10_L1.pt --attack {args.attack} --target False --eps {eps} --type certified --norm l1"
            elif norm == 2:
                command = f"python /root/project/universal-black-box-certified-defense/attack_eval.py --device 0 --model resnet18 --path /root/project/universal-black-box-certified-defense/results{eps}/substitute/{folder}/resnet34_substitute_cifar10_L2.pt --attack {args.attack} --target False --eps {eps} --type certified --norm l2"
            elif norm == -1:
                command = f"python /root/project/universal-black-box-certified-defense/attack_eval.py --device 0 --model resnet18 --path /root/project/universal-black-box-certified-defense/results{eps}/substitute/{folder}/resnet34_substitute_cifar10_L-1.pt --attack {args.attack} --target False --eps {eps} --type certified --norm linf"
                
            # 使用subprocess.Popen执行命令，并捕获标准输出
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
                # 将命令和相应的输出内容拼接到all_output变量中
            all_output += f"\n\n[COMMAND]: {command}\n"
            all_output += f"[OUTPUT]:\n{result.stdout}\n"
            with open("/root/project/adv_autopgd.txt", "a") as file:
                file.write(all_output)
            all_output = ""    
            i=i+1
            print(i)
