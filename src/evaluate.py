import argparse
from llmtuner import MRCEvaluator, MMLUEvaluator, MMTEvaluator, ATSEvaluator

def main():
    # 创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Choose an evaluator.")
    # 添加一个名为 'eval_class' 的参数
    parser.add_argument('--eval_class', type=str, default='default', help="The eval_class to be evaluated.")
    # 解析命令行参数
    args, _ = parser.parse_known_args()

    # 根据 'eval_class' 参数的值来选择一个 Evaluator
    if args.eval_class == 'mmlu':
        evaluator_class = MMLUEvaluator
    elif args.eval_class == 'mmt':
        evaluator_class = MMTEvaluator
    elif args.eval_class == 'mrc':
        evaluator_class = MRCEvaluator
    elif args.eval_class == 'ats':
        evaluator_class = ATSEvaluator
    else:
        raise NotImplementedError(f"eval_class {args.eval_class} is not implemented.")
        # evaluator_class = Evaluator  # Default evaluator

    # 使用所选的 Evaluator
    evaluator_class().eval()

if __name__ == "__main__":
    main()
