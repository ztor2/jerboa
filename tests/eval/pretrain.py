"""Pre-training Stage Evaluation Suite for JerboaLM (`tests/eval/pretrain.py`).

Evaluates foundation base model quality prior to post-training (SFT / RL):
1. Domain-specific Held-out Perplexity (PPL) & Bits-Per-Byte (BPB) across English, Korean, and Python.
2. Zero-shot Log-Likelihood Multiple-Choice Accuracy (PMI prior-calibrated NLL ranking).
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import JerboaForCausalLM, get_default_tokenizer


HELD_OUT_PASSAGES: Dict[str, List[str]] = {
    "English (FineWeb-Edu)": [
        (
            "Photosynthesis is the biological process by which green plants, algae, and certain bacteria "
            "convert light energy from the sun into chemical energy stored in glucose molecules. "
            "During this process, carbon dioxide and water are absorbed and transformed into oxygen and organic compounds."
        ),
        (
            "The central limit theorem states that when independent random variables are summed up, "
            "their properly normalized sum tends toward a normal distribution even if the original variables "
            "themselves are not normally distributed. This makes it a cornerstone of statistical inference."
        ),
        (
            "In computer architecture, a CPU cache is a smaller, faster memory located closer to a processor core, "
            "which stores copies of the data from frequently used main memory locations to reduce average memory access time."
        ),
    ],
    "Korean (AI-Hub / Knowledge)": [
        (
            "질문: 2012년 런던 패럴림픽 양궁 여자 개인전에서 금메달을 획득한 선수의 기록적 의미는 무엇인가요?\n"
            "답변: 해당 선수는 런던 패럴림픽 양궁 여자 개인 컴파운드 및 리커브 부문에서 뛰어난 집중력을 발휘하여 금메달을 획득하였으며, "
            "이는 한국 장애인 양궁의 국제적 위상을 높이는 중요한 계기가 되었습니다."
        ),
        (
            "질문: 머신러닝에서 과적합(Overfitting)이 발생하는 원인과 이를 방지하기 위한 정규화 기법에는 어떤 것들이 있나요?\n"
            "답변: 과적합은 모델이 훈련 데이터의 노이즈까지 과도하게 학습하여 새로운 테스트 데이터에서 일반화 성능이 떨어지는 현상입니다. "
            "이를 방지하기 위해 가중치 감쇠(Weight Decay), 드롭아웃(Dropout), 조기 종료(Early Stopping) 등의 기법을 사용합니다."
        ),
        (
            "질문: 조선 시대의 금속 활자 인쇄술이 세계 인쇄사에서 가지는 가치는 무엇인가요?\n"
            "답변: 조선 시대에는 갑인자 등 정교한 금속 활자를 주조하여 서적을 편찬하였으며, 이는 지식의 보급과 학문 발전에 크게 기여하였습니다."
        ),
    ],
    "Python (CodeParrot)": [
        (
            "import os\nimport json\nfrom typing import Dict, Any\n\n"
            "def load_json_config(file_path: str) -> Dict[str, Any]:\n"
            "    if not os.path.exists(file_path):\n"
            "        raise FileNotFoundError(f'Config file not found: {file_path}')\n"
            "    with open(file_path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)\n"
        ),
        (
            "def binary_search(arr: list[int], target: int) -> int:\n"
            "    low, high = 0, len(arr) - 1\n"
            "    while low <= high:\n"
            "        mid = (low + high) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            low = mid + 1\n"
            "        else:\n"
            "            high = mid - 1\n"
            "    return -1\n"
        ),
        (
            "class MovingAverage:\n"
            "    def __init__(self, window_size: int):\n"
            "        self.window_size = window_size\n"
            "        self.values = []\n\n"
            "    def update(self, val: float) -> float:\n"
            "        self.values.append(val)\n"
            "        if len(self.values) > self.window_size:\n"
            "            self.values.pop(0)\n"
            "        return sum(self.values) / len(self.values)\n"
        ),
    ],
}


MULTIPLE_CHOICE_TASKS: Dict[str, List[Dict]] = {
    "English Commonsense & Science (ARC / PIQA / HellaSwag)": [
        {
            "context": "Photosynthesis in green plants requires sunlight, water, and carbon dioxide to produce glucose and release",
            "choices": [
                " oxygen into the atmosphere.",
                " nitrogen into the soil.",
                " iron into the roots.",
                " helium into space.",
            ],
            "answer": 0,
        },
        {
            "context": "When water is cooled below zero degrees Celsius at standard atmospheric pressure, it transitions from a liquid into",
            "choices": [
                " a solid crystal structure known as ice.",
                " a superheated plasma state.",
                " pure hydrogen gas.",
                " liquid mercury.",
            ],
            "answer": 0,
        },
        {
            "context": "In computer science, a stack data structure operates on the principle of",
            "choices": [
                " Last-In, First-Out (LIFO), where the most recently added element is removed first.",
                " random access sorting using prime number multiplication.",
                " converting all binary integers into floating-point audio signals.",
                " deleting the operating system kernel on every read operation.",
            ],
            "answer": 0,
        },
        {
            "context": "Gravity is the fundamental force of attraction that causes objects with mass to",
            "choices": [
                " accelerate toward one another, keeping planets in orbit around stars.",
                " repel each other when placed in a vacuum chamber.",
                " lose all their electrons at room temperature.",
                " travel faster than the speed of light in water.",
            ],
            "answer": 0,
        },
        {
            "context": "DNA carries the genetic instructions used in the growth, development, functioning, and reproduction of",
            "choices": [
                " all known living organisms and many viruses.",
                " igneous rocks formed by volcanic eruptions.",
                " silicon semiconductors in solar panels.",
                " atmospheric greenhouse gases like methane.",
            ],
            "answer": 0,
        },
    ],
    "Korean Knowledge & Reasoning (KoBEST / AI-Hub QA)": [
        {
            "context": "질문: 기온이 영하로 내려가면 액체 상태의 물은 어떻게 변하나요?\n답변:",
            "choices": [
                " 물이 얼어서 고체 상태인 얼음이 됩니다.",
                " 물이 끓어올라 수증기로 즉시 폭발합니다.",
                " 물이 철이나 구리 같은 금속으로 바뀝니다.",
                " 물의 무게가 갑자기 100배 이상 늘어납니다.",
            ],
            "answer": 0,
        },
        {
            "context": "질문: 딥러닝 모델 학습 시 학습률(Learning Rate)이 지나치게 크면 어떤 문제가 발생하나요?\n답변:",
            "choices": [
                " 손실 함수(Loss)가 최솟값으로 수렴하지 못하고 발산할 수 있습니다.",
                " 모델의 파라미터 개수가 자동으로 10배로 줄어듭니다.",
                " 그래픽카드(GPU)의 메모리 용량이 두 배로 늘어납니다.",
                " 이미지 파일이 오디오 파일로 자동 변환됩니다.",
            ],
            "answer": 0,
        },
        {
            "context": "질문: 식물이 빛 에너지를 이용해 이산화탄소와 물로부터 양분과 산소를 만들어내는 과정은 무엇인가요?\n답변:",
            "choices": [
                " 광합성(Photosynthesis)입니다.",
                " 핵분열(Nuclear Fission)입니다.",
                " 중화 반응(Neutralization)입니다.",
                " 전자기 유도(Electromagnetic Induction)입니다.",
            ],
            "answer": 0,
        },
        {
            "context": "질문: 컴퓨터에서 데이터를 영구적으로 저장하지 않고 실행 중인 프로그램의 데이터를 일시적으로 저장하는 주기억장치는 무엇인가요?\n답변:",
            "choices": [
                " 램(RAM, Random Access Memory)입니다.",
                " 마우스(Mouse)와 키보드입니다.",
                " 냉각팬(Cooling Fan)입니다.",
                " 전원 공급 장치(Power Supply)입니다.",
            ],
            "answer": 0,
        },
        {
            "context": "질문: 지구가 태양의 둘레를 일 년에 한 바퀴씩 도는 운동을 무엇이라고 하나요?\n답변:",
            "choices": [
                " 지구의 공전(Revolution)이라고 합니다.",
                " 지진파의 굴절이라고 합니다.",
                " 대류권의 기상 현상이라고 합니다.",
                " 바닷물의 밀물과 썰물이라고 합니다.",
            ],
            "answer": 0,
        },
    ],
    "Python Code Completion & Semantics": [
        {
            "context": "def factorial(n: int) -> int:\n    if n <= 1:\n        return 1\n    return",
            "choices": [
                " n * factorial(n - 1)",
                " 'hello_world'.upper()",
                " [x for x in range(1000)]",
                " None + 12345",
            ],
            "answer": 0,
        },
        {
            "context": "import json\n\ndef parse_json(text: str) -> dict:\n    return",
            "choices": [
                " json.loads(text)",
                " json.dump_to_file_handle()",
                " text.split_into_integers()",
                " int(text) ** 2",
            ],
            "answer": 0,
        },
        {
            "context": "def is_even(number: int) -> bool:\n    return",
            "choices": [
                " number % 2 == 0",
                " number / 0 == 1",
                " len(number) > 0",
                " isinstance(number, str)",
            ],
            "answer": 0,
        },
        {
            "context": "with open('data.txt', 'r', encoding='utf-8') as f:\n    lines =",
            "choices": [
                " f.readlines()",
                " f.mkdir_p()",
                " f.compile_cuda_kernel()",
                " f + 100",
            ],
            "answer": 0,
        },
        {
            "context": "numbers = [3, 1, 4, 1, 5, 9]\nsorted_numbers =",
            "choices": [
                " sorted(numbers)",
                " numbers.open_file()",
                " dict(numbers)",
                " numbers / 2",
            ],
            "answer": 0,
        },
    ],
}


@torch.no_grad()
def evaluate_perplexity_and_bpb(
    model: JerboaForCausalLM,
    tokenizer,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compute held-out Cross-Entropy Loss, Perplexity (PPL), and Bits-Per-Byte (BPB)."""
    results = {}
    for domain, texts in HELD_OUT_PASSAGES.items():
        total_nll_nats = 0.0
        total_tokens = 0
        total_bytes = 0

        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids, use_cache=False)

            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            n_pred = shift_labels.numel()

            loss_nats = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                reduction="sum",
            ).item()

            total_nll_nats += loss_nats
            total_tokens += n_pred
            total_bytes += len(text.encode("utf-8"))

        avg_ce_loss = total_nll_nats / max(total_tokens, 1)
        ppl = math.exp(min(avg_ce_loss, 20.0))
        bpb = (total_nll_nats / math.log(2.0)) / max(total_bytes, 1)

        results[domain] = {
            "ce_loss": avg_ce_loss,
            "ppl": ppl,
            "bpb": bpb,
            "tokens": total_tokens,
            "bytes": total_bytes,
        }
    return results


@torch.no_grad()
def score_continuation_loss(
    model: JerboaForCausalLM,
    tokenizer,
    context: str,
    continuation: str,
    device: torch.device,
    calibration_prefix: str = "",
) -> float:
    """Compute length-normalized NLL, optionally calibrated by unconditional prior (`lm-eval` PMI norm)."""
    ctx_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)

    cont_start = len(ctx_ids)
    if len(full_ids) <= cont_start:
        cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
        full_ids = ctx_ids + cont_ids
        cont_start = len(ctx_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, use_cache=False)

    shift_logits = outputs.logits[0, cont_start - 1 : -1, :]
    target_ids = input_ids[0, cont_start:]
    cond_loss = F.cross_entropy(shift_logits, target_ids, reduction="mean").item()

    if calibration_prefix:
        cal_ctx_ids = tokenizer.encode(calibration_prefix, add_special_tokens=False)
        cal_full_ids = tokenizer.encode(calibration_prefix + continuation, add_special_tokens=False)
        cal_start = len(cal_ctx_ids)
        if len(cal_full_ids) <= cal_start:
            cont_ids = tokenizer.encode(continuation, add_special_tokens=False)
            cal_full_ids = cal_ctx_ids + cont_ids
            cal_start = len(cal_ctx_ids)
        cal_input = torch.tensor([cal_full_ids], dtype=torch.long, device=device)
        cal_out = model(input_ids=cal_input, use_cache=False)
        cal_logits = cal_out.logits[0, cal_start - 1 : -1, :]
        cal_targets = cal_input[0, cal_start:]
        prior_loss = F.cross_entropy(cal_logits, cal_targets, reduction="mean").item()
        return cond_loss - 0.5 * prior_loss

    return cond_loss


@torch.no_grad()
def evaluate_multiple_choice(
    model: JerboaForCausalLM,
    tokenizer,
    device: torch.device,
    verbose: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Evaluate zero-shot multiple-choice accuracy via log-likelihood ranking."""
    category_scores = {}
    total_correct = 0
    total_questions = 0

    for category, items in MULTIPLE_CHOICE_TASKS.items():
        correct = 0
        cal_prefix = "답변:" if "Korean" in category else ("Answer:" if "English" in category else "")
        if verbose:
            print(f"\n  [{category}]")
        for idx, item in enumerate(items, 1):
            ctx = item["context"]
            choices = item["choices"]
            gold = item["answer"]

            losses = [
                score_continuation_loss(model, tokenizer, ctx, c, device, calibration_prefix=cal_prefix)
                for c in choices
            ]
            pred = int(min(range(len(losses)), key=lambda i: losses[i]))
            is_hit = (pred == gold)
            if is_hit:
                correct += 1
            total_correct += int(is_hit)
            total_questions += 1

            if verbose:
                mark = "PASS" if is_hit else "FAIL"
                best_text = choices[pred].strip()
                print(
                    f"    Q{idx} [{mark}] (gold_score={losses[gold]:.3f}, best_other={min(l for i, l in enumerate(losses) if i != gold):.3f}) "
                    f"-> {best_text[:55]}"
                )

        acc = (correct / len(items)) * 100.0
        category_scores[category] = {"correct": correct, "total": len(items), "accuracy": acc}

    overall_acc = (total_correct / max(total_questions, 1)) * 100.0
    return category_scores, overall_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate Jerboa Pre-trained Base Model")
    if os.path.exists("checkpoints/pretrain/model"):
        default_ckpt = "checkpoints/pretrain/model"
    elif os.path.exists("checkpoints/pretrain/hub_latest"):
        default_ckpt = "checkpoints/pretrain/hub_latest"
    else:
        default_ckpt = "ztor2/jerboa-base"
    parser.add_argument("--model", type=str, default=default_ckpt, help="Local checkpoint path or HF Hub repo ID")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 78)
    print(f"  JERBOA PRE-TRAIN EVALUATION SUITE ({args.model} on {device})")
    print("=" * 78)

    tokenizer = get_default_tokenizer()
    model = JerboaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded JerboaForCausalLM ({num_params:.1f}M parameters)")

    # 1. Held-out Perplexity & Bits-Per-Byte
    print("\n[1/2] Held-Out Perplexity (PPL) & Bits-Per-Byte (BPB)")
    print("-" * 78)
    print(f"{'Domain':<30} | {'CE Loss':>9} | {'Perplexity':>11} | {'Bits/Byte (BPB)':>15}")
    print("-" * 78)
    ppl_results = evaluate_perplexity_and_bpb(model, tokenizer, device)
    for domain, m in ppl_results.items():
        print(f"{domain:<30} | {m['ce_loss']:>9.4f} | {m['ppl']:>11.2f} | {m['bpb']:>15.4f}")
    print("-" * 78)

    # 2. Zero-Shot Log-Likelihood Multiple Choice
    print("\n[2/2] Zero-Shot Log-Likelihood Multiple-Choice Evaluation")
    print("-" * 78)
    mc_scores, overall_acc = evaluate_multiple_choice(model, tokenizer, device, verbose=True)
    print("-" * 78)
    for cat, s in mc_scores.items():
        print(f"  {cat:<52}: {s['correct']}/{s['total']} ({s['accuracy']:.1f}%)")
    print("-" * 78)
    print(f"  OVERALL ZERO-SHOT ACCURACY (Random Chance = 25.0%)  : {overall_acc:.1f}%")
    print("=" * 78)


if __name__ == "__main__":
    main()
