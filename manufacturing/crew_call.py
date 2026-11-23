"""Read one five-minute inspection check for an adverse shift before dispatch."""

import argparse
import json

from staggered_support import CHECK, DURATION, HISTORY, RESPONSE, ROOT, SHIFT_SECONDS, trigger
from support_budget import MINUTES
from peak_crew_budget import OPTIONS
from repair_space_budget import SLOTS

BUDGET_OPTIONS = set(OPTIONS) | {(2, m) for m in MINUTES}


def decision(minute, waiting, seen, recent_repairs, policy, duration=DURATION // 60, people=2, repair_slots=12):
    if (people, duration) not in BUDGET_OPTIONS or (people > 2 and policy[2] != 0):
        raise ValueError("请输入实验覆盖的支援人数与时长组合")
    if repair_slots not in SLOTS or (repair_slots != 12 and (people, duration) != (2, 120)):
        raise ValueError("增加等待位的试验采用两人各 120 分钟")
    latest_start = SHIFT_SECONDS // 60 - duration
    last_check = latest_start - RESPONSE // 60
    if not 0 <= minute <= last_check or minute % (CHECK // 60):
        raise ValueError(f"请输入开班后 0 至 {last_check} 之间的五分钟检查点")
    if not 0 <= waiting <= repair_slots + 1 or not 0 <= recent_repairs <= min(seen, HISTORY):
        raise ValueError(f"等待件数应为 0 至 {repair_slots + 1}，最近返工数不得超过实际首检件数或 40 件")
    threshold, minimum_repairs, lag = policy
    if trigger(waiting, seen, recent_repairs, threshold, minimum_repairs):
        reason = "达到支援条件"
    elif minute == last_check:
        reason = "到达最晚通知时间"
    else:
        return f"暂缓调人，第 {minute + CHECK // 60} 分钟复查。"
    first = minute + RESPONSE // 60
    second = min(first + lag, latest_start)
    if first == second:
        crew = "两人" if people == 2 else f"{people} 人"
        return f"{reason}，现在通知{crew}，第 {first} 分钟同时到岗，第 {first + duration} 分钟结束。"
    return (f"{reason}，第一人第 {first} 至 {first + duration} 分钟在岗，"
            f"第二人第 {second} 至 {second + duration} 分钟在岗。")


def main():
    parser = argparse.ArgumentParser(description="异常班尚未调人时，输入当前检查记录，计算支援时间。")
    parser.add_argument("--minute", type=int, required=True, help="距开班的净工作分钟")
    parser.add_argument("--waiting", type=int, required=True, help="已判返工且尚未开修的件数，含旧批次")
    parser.add_argument("--seen", type=int, required=True, help="本批次累计完成首检的件数")
    parser.add_argument("--recent-repairs", type=int, required=True, help="本批最近至多40件首检中判返工的件数")
    parser.add_argument("--repair-slots", type=int, choices=SLOTS, default=12, help="返工前可用等待位")
    parser.add_argument("--people", type=int, default=2, help="同时支援的人数")
    parser.add_argument("--minutes-per-person", type=int, choices=sorted({m for _, m in BUDGET_OPTIONS}), default=DURATION // 60,
                        help="每人每个异常班预留的支援分钟数")
    args = parser.parse_args()
    policy = json.loads((ROOT / "results/staggered_support.json").read_text())["selected"]
    try:
        print(decision(args.minute, args.waiting, args.seen, args.recent_repairs, policy, args.minutes_per_person, args.people, args.repair_slots))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
