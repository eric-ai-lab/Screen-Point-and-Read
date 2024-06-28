from glob import glob
import json
    
if __name__ == '__main__':
    data_root = "../mobile_data/failed_agent_trajectory"
    data_files = glob(f"{data_root}/*/plan_execution_annotated_status.json")
    data_files = sorted(data_files, key=lambda x: int(x.split("/")[-2].split("_")[-2]))
    correct_step_cnt = 0
    click_event = 0
    step_total = 0
    total_considered_event = 0
    event_to_number = {}
    for data_file in data_files:
        with open(data_file, 'r') as f:
            data = json.load(f)
            for index, step in enumerate(data["executions"]):
                step_total += 1
                if 'action' in step and (step['action'] == "click" or step['action'] == "input"):
                    total_considered_event += 1
                if "action_evaluation" in step:
                    correct_step_cnt += 1
                    if len(step["action_evaluation"]["correct_action"]) > 0:
                        print(f"{data_file}, step index: {index + 1}, Original action: {step['action']}, correct to: {step['action_evaluation']['correct_action']}, ")
                    if step['action'] == "click":
                        click_event += 1
                    event_to_number[step['action']] = event_to_number.setdefault(step['action'], 0) + 1
    print(f"Total correct steps: {correct_step_cnt}, event_to_number: {event_to_number}")
    print(f"Total steps: {step_total}, considered total event: {total_considered_event}")