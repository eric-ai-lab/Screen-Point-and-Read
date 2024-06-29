import os
import json
import argparse
from pathlib import Path
from mmdet.apis import DetInferencer
from time import time
from glob import glob
from copy import copy

def conf():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_folder", type=str, default="../data/failed_agent_trajectory")
    argparser.add_argument("--model_config", type=str, default="../models/tol_gui_region_detection/configs/dino/dino-4scale_r50_8xb2-90e_mobile_multi_bbox.py")
    argparser.add_argument("--checkpoint", type=str, default="../models/tol_gui_region_detection/work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth")
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--img_type", type=str, default="png;jpg;jpeg")
    return argparser.parse_args()

def list_trajectory_folder(data_root):
    data_files = glob(f"{data_root}/*/plan_execution_annotated_status.json")
    data_files = sorted(data_files, key=lambda x: int(x.split("/")[-2].split("_")[-2]))
    data_files = [Path(data_file).parent.__str__() for data_file in data_files]
    return data_files

if __name__ == "__main__":
    args = conf()
    conf_file = Path(args.model_config)
    trajectory_folder = list_trajectory_folder(args.input_folder)
    
    if not Path(args.model_config).exists() or not Path(args.checkpoint).exists():
        print("Please provide valid model config and checkpoint paths")
        exit(1)
        
    cuda_id = os.getenv("CUDA_VISIBLE_DEVICES", "-1")
    if cuda_id == "-1":
        device = "cpu"
    else:
        device = "cuda"
    inferencer = DetInferencer(model=args.model_config, weights=args.checkpoint, device=device, show_progress=True)
    start = time()
    image_types = args.img_type.split(";")
    for trajectory in trajectory_folder:
        print(f"Processing {trajectory}")
        local_status_file = f"{trajectory}/plan_execution_annotated_local_status.json"
        if Path(local_status_file).exists():
            # see: if already exists, skip
            continue
        image_files = []
        for img_type in image_types:
            image_files += glob(f"{trajectory}/*.{img_type}")
        image_files = sorted(image_files)
        output_dir = f"{trajectory}/annotated_images"
        infer_results = inferencer(image_files, out_dir=output_dir, batch_size=args.batch_size)
        assert len(image_files) == len(infer_results["predictions"])
        image_to_bbox = {}
        for image_file, prediction in zip(image_files, infer_results["predictions"]):
            image_simplified_file = Path(image_file).name
            image_to_bbox[image_simplified_file] = {
                "small_bbox": [],
                "large_bbox": [],
            }
            for label, bbox, score in zip(prediction["labels"], prediction["bboxes"], prediction["scores"]):
                label_text = "small_bbox" if label == 0 else "large_bbox"
                image_to_bbox[image_simplified_file][label_text].append({
                    "region": bbox,
                    "score": score
                })
        with open(f"{trajectory}/image_to_bbox_and_scores.json", "w") as f:
            json.dump(image_to_bbox, f, indent=1)
        with open(f"{trajectory}/plan_execution_annotated_status.json", "r") as f0:
            plan_execution_annotated_status = json.load(f0)
            for index, step in enumerate(copy(plan_execution_annotated_status["executions"])):
                plan_execution_annotated_status["executions"][index]["image"] = Path(step["image"]).name
            with open(f"{trajectory}/plan_execution_annotated_local_status.json", "w") as f1:
                json.dump(plan_execution_annotated_status, f1, indent=1)
                
    with open(f"{args.input_folder}/trajectory.json", "w") as f:
        trajectory_folder = sorted([Path(trajectory).name for trajectory in trajectory_folder], key=lambda x: int(x[:x.index("_")]))
        json.dump(trajectory_folder, f, indent=1)