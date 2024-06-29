# Verification of Mobile Navigation Agent Actions

This guide shows you how to finish mobile agent verification with the help of Lot Agent. You need to switch to [src/mobile_agent](src/mobile) and install extra components before starting steps:

```bash
cd src/mobile_agent
pip install numpy cv openai python-dotenv jupyter scikit-learn
```

## 1. Data synchronization

 Initialize lfs storage and pull from remote:

```bash
git lfs install
git lfs pull
```

The picked-up mobile trajectories will be sync at [data/failed_agent_trajectory](../data/failed_agent_trajectory).

## 2. Trigger LoT agent inference

Next, start LoT agent inference for all mobile screens. You are supposed to store [LoT agent weights](https://drive.google.com/file/d/1IN3EfDKyXwu5WegqyFOWfXH6ttJ3zNdx/view?usp=drive_link) at the folder [src/models/tol_gui_region_detection/work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth](../models/tol_gui_region_detection/work_dirs/dino-4scale_r50_8xb2-90e_mobile_multi_bbox/epoch_90.pth)

```bash
python inference_mobile_segment.py
```

Verify that for each trajectory, a **plan_execution_annotated_local_status.json** file and **image_to_bbox_and_scores.json** file that contains local and global region candidates will be generated.

## 3. Analyze trajectory

To understand different actions involving with target trajectories, using

```bash
python failure_analysis.py
```

Next, use GPT4-o to do action verification. By default, we use GPT4-o on AZURE cloud and also allows you to switch to openAI endpoints.

+ Create .env file under current folder, which contains AZURE or openai key likes:

```bash
# OpenAI key
OPENAI_API_KEY=<Your openai key>
# or AZURE setting
AZURE_OPENAI_API_KEY=<Your AZURE key>
AZURE_OPENAI_BASE=<Your openai base url on AZURE server>
AZURE_OPENAI_DEPLOYMENT_NAME=<Your deployment name on AZURE server>
AZURE_OPENAI_API_VERSION=<Your openai version on AZURE server>
```

Fill parameters based on your server choice.

+ Trigger inference

If you use Openai api on AZURE, using

```bash
python analyze_agent_trajectory.py --candidate_actions click input --use_official_openai_url True
```

If you use Openai official API, using

```bash
python analyze_agent_trajectory.py --candidate_actions click input --use_official_openai_url True
```

A result json file will be generated with the name `ssr_statistic_agent_<The triggering datetime>_click_input_prompt.json`. Check your result is similar to one of our experimental output json files [ssr_statistic_agent_2024-06-05 17:17:53_click_input_prompt.json](src/mobile_agent/ssr_statistic_agent_2024-06-05 17:17:53_click_input_prompt.json).

+ Resume inference in case of unexpected failure

Considering that calling openai API may be broken by accident (API request quote, unstable server status), we also provide **"exception-resume"** mechanism. You can use your unfinished result json file to resume your trajectory analysis, using

```bash
python analyze_agent_trajectory.py --candidate_actions click input --use_official_openai_url True  --resume_previous_running_file "<your previous result json file name>"
```

## 4. Generate statistics based on multiple executions

In our experiment, we carried out about 10 times of trajectory analysis to understand if our verification keep consistent. You can start your jupyter server locally and run report_ssr_agent_for_correcting_mobile_agent.ipynb. In the section "Final statistics: LoT Agent + Simplified Action Validation Agent", you can find the final statistics as follow:

```bash
# section 1: classification report
               precision   recall  f1-score   support
negative       0.38:0.03  0.81:0.03 0.52:0.03 47:4
positive       0.92:0.02  0.62:0.01 0.74:0.01 162:4
accuracy                           0.66:0.01 209
macro avg       0.65:0.01 0.72:0.02 0.63:0.02  209
weighted avg       0.80:0.02 0.66:0.01 0.69:0.01 209

# section 2: application statistic
tp 100.20:2.04
fp 8.80:2.04
tn 38.10:2.91
fn 61.90:2.91
tpr 0.92:0.02
fpr 0.08:0.02
tnr 0.38:0.03
fnr 0.62:0.03
accuracy_v 0.66:0.01
precision_v 0.92:0.02
ecall_v 0.62:0.01
fp_failed_detect_repetition 0.00:0.00
repetition_detect_in_click 8.80:0.75
repetition_detect_in_click rate 0.44:0.04
total 209.00:0.00
```

All statistics have been formatted in `<mean>:<standard deviation>` format. For instance,  "tp 100.20:2.04" means true positive have a mean value of 100 and standard deviation 2.04 throughout all of our experiments.

For the output result, the first section of precision, recall, f1-score, support follows the same pattern of classification report in sklearn, refer to [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html). The second section contains application statistics: tpr, fpr, tnr, fnr stands for true positive rate, false positive rate, true negative rate, false negative rate; repetition_detect_in_click, repetition_detect_in_click rate describes how many repeated actions ("execution loop" issue mentioned in the paper) has been detected and the ratio of the total actions with such issue.