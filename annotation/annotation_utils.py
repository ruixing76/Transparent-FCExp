# Description: This file contains utility functions for analyzing the results of the annotation tasks.
import os
import pandas as pd
from .mturk_utils import *
from typing import List
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ANSWER_FILENAMES = ['ANSWER_FILE.json']


def flatten_dict_list(dict_list):
    """
    Flatten a list of dicts into one dict
    """
    result = {}
    for d in dict_list:
        result.update(d)
    return result


def add_gt_to_answers(client, model_name, task_time=PILOT_inner,
                      gt_path=f'data/pilot_study/hit_data_refined.json',
                      output_path='data/pilot_study/pilot_answers.json',
                      is_save=True):
    all_hits_answers = get_hits_answers_by_time(client, hit_title=TASK_TITLE, task_time=task_time)
    # ground-truth data
    with open(gt_path, 'r') as f:
        hit_data = json.load(f)
    hit_data_flatten = flatten_dict_list(hit_data)
    for hit_id in all_hits_answers:
        assignments = all_hits_answers[hit_id]
        for assignment in assignments:
            # combine all dicts in assignment['Answer'] into one dict
            answers = {}
            for answer in assignment['Answer']:
                answers.update(answer)
            assignment['Answer'] = answers
            for key, sample in assignment['Answer'].items():
                sample['type'] = hit_data_flatten[key]['type']
                if sample['type'] == 'negative':
                    sample['gt_answer'] = [-2]
                else:
                    sample['gt_answer'] = hit_data_flatten[key]['ans_sens'][model_name]
    if is_save:
        with open(output_path, 'w') as f:
            json.dump(all_hits_answers, f, indent=4)
    return all_hits_answers


def jaccard_similarity(selected: List, ground_truth: List):
    # make sure two lists are all strings
    selected = [str(s) for s in selected]
    ground_truth = [str(s) for s in ground_truth]
    # Jaccard similarity = intersection / union
    intersection = len(set(selected).intersection(set(ground_truth)))
    union = len(set(selected).union(set(ground_truth)))
    return intersection / union


def F1_score(selected: List, ground_truth: List, is_rtn_pr=False):
    # make sure two lists are all strings
    selected = [str(s) for s in selected]
    ground_truth = [str(s) for s in ground_truth]
    intersection = len(set(selected).intersection(set(ground_truth)))
    precision = intersection / len(selected) if len(selected) != 0 else 0
    recall = intersection / len(ground_truth) if len(ground_truth) != 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 2) if precision + recall != 0 else 0
    if is_rtn_pr is True:
        return precision, recall, f1
    else:
        return f1

def get_sen_id(sen, sen_list):
    # assume sen and target are exactly the same
    for i, s in enumerate(sen_list):
        if sen in s:
            return i
    return -1


def get_acc_distribution(all_hits_answers, is_plot=True, fig_save=False):
    # get assignment accuracy distribution, use F1 score and Jaccard similarity
    F1_acc = []
    Jaccard_acc = []
    worker_ids = []
    assign_ids = []

    assign_dict = {}
    for hit_id in all_hits_answers:
        for assignment in all_hits_answers[hit_id]:
            F1_one_assign = []
            Jaccard_one_assign = []
            worker_ids.append(assignment['WorkerId'])
            assign_ids.append(assignment['AssignmentId'])
            for key, sample in assignment['Answer'].items():
                F1_one_assign.append(F1_score(sample['selected_explanations'], sample['gt_answer']))
                Jaccard_one_assign.append(jaccard_similarity(sample['selected_explanations'], sample['gt_answer']))
            F1_acc.append(round(np.mean(F1_one_assign), 2))
            Jaccard_acc.append(round(np.mean(Jaccard_one_assign), 2))
            assign_dict[assignment['AssignmentId']] = {'Worker_ID': assignment['WorkerId'],
                                                       'F1_score': round(np.mean(F1_one_assign), 2)}
    if is_plot:
        # plot accuracy distribution
        fig, ax = plt.subplots(figsize=(30, 6))
        x = np.arange(len(worker_ids))
        species = x
        penguin_means = {
            'F1 Score': F1_acc,
            'Jaccard Similarity': Jaccard_acc,
        }

        width = 0.4  # the width of the bars
        multiplier = 0

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=4)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy')
        ax.set_title('Assignment Accuracy Distribution')
        ax.set_xticks(x + width, species)
        ax.legend(loc='upper right', ncols=4)
        ax.set_ylim(0, 1)
        if fig_save:
            plt.savefig('assign_performance.png')
        plt.show()
    return assign_dict


def get_time_distribution(all_hits_answers, is_plot=True, fig_save=False):
    working_time = []
    for hit_id in all_hits_answers:
        for assignment in all_hits_answers[hit_id]:
            working_time.append(round(assignment['WorkingTime'], 2))
    if is_plot:
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(25, 15))

        # Create a boxplot using Matplotlib
        x = list(range(len(working_time)))
        bar_plot = ax.bar(x, working_time)

        # Set the x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(x)

        # Set the title and labels for the plot
        ax.set_title('Assignment Working Time')
        ax.set_xlabel('Annotator ID')
        ax.set_ylabel('Time Duration (Minutes)')

        # Add numbers on top of the bars
        for bar in bar_plot:
            height = bar.get_height()
            ax.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom')

        # Show the plot
        plt.ylim(0, 60)
        if fig_save:
            plt.savefig('working_time.png')
        plt.show()
    return working_time


def get_worker_performance(all_hits_answers, is_plot=True, fig_save=False):
    # plot worker average accuracy distribution, use F1 score
    worker_performance = {}
    for hit_id in all_hits_answers:
        for assignment in all_hits_answers[hit_id]:
            F1_one_assign = []
            Jaccard_one_assign = []
            # calculate average performance for each assignment
            for key, sample in assignment['Answer'].items():
                F1_one_assign.append(F1_score(sample['selected_explanations'], sample['gt_answer']))
                Jaccard_one_assign.append(jaccard_similarity(sample['selected_explanations'], sample['gt_answer']))
            F1_acc_mean = np.mean(F1_one_assign)
            Jaccard_mean = np.mean(Jaccard_one_assign)
            # print(f"Worker {assignment['WorkerId']} has F1 score {F1_acc_mean} on assignment {assignment['AssignmentId']}")
            if assignment['WorkerId'] not in worker_performance:
                worker_performance[assignment['WorkerId']] = {'F1': [F1_acc_mean], 'Jaccard': [Jaccard_mean]}
            else:
                worker_performance[assignment['WorkerId']]['F1'].append(F1_acc_mean)
                worker_performance[assignment['WorkerId']]['Jaccard'].append(Jaccard_mean)
    # calculate average performance for each worker
    worker_performance_mean = {worker: {'F1': round(np.mean(performance['F1']), 3),
                                        'Jaccard': round(np.mean(performance['Jaccard']), 3)}
                               for worker, performance in worker_performance.items()}

    df = pd.DataFrame.from_dict(data=worker_performance_mean, orient='index')
    df_melted = pd.melt(df.reset_index(), id_vars='index', var_name='Performance', value_name='Value')
    df_melted['ShortLabel'] = df_melted['index'].map(lambda x: x[::])  # Adjust the slice as needed

    if is_plot:
        plt.figure(figsize=(50, 10))
        ax = sns.barplot(x='ShortLabel', y='Value', hue='Performance', data=df_melted)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
        plt.xticks(rotation=90)
        plt.xlabel('Worker ID')
        plt.ylabel('Performance')
        plt.title('Worker Average Performance')
        plt.legend(title='Performance')
        plt.ylim(0, 1)
        # Show the plot
        if fig_save:
            plt.savefig('worker_performance.png')
        plt.show()
    return worker_performance_mean


def print_answer_statistics(answer_path='data/pilot_study/my_pilot_answers.json'):
    with open(answer_path, 'r') as f:
        all_hits_answers = json.load(f)
    # count how many unique workers
    worker_ids = set()
    assign_cnt = 0
    for hit_id in all_hits_answers:
        assign_cnt += len(all_hits_answers[hit_id])
        for assignment in all_hits_answers[hit_id]:
            worker_ids.add(assignment['WorkerId'])
    print(f"Total number of workers: {len(worker_ids)}, total number of assignments: {assign_cnt}")
    print(f"The worker set is: {worker_ids}")


def filter_history_annotators(answer_filenames, task_type=None, threshold=REWARD_THRESHOLD):
    if task_type is None or task_type not in ['pilot_study', 'actual_task']:
        raise AssertionError("Please specify the task type as one of 'pilot_study' or 'actual_task'.")
    # return bad and good annotators based on threshold, combine them you get all annotators
    bad_annotator = {}
    good_annotator = {}
    for answer_filename in answer_filenames:
        with open(f'data/{task_type}/{answer_filename}', 'r') as f:
            pilot_answer = json.load(f)
        worker_performance = get_worker_performance(pilot_answer, is_plot=False)
        for worker, performance in worker_performance.items():
            if performance['F1'] < threshold:
                bad_annotator.update({worker: performance})
            else:
                good_annotator.update({worker: performance})
    return bad_annotator, good_annotator


def filter_history_assignments(answer_filenames, task_type=None, threshold=REWARD_THRESHOLD):
    if task_type is None or task_type not in ['pilot_study', 'actual_task']:
        raise AssertionError("Please specify the task type as one of 'pilot_study' or 'actual_task'.")
    # return bad and good assignments based on threshold, combine them you get all annotators
    bad_assignments = []
    good_assignments = []
    for answer_filename in answer_filenames:
        with open(f'data/{task_type}/{answer_filename}', 'r') as f:
            pilot_answer = json.load(f)
        assign_dict = get_acc_distribution(pilot_answer, is_plot=False)
        for assign_id, performance in assign_dict.items():
            if performance['F1_score'] < threshold:
                bad_assignments.append((assign_id, performance))
            else:
                good_assignments.append((assign_id, performance))
    return bad_assignments, good_assignments


def pilot_analysis(task_time, if_production=False, model_name=None, gt_path='./data/hit_data_actual.json',
                   ans_path=None, fig_save=False):
    client, _ = connect_to_turk(create_hits_in_production=if_production)
    if ans_path is None:
        raise AssertionError("Please specify the filename of the pilot study.")
    if model_name is None:
        raise AssertionError("Please specify the model name.")
    if os.path.exists(ans_path):
        with open(ans_path, 'r') as f:
            pilot_answer = json.load(f)
    else:
        pilot_answer = add_gt_to_answers(client, model_name, task_time=task_time, gt_path=gt_path,
                                         output_path=ans_path, is_save=True)
    get_acc_distribution(pilot_answer, is_plot=True, fig_save=fig_save)
    get_time_distribution(pilot_answer, is_plot=True, fig_save=fig_save)
    get_worker_performance(pilot_answer, is_plot=True, fig_save=fig_save)


def approve_tasks(unprocessed_studies, task_type=None, is_approve=False):
    """
    @param unprocessed_studies: a list of filenames of unprocessed studies
    @param is_approve: whether to approve the assignments
    @return: None
    This function deals with unprocessed_studies in three steps:
    1. approve hits in unprocessed_pilot
    2. reward good assignments in unprocessed_pilot
    3. notify good annotators and bad annotators in unprocessed_pilot
    """
    if task_type is None or task_type not in ['pilot_study', 'actual_task']:
        raise AssertionError("Please specify the task type as one of 'pilot_study' or 'actual_task'.")
    client, _ = connect_to_turk(create_hits_in_production=True)
    for pilot_filename in unprocessed_studies:
        with open(f'data/{task_type}/{pilot_filename}', 'r') as f:
            pilot_answer = json.load(f)
        hit_ids = list(pilot_answer.keys())
        if is_approve:
            approve_hits(client, hit_ids)


def reward_good_assignments(client, unprocessed_tasks, task_type=None, is_reward=False, threshold=REWARD_THRESHOLD):
    """
    @param unprocessed_tasks: a list of filenames of unprocessed pilot studies
    @param is_reward: whether to reward good assignments
    @param threshold: threshold to determine good assignments
    @param task_type: task type, one of 'pilot_study' or 'actual_task'
    @return: None
    This function rewards good assignments in unprocessed_pilot
    """
    if task_type is None or task_type not in ['pilot_study', 'actual_task']:
        raise AssertionError("Please specify the task type as one of 'pilot_study' or 'actual_task'.")
    bad_assignments, good_assignments = filter_history_assignments(unprocessed_tasks, task_type=task_type,
                                                                   threshold=threshold)
    print(f"good assignments:\n{good_assignments}\n")
    print(f"len of good assignments: {len(good_assignments)}")
    if len(unprocessed_tasks) == 1:
        good_assign_filename = unprocessed_tasks[0].split('.')[0] + '_good_assignments.json'
    else:
        good_assign_filename = 'good_assignments.json'
    with open(f'data/{task_type}/{good_assign_filename}', 'w') as f:
        good_assignments_dict = {assign_id: detail for assign_id, detail in good_assignments}
        json.dump(good_assignments_dict, f, indent=4)
    # send bonus to good assignments
    if is_reward:
        for assign_id, detail in good_assignments:
            client.send_bonus(WorkerId=detail['Worker_ID'], BonusAmount=str(REWARD_PER_HIT), AssignmentId=assign_id,
                              Reason=f'Good job on the task \"{TASK_TITLE}\"! Thank you for your participation! Please '
                                     'contact us if you have any suggestions or questions about this task.')
            print(
                f"Bonus {REWARD_PER_HIT} has been sent to worker {detail['Worker_ID']} for assignment {assign_id}.")


def notify_annotators(client, unprocessed_tasks, task_type=None, threshold=REWARD_THRESHOLD, is_notify=False):
    if task_type is None or task_type not in ['pilot_study', 'actual_task']:
        raise AssertionError("Please specify the task type as one of 'pilot_study' or 'actual_task'.")
    # notify good annotators and bad annotators
    bad_annotators, good_annotators = filter_history_annotators(unprocessed_tasks, task_type=task_type,
                                                                threshold=threshold)
    print(f"good annotators:\n{good_annotators}\n")
    print(f"len of good annotators: {len(good_annotators)}")
    if len(unprocessed_tasks) == 1:
        good_annotator_filename = bad_annotator_filename = unprocessed_tasks[0].split('.')[0] + '_'
    else:
        good_annotator_filename = bad_annotator_filename = ''
    # record good and bad annotators to file
    with open(f'data/{task_type}/{good_annotator_filename}good_annotators.json', 'w') as f:
        json.dump(good_annotators, f, indent=4)
    with open(f'data/{task_type}/{bad_annotator_filename}bad_annotators.json', 'w') as f:
        json.dump(bad_annotators, f, indent=4)
    if is_notify:
        notify(client, list(good_annotators.keys()), is_good_annotator=True)
        notify(client, list(bad_annotators.keys()), is_good_annotator=False)


if __name__ == '__main__':
    # test pilot study analysis
    client, _ = connect_to_turk(create_hits_in_production=True)
    model_name = 'llama2-70b'
    pilot_analysis(task_time=ACTUAL_LLAMA_FULL, if_production=True, model_name=model_name,
                   gt_path=f'./data/actual_task/hit_data_{model_name}_full.json',
                   ans_path=f'./data/actual_task/{model_name}_full_answers.json')
