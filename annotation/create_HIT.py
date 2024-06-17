###
# Use jinja2 + HTML templating to send TransExp data to the front
###
from jinja2 import Environment, FileSystemLoader
import os
import hashlib
import random
import jsonlines

random.seed(2023)

# path setting
annotation_dir = ''
annotation_platform_dir = os.path.join(annotation_dir, 'annotation_platform')

hit_generated_dir = os.path.join(annotation_platform_dir, 'HIT_generated')
hit_id_dir = os.path.join(annotation_platform_dir, 'hit_id')

actual_hit_dir = os.path.join(annotation_platform_dir, 'actual_task_hit')
pilot_hit_dir = os.path.join(annotation_platform_dir, 'pilot_study_hit')
pos_neg_por_dir = os.path.join(annotation_dir, 'pos_neg_por')

# qualifications setting
countries = COUNTRIES_LIST

# Task related configuration
TaskAttributes = {
    'MaxAssignments': MAX_ASSIGN,
    'LifetimeInSeconds': LIFE_SECONDS,  # How long the task will be available on the MTurk website
    'AssignmentDurationInSeconds': DURATION_SECONDS,  # How long Workers have to complete each item
    'AutoApprovalDelayInSeconds': AUTO_APPROVAL_SECONDS,  # How long the system will wait for auto approval
    'Reward': REWARD,  # The reward you will offer Workers for each response, in US dollars
    'QualificationRequirements': qualified_qualifications,
    'Title': 'Assessing the quality of AI fact checking',
    'Keywords': 'fact checking, explanation, evaluation',
    'Description': 'Please verify transparency and helpfulness of the explanation.'
}


def create_hit_data(nor_data_path: str, pos_data_path: str, neg_data_path: str,
                    nor_per_hit=4, pos_per_hit=1, neg_per_hit=1):
    """
    Create annotation data for the pilot study, return data is a list of dict,
    each dict contains a mixture of normal, positive and negative data for one HIT
    """
    hit_data = list()
    with open(nor_data_path, 'r') as f:
        nor_data = json.load(f)
    with open(pos_data_path, 'r') as f:
        pos_data = json.load(f)
    with open(neg_data_path, 'r') as f:
        neg_data = json.load(f)
    # mix data for each HIT, pick 6 normal data in order,
    # randomly sample 2 positive data and 1 negative data
    nor_key_list = list(nor_data.keys())
    for i in range(0, len(nor_key_list), nor_per_hit):
        nor_sample_data = {k: nor_data[k] for k in nor_key_list[i:i + nor_per_hit]}
        pos_sample_data = {k: pos_data[k] for k in random.sample(pos_data.keys(), pos_per_hit)}
        neg_sample_data = {k: neg_data[k] for k in random.sample(neg_data.keys(), neg_per_hit)}
        hit_data.append({**nor_sample_data, **pos_sample_data, **neg_sample_data})
    return hit_data


def main():
    model_name = 'llama2-70b'
    mode = 'full'
    is_creating_hit = True
    is_testing = False
    is_production = False

    # hit_data is used to create HITs directly, it's in a group of 6 normal, 2 positive and 1 negative data
    hit_data_path = f'annotation/actual_task_hit/hit_data_{model_name}_{mode}.json'
    if not os.path.exists('annotation_platform/HIT_generated'):
        os.mkdir('annotation_platform/HIT_generated')

    if not os.path.exists(hit_data_path):
        print("Creating HIT data...")
        # create pilot data by mixing normal, positive and negative data
        nor_data_path = f'{pos_neg_por_dir}/{model_name}_{mode}_nor_data.json'
        pos_data_path = f'{pos_neg_por_dir}/{model_name}_{mode}_pos_data.json'
        neg_data_path = f'{pos_neg_por_dir}/{model_name}_{mode}_neg_data.json'
        hit_data = create_hit_data(nor_data_path, pos_data_path, neg_data_path)

        with open(hit_data_path, 'w') as f:
            json.dump(hit_data, f, indent=4)
        print("Finish creating HIT data.")
    else:
        print("Loading HIT data...")
        with open(hit_data_path, 'r') as f:
            hit_data = json.load(f)
        print("Finish loading HIT data.")

    # connect to MTurk
    client, mturk_environment = connect_to_turk(create_hits_in_production=is_production)

    # change const model_name = "gpt35" in TransExp.html
    with open(f'{annotation_platform_dir}/annotation_platform.html', 'r', encoding='UTF-8') as f:
        html = f.read()
    html = html.replace('const model_name = "gpt4"', f'const model_name = "{model_name}"')

    # Initialize the Jinja2 environment
    env = Environment(loader=FileSystemLoader(".."))
    # template = env.get_template(html)
    template = env.from_string(html)

    question_xml = """<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11
    -11/HTMLQuestion.xsd"> <HTMLContent><![CDATA[{}]]></HTMLContent> <FrameHeight>650</FrameHeight> </HTMLQuestion>"""

    results = []

    # HIT settings
    if is_testing:
        hit_data = json.load(open(f'{pilot_hit_dir}/hit_data_test.json', 'r'))

    for batch in hit_data:
        rendered_html = template.render(
            batch_data=batch,
        )
        if is_creating_hit:
            question_xml = question_xml.format(rendered_html)
            try:
                response = client.create_hit(
                    **TaskAttributes,
                    Question=question_xml
                )
                result = {
                    'hit_id': response['HIT']['HITId'],
                    'hit_type_id': response['HIT']['HITTypeId'],
                    'view_link': mturk_environment['preview'] + "?groupId={}".format(response['HIT']['HITTypeId']),
                    'datatime': str(datetime.now())
                }
                results.append(result)
                with open(f"{hit_generated_dir}/{result['hit_id']}.html", "w", encoding='utf-8') as f:
                    f.write(rendered_html)
                print(f"Finish creating hit {response['HIT']['HITId']}!")
            except Exception as e:
                print(e)
                print("Error in creating HITs!")
        else:
            if is_testing:
                hash_code = 'test'
            else:
                hash_code = hashlib.md5(str(batch).encode('utf-8')).hexdigest()
            with open(f"{hit_generated_dir}/{hash_code}.html", "w", encoding='utf-8') as f:
                f.write(rendered_html)
        if is_testing:
            break

    if is_creating_hit:
        with jsonlines.open(f"{hit_id_dir}/hit_ids_{model_name}_{mode}.json", 'w') as writer:
            writer.write_all(results)
            print(f"Finish writing hit_ids_{model_name}_{mode}.json!")


if __name__ == '__main__':
    main()
