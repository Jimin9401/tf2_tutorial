import json

def load_json(f_name):
    dataset = json.load(open(f_name))
    context = []
    question = []
    answer = []
    start_idx = []
    for d in dataset['data']:
        for i, qa in enumerate(d['paragraphs']):
            context.append(qa['context'])
            for qas in qa['qas']:
                answer.append(qas['answers'][0]['text'])
                start_idx.append(qas['answers'][0]['answer_start'])
                question.append((qas['question'], i))

    return context, answer, start_idx, question


