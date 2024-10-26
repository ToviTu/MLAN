def cosmosqa_prompt_em(input):
    context = input['context']
    question = input['question']
    answer0 = input['answer0']
    answer1 = input['answer1']
    answer2 = input['answer2']
    answer3 = input['answer3']
    
    return (
        f"User: {context}\n"
        f"Question: {question} Choose the best option from the choices provided:\n"
        f"A: {answer0}\n"
        f"B: {answer1}\n"
        f"C: {answer2}\n"
        f"D: {answer3}\n"
        f"Assistant: The best answer is\""
    )
    
def cosmosqa_prompt(input):
    context = input['context']
    question = input['question']
    answer0 = input['answer0']
    answer1 = input['answer1']
    answer2 = input['answer2']
    answer3 = input['answer3']

    return (
        f"Context: {context}\n Question: {question}\n"
        f"A: {answer0}, "
        f"B: {answer1}, "
        f"C: {answer2}, "
        f"D: {answer3}\n"
        f"Answer:"
    )
    
def label_to_letter(input):
    label = input['label']
    return ["A", "B", "C", "D"][label]

def doc_to_target(doc):
    label = doc["label"]
    choices = [doc["answer0"], doc["answer1"], doc["answer2"], doc["answer3"]]
    return choices[label]

def doc_to_choice(doc):
    return [doc["answer0"], doc["answer1"], doc["answer2"], doc["answer3"]]