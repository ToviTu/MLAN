import pdb

def boolq_prompt(input):
    passage = input['passage']
    question = input['question']

    return f"User: Context: {passage}\nQuestion: {question}?\nPlease answer the question with 'yes' or 'no'.\nAssistant: The best answer is\""
    # return f"{passage}\nQuestion: {question}?\nAnswer:"

def doc_to_target(doc):
    return "yes" if doc["answer"] == True else "no"