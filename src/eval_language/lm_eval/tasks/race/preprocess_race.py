import ast


def process_ast(string):
    return ast.literal_eval(string)


def last_problem(doc):
    return process_ast(doc["problems"])[-1]


def get_answer_option(problem):
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    answer = letter_to_num[problem["answer"]]
    return problem["options"][answer]


def doc_to_choice(doc):
    problem = last_problem(doc)
    choices = [problem["options"][i] for i in range(4)]
    return choices


def doc_to_text(doc):
    text = "Article: " + doc["article"] + "\n\n"
    for problem in process_ast(doc["problems"])[:-1]:
        if problem["question"][-6:] == "  _  .":
            text += problem["question"][-5:] + get_answer_option(problem) + "\n"
        else:
            question = "Question: " + problem["question"] + "\n"
            answer = "Answer: " + get_answer_option(problem) + "\n"
            text += question + answer
    text += last_problem(doc)["question"]
    return text

def doc_prompt(input):
    article = input["article"]
    problem = last_problem(input)
    question = problem["question"]
    choices_text = problem["options"]
    choices_label = ["A", "B", "C", "D"]
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])

    return f"User: Context: {article}\nQuestion: {question}\nChoose the best option from the choices provided:\n{choices_formatted}\nAssistant: The best answer is\""

def doc_to_target(doc):
    letter_to_num = {"A": 0, "B": 1, "C": 2, "D": 3}
    answer = letter_to_num[last_problem(doc)["answer"]]
    return answer

def doc_to_target_em(doc):
    answer = last_problem(doc)["answer"]
    return answer
