def trivia_qa_prompt(input):
    question = input['question']

    return f"User: {question}\nAnswer the question using a single word or phrase\nAssistant: The best answer is\""