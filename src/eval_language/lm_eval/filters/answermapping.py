from lm_eval.api.filter import Filter
import pdb
from lm_eval.api.registry import register_filter
import string
import re


@register_filter("answermapping")
class AnswerMappingFilter(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(self, resps, docs):
        def clean_response(resp):
            return re.sub(r'[^\w\s]', '', resp).strip().lower()

        def word_match_score(resp, choice):
            resp_words = set(clean_response(resp).split())
            choice_words = set(clean_response(choice).split())
            common_words = resp_words & choice_words
            total_words = choice_words
            if len(total_words) == 0:
                return 0
            return len(common_words) / len(total_words)

        def filter_set(inst, doc):
            mapped_resps = []
            for resp in inst:
                # if it is already a valid response, keep it
                if resp in doc['choices']['label']:
                    mapped_resps.append(resp)
                else:
                    cleaned_resp = clean_response(resp)
                    max_score = 0
                    best_match = resp

                    for label, text in zip(doc['choices']['label'], doc['choices']['text']):
                        score = word_match_score(resp, text)
                        if score > max_score:
                            max_score = score
                            best_match = label if score >= 1.0 else resp

                    mapped_resps.append(best_match)

            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps
    
@register_filter("answermapping_hellaswag")
class AnswerMappingFilter_hellaswag(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(self, resps, docs):
        def clean_response(resp):
            return re.sub(r'[^\w\s]', '', resp).strip().lower()

        def word_match_score(resp, choice):
            resp_words = set(clean_response(resp).split())
            choice_words = set(clean_response(choice).split())
            common_words = resp_words & choice_words
            total_words = choice_words
            if len(total_words) == 0:
                return 0
            return len(common_words) / len(total_words)

        def filter_set(inst, doc):
            mapped_resps = []
            label_map = ['A', 'B', 'C', 'D']  # Assuming the choices are ordered and labeled A, B, C, D

            for resp in inst:
                # if it is already a valid response, keep it
                if resp in label_map:
                    mapped_resps.append(resp)
                else:
                    cleaned_resp = clean_response(resp)
                    max_score = 0
                    best_match = resp

                    for i, text in enumerate(doc['choices']):  # Iterate over the choices directly
                        score = word_match_score(cleaned_resp, text)
                        if score > max_score:
                            max_score = score
                            best_match = label_map[i] if score >= 1.0 else resp

                    mapped_resps.append(best_match)

            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps
    
@register_filter("answermapping_cosmosqa")
class AnswerMappingFilter_cosmosqa(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def apply(self, resps, docs):
        def clean_response(resp):
            return re.sub(r'[^\w\s]', '', resp).strip().lower()

        def word_match_score(resp, choice):
            resp_words = set(clean_response(resp).split())
            choice_words = set(clean_response(choice).split())
            common_words = resp_words & choice_words
            total_words = choice_words
            if len(total_words) == 0:
                return 0
            return len(common_words) / len(total_words)

        def filter_set(inst, doc):
            mapped_resps = []
            for resp in inst:
                # if it is already a valid response, keep it
                if resp in ['A', 'B', 'C', 'D']:
                    mapped_resps.append(resp)
                else:
                    cleaned_resp = clean_response(resp)
                    max_score = 0
                    best_match = resp

                    choices = [doc[f'answer{i}'] for i in range(4)]
                    labels = ['A', 'B', 'C', 'D']
                    for label, text in zip(labels, choices):
                        score = word_match_score(resp, text)
                        if score > max_score:
                            max_score = score
                            best_match = label if score >= 1.0 else resp

                    mapped_resps.append(best_match)

            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps

@register_filter("answermapping_race")
class AnswerMappingFilterRACE(Filter):
    def __init__(self, fallback: str = "[invalid]") -> None:
        self.fallback = fallback

    def process_ast(self, string):
        import ast
        return ast.literal_eval(string)

    def apply(self, resps, docs):
        def clean_response(resp):
            return re.sub(r'[^\w\s]', '', resp).strip().lower()

        def word_match_score(resp, choice):
            resp_words = set(clean_response(resp).split())
            choice_words = set(clean_response(choice).split())
            common_words = resp_words & choice_words
            total_words = choice_words
            if len(total_words) == 0:
                return 0
            return len(common_words) / len(total_words)

        def filter_set(inst, doc):
            mapped_resps = []
            for resp in inst:
                # if it is already a valid response, keep it
                if resp in ['A', 'B', 'C', 'D']:
                    mapped_resps.append(resp)
                else:
                    cleaned_resp = clean_response(resp)
                    max_score = 0
                    best_match = resp

                    try:
                        last_problem = self.process_ast(doc['problems'])[-1]
                        choices = last_problem['options']
                        labels = ['A', 'B', 'C', 'D']
                        for label, text in zip(labels, choices):
                            score = word_match_score(resp, text)
                            if score > max_score:
                                max_score = score
                                best_match = label if score >= 1.0 else resp
                    except (KeyError, IndexError, ValueError):
                        best_match = resp

                    mapped_resps.append(best_match)

            return mapped_resps

        filtered_resps = []
        for i, resp in enumerate(resps):
            filtered_resps.append(filter_set(resp, docs[i]))

        flat_filtered_resps = [item for sublist in filtered_resps for item in sublist] if any(isinstance(i, list) for i in filtered_resps) else filtered_resps

        return flat_filtered_resps

@register_filter("true_false_conversion")
class TrueFalseConversionFilter(Filter):
    def __init__(self) -> None:
        super().__init__()

    def apply(self, resps, docs):
        def convert_true_false(resp):
            if resp.lower() == "true":
                return "yes"
            elif resp.lower() == "false":
                return "no"
            else:
                return resp

        converted_resps = []
        for resp in resps:
            converted_resps.append([convert_true_false(r) for r in resp])

        flat_filtered_resps = [item for sublist in converted_resps for item in sublist] if any(isinstance(i, list) for i in converted_resps) else converted_resps

        return flat_filtered_resps
    
def test_answer_mapping_filter():
    filter = AnswerMappingFilter()

    responses = [["aaqgqgqg qfqg"]]
    docs = [{
        "id": "Mercury_7202353",
        "question": "When the soil is saturated, earthworms move to the top of the soil to obtain oxygen. This behavior is evidence of which biological concept in earthworms?",
        "choices": {
            "text": ["learned behavior", "migration instinct", "response to stimuli", "reproductive strategy"],
            "label": ["A", "B", "C", "D"]
        },
        "answerKey": "C"
    }]

    expected_responses = ["B"] 

    filtered_responses = filter.apply(responses, docs)
    
    assert filtered_responses == expected_responses, f"Expected {expected_responses} but got {filtered_responses}"
    print("AnswerMappingFilter test passed!")
    


if __name__ == "__main__":
    test_answer_mapping_filter()