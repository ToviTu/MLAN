from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter

@register_filter("before_first_newline")
class BeforeFirstNewlineFilter(Filter):
    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):
        def process_response(response):
            newline_index = response.find('\n')
            if newline_index != -1:
                # Return the content before the first newline
                return response[:newline_index]
            else:
                # If no newline is found, return the whole response
                return response
        
        filtered_resps = []
        for resp in resps:
            filtered_resp = [process_response(r) for r in resp]
            filtered_resps.append(filtered_resp)
        
        return filtered_resps

# Test the filter
def test_before_first_newline_filter():
    filter = BeforeFirstNewlineFilter()

    responses = [["sunlight is the source of energy for nearly all ecosystems\nEcosystems are"]]
    expected_responses = [["sunlight is the source of energy for nearly all ecosystems"]]

    filtered_responses = filter.apply(responses, [])
    
    assert filtered_responses == expected_responses, f"Expected {expected_responses} but got {filtered_responses}"
    print("BeforeFirstNewlineFilter test passed!")

if __name__ == "__main__":
    test_before_first_newline_filter()