"""Text instructions bound to the chosen contour color."""



def rewrite_question(question, color='blue'):
    return f'Within the {color}-outlined area of the CT volume, {question.strip()}'


def system_prompt(color='blue', thickness=1):
    return (f'The CT slices contain a {color} boundary of thickness {thickness} pixel(s). '
            'Answer the question using the marked region. Describe only the requested structure. '
            'If the requested abnormality is absent, answer No finding. Keep the answer concise.')
