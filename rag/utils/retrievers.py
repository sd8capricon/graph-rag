from rag.schema.retrievers import Node


def collect_answers(node: Node):
    answers: list[str] = []
    if node.answer:
        answers.append(node.answer)
    for child in node.children:
        answers.extend(collect_answers(child))
    return answers
