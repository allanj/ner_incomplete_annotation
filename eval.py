

from span import Span


def save_results(insts, res_file):
    f = open(res_file, 'w', encoding='utf-8')
    for inst in insts:
        output = ' '.join(inst.output)
        prediction = ' '.join(inst.prediction)
        sent = ' '.join(inst.input.words)
        f.write(sent + "\n")
        f.write(output + "\n")
        f.write(prediction + "\n")
        f.write("\n")
    f.close()

def save_spans(spans, res_file):
    f = open(res_file, 'w', encoding='utf-8')
    for span in spans:
        f.write(span + "\n")
    f.close()


## the input to the evaluation should already have
## have the predictions which is the label.
## iobest tagging scheme
def evaluate(insts):

    p = 0
    total_entity = 0
    total_predict = 0

    for inst in insts:

        output = inst.output
        prediction = inst.prediction
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return [precision, recall, fscore]
