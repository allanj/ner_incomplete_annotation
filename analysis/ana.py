
from typing import List
from span import Span


dataset = "youku"

four = "../results/"+dataset+".our_soft.0.5_o_prefer.final.m.res.first"
fcomp = "../results/"+dataset+".simple.1.0.m.res.first"
fspan = "../results/"+dataset+".our_soft.0.5.spans.txt"


fs = open(fspan, 'r', encoding='utf-8')
ss = fs.readlines()
spans = {}
for line in ss:
    line = line.rstrip()
    vals = line.split()
    words = ' '.join(vals[1:])
    if words in spans:
        spans[words].append(vals[0])
    else:
        spans[words] = [vals[0]]
fs.close()

def to_sapn(output):
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
    return output_spans

def read_inst(file) -> List:

    f = open(file, 'r', encoding='utf-8')
    lines = f.readlines()
    insts = []
    inst = []
    for line in lines:
        line = line.rstrip()
        if line == "":
            insts.append(inst)
            inst = []
        else:
            inst.append(line.split())
    f.close()
    return insts

our_insts = read_inst(four)
comp_insts = read_inst(fcomp)


total_comp_correct = 0
total_num_non_pred = 0
total_in_span = 0
for i in range(len(our_insts)):
    our_inst = our_insts[i]
    comp_inst = comp_insts[i]
    words = our_inst[0]
    our_pred = to_sapn(our_inst[2])
    comp_pred = to_sapn(comp_inst[2])
    gold = to_sapn(our_inst[1])

    comp_corrects = comp_pred.intersection(gold)
    non_predtable = comp_corrects.difference(our_pred)
    total_num_non_pred += len(non_predtable)
    total_comp_correct += len(comp_corrects)

    for span in non_predtable:
        span_str = ' '.join(words[span.left:(span.right+1)])
        print(span_str + " " + span.type)
        type = span.type
        if span_str in spans and type in spans[span_str] :
            total_in_span += 1


err = total_num_non_pred / total_comp_correct * 100
err_span = total_in_span / total_num_non_pred * 100
print("Error is correctly predicted comp by wrongly by our apprapch: {:.2f}%".format(err))
print("In this error, caused by span error: {:.2f}%".format(err_span))



