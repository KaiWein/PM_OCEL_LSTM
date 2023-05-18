import csv
from operator import itemgetter
import numpy as np
from numpy.compat import unicode

def folding_this(lines,Ptimeseqs,Ptimeseqs2,Ptimeseqs3,Ptimeseqs4,nb_itemseqs,PtimeseqsF,seeded = 42, train =True):
    np.random.seed(seeded)
    numlines = len(lines)
    indices = np.random.permutation(numlines)
    elems_per_fold = int(round((numlines+1) / 3))

    idx1 = indices[:elems_per_fold]
    idx2 = indices[elems_per_fold:2 * elems_per_fold]
    idx3 = indices[2 * elems_per_fold:]

            
    fold1 = list(itemgetter(*idx1)(lines))
    fold1_t = list(itemgetter(*idx1)(Ptimeseqs))
    fold1_t2 = list(itemgetter(*idx1)(Ptimeseqs2))
    fold1_t3 = list(itemgetter(*idx1)(Ptimeseqs3))
    fold1_t4 = list(itemgetter(*idx1)(Ptimeseqs4))
    fold1_t5 = list(itemgetter(*idx1)(nb_itemseqs))
    fold1_t6 = list(itemgetter(*idx1)(PtimeseqsF))

    with open('./output_files/folds/fold1.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold1, fold1_t):
            spamwriter.writerow([unicode(s) + '#{}'.format(t) for s, t in zip(row, timeseq)])

    fold2 = list(itemgetter(*idx2)(lines))
    fold2_t = list(itemgetter(*idx2)(Ptimeseqs))
    fold2_t2 = list(itemgetter(*idx2)(Ptimeseqs2))
    fold2_t3 = list(itemgetter(*idx2)(Ptimeseqs3))
    fold2_t4 = list(itemgetter(*idx2)(Ptimeseqs4))
    fold2_t5 = list(itemgetter(*idx2)(nb_itemseqs))
    fold2_t6 = list(itemgetter(*idx2)(PtimeseqsF))
    with open('./output_files/folds/fold2.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold2, fold2_t):
            spamwriter.writerow([unicode(s) + '#{}'.format(t) for s, t in zip(row, timeseq)])


    fold3 = list(itemgetter(*idx3)(lines))
    fold3_t = list(itemgetter(*idx3)(Ptimeseqs))
    fold3_t2 = list(itemgetter(*idx3)(Ptimeseqs2))
    fold3_t3 = list(itemgetter(*idx3)(Ptimeseqs3))
    fold3_t4 = list(itemgetter(*idx3)(Ptimeseqs4))
    fold3_t5 = list(itemgetter(*idx3)(nb_itemseqs))
    fold3_t6 = list(itemgetter(*idx3)(PtimeseqsF))
    with open('./output_files/folds/fold3.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row, timeseq in zip(fold3, fold3_t):
            spamwriter.writerow(
                [unicode(s).encode("utf-8") + '#{}'.format(t).encode('utf-8') for s, t in zip(row, timeseq)])
    if train:
        lines = fold1 + fold2
        lines_t = fold1_t + fold2_t
        lines_t2 = fold1_t2 + fold2_t2
        lines_t3 = fold1_t3 + fold2_t3
        lines_t4 = fold1_t4 + fold2_t4
        lines_t5 = fold1_t5 + fold2_t5
        lines_t6 = fold1_t6 + fold2_t6
    else:
        return 0

    return lines, lines_t, lines_t2, lines_t3, lines_t4, lines_t5, lines_t6