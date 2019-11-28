import numpy as np
import glob


def readFile(filePath):
    contents = []
    for i in range(len(filePath)):
        f = open(filePath[i], 'r', encoding="utf8", errors='ignore')
        content = f.read()
        contents.append(content)
    return contents


if __name__ == '__main__':
    listFileSummary = glob.glob("outputs/hyp/cnn_dailymail/RNN_RNN_9F/SRSF_RNN_RNN_V4_2019_11_13_11_20_45_2_seed_1_10st/*")
    listFileSummary.sort()
    listFilereferences = glob.glob("outputs/ref_lables_cnn_dailymail/*")
    listFilereferences.sort()
    print(len(listFileSummary))
    print(len(listFilereferences))
    summary = readFile(listFileSummary)
    references = readFile(listFilereferences)
    sentence_of_system = []
    for index in range(len(summary)):
        sentences = summary[index].split("\n")
        sentence_of_system.append(sentences)

    sentence_of_ref = []
    for index in range(len(references)):
        sentences = references[index].split("\n")
        sentence_of_ref.append(sentences)

    list_all_true_candidates = []
    for index in range(len(sentence_of_system)):
        list_true_candidates = []
        for candidates_selected in sentence_of_system[index]:
            for refer in sentence_of_ref[index]:
                if (candidates_selected == refer):
                    list_true_candidates.append(candidates_selected)
        list_all_true_candidates.append(list_true_candidates)

    p = []
    r = []
    f1 = []
    for i in range(len(list_all_true_candidates)):
        p0 = len(list_all_true_candidates[i]) / len(sentence_of_system[i])
        p.append(p0)
        r0 = len(list_all_true_candidates[i]) / len(sentence_of_ref[i])
        r.append(r0)
        if (p0 + r0) == 0:
            f1.append(0)
        else:
            f1.append(2 * p0 * r0 / (p0 + r0))
    f_score = (2 * np.mean(r) * np.mean(p)) / (np.mean(r) + np.mean(p))

    print(f_score)
    print(np.mean(r))
    print(np.mean(p))
