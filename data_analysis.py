# more common imports
import pandas as pd
import numpy as np
# visualization imports
# from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# import matplotlib.image as mpimg
# import base64
# import io
#%matplotlib inline
sns.set()


def readFile(filePath):
    contents = []
    for i in range(len(filePath)):
        f = open(filePath[i], 'r', encoding="utf8", errors='ignore')
        content = f.read()
        contents.append(content)
    return contents


if __name__ == '__main__':
    listFileSentences_Original = glob.glob("outputs/gold_summary_val_cnn_dailymail/*")
    listFileSentences_Original.sort()
    senten_original = readFile(listFileSentences_Original)
    sentences_original = []
    sent = []
    print(senten_original[14])
    for senten in senten_original:
        sen = senten.split('\n')
        sentences_original.append(sen)
    sentences_of_document_lengths_origin = np.array(list(map(len, sentences_original)))
    print(len(sentences_of_document_lengths_origin))
    #print(sentences_of_document_lengths_origin)
    for i in range(len(sentences_of_document_lengths_origin)):
        if(sentences_of_document_lengths_origin[i] == 1):
            print(i)

    print("The average number of sentens in a document is: {}.".format(np.mean(sentences_of_document_lengths_origin)))
    print("The minimum number of sentens in a document is: {}.".format(min(sentences_of_document_lengths_origin)))
    print("The maximum number of sentens in a document is: {}.".format(max(sentences_of_document_lengths_origin)))
    print("There are {} documents with over 10350 document.".format(sum(sentences_of_document_lengths_origin >= 4)))
    print("There are {} documents with over 10350 document.".format(sum(sentences_of_document_lengths_origin < 4)))
    print("There are {} documents with over 10350 document.".format(sum(sentences_of_document_lengths_origin <= 1)))

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.set_title("Distribution of number of sentences(origin_train_cnn_dailymail)", fontsize=16)
    ax.set_xlabel("Number of sentences")
    sns.distplot(sentences_of_document_lengths_origin, bins=50, ax=ax);
    plt.show()

