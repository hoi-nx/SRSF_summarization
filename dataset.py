#!/usr/bin/env python3
def worker(files):
    examples = []
    parts = open(files,encoding='latin-1').read().split('\n\n')
    entities = { line.strip().split(':')[0]:line.strip().split(':')[1].lower() for line in parts[-1].split('\n')}
    sents,labels,summaries = [],[],[]
    # content
    for line in parts[1].strip().split('\n'):
        content, label = line.split('\t\t\t')
        tokens = content.strip().split()
        for i,token in enumerate(tokens):
            if token in entities:
                tokens[i] = entities[token]
        label = '1' if label == '1' else '0'
        sents.append(' '.join(tokens))
        labels.append(label)
    # summary
    for line in parts[2].strip().split('\n'):
        print(line)
        tokens = line.strip().split()
        for i, token in enumerate(tokens):
            if token in entities:
                tokens[i] = entities[token]
        line = ' '.join(tokens).replace('*','')
        summaries.append(line)
    ex = {'doc':'\n'.join(sents),'labels':'\n'.join(labels),'summaries':'\n'.join(summaries)}
    examples.append(ex)
    print(examples)
    return examples

if __name__ == '__main__':
   worker("0a0a7140b649fb724b60086c3f914c16f2a9625e.summary")
