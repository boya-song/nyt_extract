import xml.etree.ElementTree as etree
import os
import sys


def concat(item):
    p = item.findall('./p')
    sentence = []
    for i in p:
        sentence.append(i.text)
    return '\n'.join(sentence)


def parse(dname):
    inodes = [os.path.join(dname, f) for f in os.listdir(dname)]
    files = [f for f in inodes if os.path.isfile(f)]
    dirs = [f for f in inodes if os.path.isdir(f)]
    print('Parse dir: ', dname)

    for d in dirs:
        parse(d)
    for f in files:
        print('Parse file: ', f)
        if (f.endswith("tgz")):
            continue
        try:
            tree = etree.parse(f)
            root = tree.getroot()
            abstract_item = root.findall('./body/body.head/abstract')
            full_text = root.findall('./body/body.content/block[@class="full_text"]')
            if len(abstract_item) > 0:
                abstract = concat(abstract_item[0])
                text = concat(full_text[0])
                fname = f.split('/')[-1].split('.')[0] + '.txt'
                with open(os.path.join(parsed_dir, fname), 'w') as file:
                    file.write('Abstract: ' + abstract + '\n\n\n' + text)
        except:
            print("ERROR: %s" % e)
            e = sys.exc_info()[0]
            error.append(f)
            error.append(e)


nyt_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_corpus/data'
error = []
parsed_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_parsed'

if os.path.exists(parsed_dir):
    os.system('rm -rf ../nyt_parsed')

os.mkdir(parsed_dir)
parse(nyt_dir)
with open(os.path.join(parsed_dir, 'error.txt'), 'w') as f:
    f.write('\n'.join(error))
