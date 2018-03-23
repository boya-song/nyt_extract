# 654789 pairs
import xml.etree.ElementTree as etree
import os
import sys
import re

NUMBER = re.compile(r"\b[-+]?\d*\.\d+|\d+\b")

def concat(item):
    p = item.findall('./p')
    # ~85 articles is like this: <p></p>
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
        if (f.endswith("tgz")):
            continue
        try:
            tree = etree.parse(f)
            root = tree.getroot()
            metas = root.findall('./head/meta')
            for item in metas:
                if item.attrib['name'] == 'online_sections':
                    if item.attrib['content'] == 'Opinion':
                        abstract_item = root.findall('./body/body.head/abstract')
                        full_text = root.findall('./body/body.content/block[@class="full_text"]')
                        hd_online = root.findall('./body[1]/body.head/hedline/hl2')
                        hd = root.findall('./body[1]/body.head/hedline/hl1')
                        if len(abstract_item) and len(full_text)> 0:
                            abstract = concat(abstract_item[0])
                            text = concat(full_text[0])
                            if len(hd_online) > 0:
                                text = hd_online[0].text + '. ' + text
                            elif len(hd) > 0:
                                text = hd[0].text + '. ' + text
                            else:
                                print("ERROR: It doesn't have a headline.")
                            fname = f.split('/')[-1].split('.')[0] + '.txt'
                            # preprocess(abstract)
                            # preprocess(article)
                            opinions.append(f)
                            with open(os.path.join(parsed_dir, fname), 'w') as file:
                                file.write(abstract + '\n\n\n' + text)
                    break
        except:
            e = sys.exc_info()[0]
            print("ERROR: %s" % e)


nyt_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_corpus/data'
parsed_dir = '/Users/Boya/Desktop/Courses/MATERIAL/nyt_parsed_opinion'

if os.path.exists(parsed_dir):
    os.system('rm -rf ../nyt_parsed_opinion')

os.mkdir(parsed_dir)
opinions = []
parse(nyt_dir)
with open('Opinions.txt', 'w') as f:
    f.write('\n'.join(opinions))
