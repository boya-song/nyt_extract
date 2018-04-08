import sys
import os
import hashlib
import struct
import subprocess
import collections
import re

PATTERN1 = re.compile(r'\s*\b(photo|graph|chart|map|table|drawing)s*\b\s*$')
PATTERN2 = re.compile(r';\s*\b(photo|graph|chart|map|table|drawing)s*\b\s*;')

nyt_tokenized_dir = "nyt_tokenized"
finished_files_dir = "finished_files"

def preprocess(line):
    line = line.replace('-LRB- S -RRB-'.lower(), '')
    line = line.replace('-LRB- M -RRB-'.lower(), '')
    line = PATTERN1.sub('', line).strip()
    line = PATTERN2.sub('', line).strip()
    if len(line) > 0:
        if not line.endswith('.'):
            line += ' .'
        return line
    else:
        return None


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." %
          (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s),
                                    os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer',
               '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." %
          (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of
    # files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" %
          (stories_dir, tokenized_stories_dir))


def get_art_abs(story_file):
    with open(story_file, 'r') as f:
        abstract, article = f.read().split('\n\n\n')

    abstract_lines = abstract.strip().split(';')
    article_lines = article.strip().split('\n')

    abstract_lines = [preprocess(l.lower()) for l in abstract_lines]
    abstract_lines = [l for l in abstract_lines if l is not None]

    article_lines = [l.lower() for l in article_lines]

    # Make article into a single string
    article = ' '.join(article_lines)
    abstract = ' '.join(abstract_lines)

    return article, abstract


def write_to_file(stories, out_src, out_tgt):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

    story_fnames = stories
    num_stories = len(story_fnames)
    articles = []
    abstracts = []

    for idx, s in enumerate(story_fnames):
        if idx % 1000 == 0:
            print("Writing story %i of %i; %.2f percent done" %
                  (idx, num_stories, float(idx) * 100.0 / float(num_stories)))
        # Get the strings to write to .bin file
        article, abstract = get_art_abs(os.path.join(nyt_tokenized_dir, s))
        articles.append(article)
        abstracts.append(abstract)

    with open(out_src, 'w') as f:
        f.write('\n'.join(articles))

    with open(out_tgt, 'w') as f:
        f.write('\n'.join(abstracts))


    print("Finished writing file %s\n" % out_file)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("USAGE: python make_datafiles.py <nyt_dir>")
        sys.exit()
    nyt_dir = sys.argv[1]

    # Create some new directories
    if not os.path.exists(nyt_tokenized_dir):
        os.makedirs(nyt_tokenized_dir)
    if not os.path.exists(finished_files_dir):
        os.makedirs(finished_files_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized
    # stories directories
    # replace_numbers(nyt_dir)
    tokenize_stories(nyt_dir, nyt_tokenized_dir)

    stories = os.listdir(nyt_tokenized_dir)
    stories.sort()
    num_of_stories = len(stories)
    num_of_train = int(num_of_stories * 0.9)
    num_of_val = int(num_of_stories * 0.95)

    # Read the tokenized stories, do a little postprocessing then write to bin
    # files
    write_to_file(stories[:num_of_train], os.path.join(finished_files_dir, "train.txt.src", os.path.join(finished_files_dir, "train.txt.tgt")) # 589309
    write_to_file(stories[num_of_train:num_of_val], os.path.join(finished_files_dir, "valid.txt.src"), os.path.join(finished_files_dir, "valid.txt.tgt")) # 32739
    write_to_file(stories[num_of_val:], os.path.join(
        finished_files_dir, "test.txt.src"), os.path.join(
        finished_files_dir, "test.txt.tgt")) # 32740
