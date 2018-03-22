import sys
import os
import hashlib
import struct
import subprocess
import collections
import re
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote,
              dm_double_close_quote, ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PATTERN = re.compile(r'\A\s\b(photo|graph|chart|map|table|drawing)\b')
NUMBER = re.compile(r"\b[-+]?\d*\.\d+|\d+\b")

nyt_tokenized_dir = "nyt_tokenized"
finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = 'finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(
            chunks_dir, '%s_%03d.bin' % (set_name, chunk))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack(
                    '%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)

def preprocess(line):
    line = line.replace('-LRB- S -RRB-'.lower(), '')
    line = line.replace('-LRB- M -RRB-'.lower(), '')
    line = PATTERN.sub('', line).strip()
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

    abstract_lines = abstract.split(';')
    article_lines = article.split('\n')

    abstract_lines = [preprocess(l.lower()) for l in abstract_lines]
    abstract_lines = [l for l in abstract_lines if l is not None]

    article_lines = [l.lower() for l in article_lines]

    # Make article into a single string
    article = ' '.join(article_lines)

    # Make abstract into a signle string, putting <s> and </s> tags around the
    # sentences
    abstract = ' '.join(
        ["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_lines])

    return article, abstract


def write_to_bin(stories, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

    story_fnames = stories
    num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for idx, s in enumerate(story_fnames):
            if idx % 1000 == 0:
                print("Writing story %i of %i; %.2f percent done" %
                      (idx, num_stories, float(idx) * 100.0 / float(num_stories)))

            # Get the strings to write to .bin file
            article, abstract = get_art_abs(os.path.join(nyt_tokenized_dir, s))

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature[
                'article'].bytes_list.value.extend([article.encode()])
            tf_example.features.feature[
                'abstract'].bytes_list.value.extend([abstract.encode()])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

def replace_numbers(dir):
    files = os.listdir(dir)
    for f in files:
        with open(os.path.join(dir, f), 'r') as file:
            abstract, article = file.read().split('\n\n\n')
        abstract = NUMBER.sub('0', abstract)
        article = NUMBER.sub('0', article)
        with open(os.path.join(dir, f), 'w') as file:
            file.write(abstract + '\n\n\n' + article)

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
    # tokenize_stories(nyt_dir, nyt_tokenized_dir)

    stories = os.listdir(nyt_tokenized_dir)
    stories.sort()
    num_of_stories = len(stories)
    num_of_train = int(num_of_stories * 0.9)
    num_of_val = int(num_of_stories * 0.95)

    # Read the tokenized stories, do a little postprocessing then write to bin
    # files
    write_to_bin(stories[:num_of_train], os.path.join(finished_files_dir, "train.bin"), makevocab=True) # 589309
    write_to_bin(stories[num_of_train:num_of_val], os.path.join(finished_files_dir, "val.bin")) # 32739
    write_to_bin(stories[num_of_val:], os.path.join(
        finished_files_dir, "test.bin")) # 32740

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into
    # smaller chunks, each containing e.g. 1000 examples, and saves them in
    # finished_files/chunks
    chunk_all()
