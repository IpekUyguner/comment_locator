import collections
import csv

import gensim
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

EMPTY_LINE = "EMPTY"
COMMENT_ONLY_LINE = "COMMENTEMPTY"

UNKNOWN_WORD = "-unk-"
special_symbols = [UNKNOWN_WORD]
UNKNOWN = 0
MIN_WORD_FREQ = 10  # Minimum frequency for a word to be added to vocabulary


def read_w2v_models(code_embeddibgs):
    """
    Reads pretrained w2v embeddings.
    """

    print("Loading pre-trained embeddings\n")
    print("\t sources: " + code_embeddibgs)
    code_word_2_vector_model = gensim.models.KeyedVectors.load_word2vec_format(
        code_embeddibgs, binary=True
    )
    code_word_2_vectors_dim = code_word_2_vector_model.vector_size
    print("..done.")
    word_2_vectors_models = [code_word_2_vector_model]
    word_2_vectors_dims = [code_word_2_vectors_dim]
    return word_2_vectors_models, word_2_vectors_dims


def read_comment_location_file(comment_files, max_sentence_length, skip_empty_stmt):
    """
    This method reads a comment file. It creates the line of code sequences for each file. The return value is a list of file sequences.
    Parameters
    ----------
    comment_files: string
         a file containing comment location information.
    max_sentence_length:
         maximum of code token that can be in given sentence
    skip_empty_stmt: boolean
         ignore lines where the stmt text matches exactly the empty line marker
    Return
    ------
    file_seqs: list of file_seq.
         Each file_seq is a list of line_of_codes
    all_words: counter
         code and background tokens to their counts across the dataset
    """

    print("Reading comment locations...")
    current_file = ""
    current_file_sequence = file_sequence()

    all_words = collections.Counter()
    file_sequences = []

    lines = open(comment_files, "r")
    for line in lines:
        file_id, line_count, block_body, label, _, _, code, _, _ = line.strip().split(
            "\t"
        )
        # blk_bds for hierarchical LSTM model   # 1 ==begin, 2==  mid  3 == end of block, -1 == no system of blocks

        # skip lines which are blank because they had a comment
        if code == COMMENT_ONLY_LINE:
            continue

        # skip all empty lines if the marker is set
        if code == EMPTY_LINE and skip_empty_stmt:
            continue

        # clean and truncate the line of code
        code_toks = code.split()[0 : min(len(code.split()), max_sentence_length)]
        for w in code_toks:
            all_words[w] += 1
        code = " ".join(code_toks).strip()

        cloc = line_of_code(file_id, line_count, block_body, code, float(label))
        # store the current files contents before moving to next file
        if file_id != current_file:
            if current_file_sequence.num_line_of_code() > 0:
                file_sequences.append(current_file_sequence)
                current_file_sequence = file_sequence()
            current_file = file_id

        current_file_sequence.add_line_of_code(cloc)

    if current_file_sequence.num_line_of_code() > 0:
        file_sequences.append(current_file_sequence)
    lines.close()
    print("\tThere were " + str(len(file_sequences)) + " file sequences")
    return file_sequences, all_words


def get_restricted_vocabulary(word_counts, vocab_size, min_freq, add_special):
    """ Create vocabulary of a certain max size """
    non_special_size = min(vocab_size, len(word_counts))
    if add_special and non_special_size == vocab_size:
        non_special_size -= len(special_symbols)
    word_counts_sorted = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
    words = [
        k for (k, v) in word_counts_sorted if k not in special_symbols and v >= min_freq
    ][0:non_special_size]

    if add_special:
        for symbol in special_symbols:
            words.insert(0, symbol)
    return words


def assign_vocab_ids(word_list):
    """ Given a word list, assign ids to it and return the id dictionaries """
    word_ids = [x for x in range(len(word_list))]
    word_to_id = dict(zip(word_list, word_ids))
    id_to_word = dict(zip(word_ids, word_list))
    return word_to_id, id_to_word


def code_each_word(word_list, vocab_to_id, ignore_unknown, ignore_special):
    """Given word list, return list of word ids"""
    result = []
    for word in word_list:
        if ignore_special and word in special_symbols:
            continue
        if word in vocab_to_id:
            result.append(vocab_to_id[word])
        elif not ignore_unknown:
            result.append(vocab_to_id[UNKNOWN_WORD])
    return result


def code_text(text, vocab_to_id, ignore_unknown, ignore_special):
    """ Convert a text into list of ids """
    return code_each_word(text.split(), vocab_to_id, ignore_unknown, ignore_special)


def get_avg_embedding(wids, id_to_vocab, w2v_model, word_2_vector_dimensions):
    """ Convert line into embedding but getting average of each word """
    sum_embeddings = np.zeros(word_2_vector_dimensions, dtype=np.float32)
    if w2v_model == None:
        return sum_embeddings
    found_word = 0
    for id in wids:
        token = id_to_vocab[id]
        if token != UNKNOWN_WORD and token in w2v_model.vocab:
            sum_embeddings += w2v_model.wv[token]
            found_word += 1
    if found_word > 0:
        return sum_embeddings / found_word
    return sum_embeddings


def get_embedding_vectors(w2v_model, word_2_vector_dimensions, vocab_to_id):
    """
    Get all embeddings in a [nwords * dimsize] matrix. If word not
    found, have random embedding.
    """

    embeddings = np.random.uniform(
        -0.001, 0.001, size=(len(vocab_to_id), word_2_vector_dimensions)
    )
    for word, index in vocab_to_id.items():
        if word in w2v_model.vocab:
            embeddings[index] = w2v_model.wv[word]
    return embeddings


class line_of_code(object):
    """A class to represent line code."""

    def __init__(self, file_id, line_count, block_body, code, label):
        """
        Parameters
        ----------
        file_id : int
        line_count : int
        block_body: int loc boundary (-1 for NA, 1 for start, 2 mid and 3 end of a block)
        code : string rep of source code
        label : int 0/1, whether a label appears before this code loc
        """
        self.file_id = int(file_id)
        self.line_count = int(line_count)
        self.block_body = int(block_body)
        self.code = code
        self.label = label

    def __str__(self):
        return (
            str(self.file_id)
            + "\t"
            + str(self.line_count)
            + "\t"
            + str(self.block_body)
            + "\t"
            + str(self.label)
            + "\t"
            + str(self.code)
        )


class line_of_code_features(object):
    """ A class to represents features and embeddings for a line of code """

    def __init__(self, coded, lc_embedded):
        self.coded = coded
        self.embedded = lc_embedded


class file_sequence(object):
    """ A file has a list of line of codes."""

    def __init__(self):
        self.line_codes = []

    def add_line_of_code(self, loc):
        self.line_codes.append(loc)

    def num_line_of_code(self):
        return len(self.line_codes)

    def get_line_of_code(self, bid):
        return self.line_codes[bid]

    def get_all_line_codes(self):
        return self.line_codes

    def __str__(self):
        return str([cb.__str__() for cb in self.line_codes])


class BlockDataset(Dataset):
    """Represents the dataset:
    Each file consists of code blocks
    Each code block consists of code lines
    Attributes:
        file_sequences: Sequence of files
        max_blocks: Max number of code blocks within any file
        max_length_per_block: Max length of code block
        max_length_per_sentence: Max length of code tokens in a code line
        word_2_vector_models: Word to vectors dictionary
        word_2_vector_dimensions: Word to vectors dimensions (TO DO: can be removed)
        word_vectors: Word vectors
        all_word_w2i: Word to vectors dictionary
        all_word_i2w: Index to word dictionary
    """

    def __init__(
        self,
        file_sequences,
        max_blocks,
        max_length_per_block,
        max_length_per_sentence,
        word_2_vector_models,
        word_2_vector_dimensions,
        word_vectors,
        all_wordv,
        all_word_w2i,
        all_word_i2w,
    ):
        cloc_id_to_features = {}
        ignore_unknown, ignore_special = True, True
        self.cloc_id_to_features = cloc_id_to_features
        self.word_vectors = word_vectors

        self.all_wordv = all_wordv
        self.all_word_w2i = all_word_w2i
        self.all_word_i2w = all_word_i2w
        self.all_word_vocab_size = len(self.all_word_w2i)

        self.lc_embed_size = word_2_vector_dimensions[0]
        self.word_2_vector_models = word_2_vector_models
        self.word_2_vector_dimensions = word_2_vector_dimensions

        for file_num in range(len(file_sequences)):
            file_seq = file_sequences[file_num]

            for i in range(file_seq.num_line_of_code()):
                lc = file_seq.get_line_of_code(i)
                lc_code = lc.code

                lc_coded = code_text(
                    lc_code, all_word_w2i, ignore_unknown, ignore_special
                )
                lc_embedded = get_avg_embedding(
                    lc_coded,
                    all_word_i2w,
                    word_2_vector_models[0],
                    word_2_vector_dimensions[0],
                )

                line_code_features = line_of_code_features(lc_coded, lc_embedded)
                cloc_id_to_features[
                    str(lc.file_id) + "#" + str(lc.line_count)
                ] = line_code_features

        data_by_seq_len = []
        cur_seq = []

        count_true_num_blocks = 0
        Lc_and_Features = collections.namedtuple("Lc_and_Features", ["lc", "lc_feat"])

        for file_seq in file_sequences:
            if len(data_by_seq_len) == 0 or data_by_seq_len[-1] != []:
                data_by_seq_len.append([])

            for i in range(file_seq.num_line_of_code()):
                lc = file_seq.get_line_of_code(i)
                line_code_features = self.cloc_id_to_features[
                    str(lc.file_id) + "#" + str(lc.line_count)
                ]
                if lc.block_body == 1:
                    if len(cur_seq) > 0:
                        data_by_seq_len[-1].append(cur_seq)
                        cur_seq = []
                    if len(data_by_seq_len[-1]) >= max_blocks:
                        # if exceed max_blocks for a file, creates new file for the rest of the file.
                        data_by_seq_len.append([])
                    count_true_num_blocks += 1
                    cur_seq.append(Lc_and_Features(lc, line_code_features))

                elif lc.block_body == 2 or lc.block_body == 3:
                    if len(cur_seq) < max_length_per_block:
                        cur_seq.append(Lc_and_Features(lc, line_code_features))

            if len(cur_seq) > 0:
                data_by_seq_len[-1].append(cur_seq)
                cur_seq = []

        total_sequences = len(data_by_seq_len)

        count_blocks = 0
        for i in range(len(data_by_seq_len)):
            for j in range(len(data_by_seq_len[i])):
                count_blocks += 1

        print("num_blocks read = " + str(count_blocks))
        assert count_true_num_blocks == count_blocks, (
            "Number of blks read into data %d does not match true number of blocks %d"
            % (count_blocks, count_true_num_blocks)
        )

        self.data = np.zeros(
            (
                total_sequences,
                max_blocks,
                max_length_per_block,
                max_length_per_sentence,
            ),
            dtype=np.int32,
        )
        self.data_weights = np.zeros(
            (
                total_sequences,
                max_blocks,
                max_length_per_block,
                max_length_per_sentence,
            ),
            dtype=np.float32,
        )
        self.targets = np.zeros((total_sequences, max_blocks), dtype=np.int32)
        self.target_weights = np.zeros((total_sequences, max_blocks), dtype=np.float32)

        for i in range(total_sequences):  # for each file
            for k in range(max_blocks):  # for each block
                for j in range(max_length_per_block):  # for each sentence
                    if (
                        i >= total_sequences
                        or k >= len(data_by_seq_len[i])
                        or j >= len(data_by_seq_len[i][k])
                    ):
                        wid_datum = np.array([self.all_word_w2i[UNKNOWN_WORD]])
                    else:
                        wid_datum = np.array(data_by_seq_len[i][k][j].lc_feat.coded)
                        self.targets[i][k] = data_by_seq_len[i][k][0].lc.label
                        self.target_weights[i][k] = 1.0

                    self.data[i][k][j][0 : wid_datum.shape[0]] = wid_datum
                    self.data_weights[i][k][j][0 : wid_datum.shape[0]] = np.ones(
                        wid_datum.shape[0]
                    )

        self.pretrained_vectors = get_embedding_vectors(
            self.word_2_vector_models[0],
            self.word_2_vector_dimensions[0],
            self.all_word_w2i,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return (
            self.data[item],
            self.data_weights[item],
            self.targets[item],
            self.target_weights[item],
        )


def create_vocabularies(all_words, word_vsize):
    """
    Based on word_vsize, creates the dictionary for the model.
    """
    word_vectors = get_restricted_vocabulary(all_words, word_vsize, MIN_WORD_FREQ, True)
    all_wordv = []
    all_wordv.extend(word_vectors)
    all_word_w2i, all_word_i2w = assign_vocab_ids(all_wordv)
    word_to_indexes_writer = csv.writer(open("word_to_indexes.csv", "w"), delimiter=" ")
    for key, val in all_word_w2i.items():
        # write every key and value to file
        word_to_indexes_writer.writerow([key, val])
    return (word_vectors, all_wordv, all_word_w2i, all_word_i2w)


class DataModule(pl.LightningDataModule):
    """
    Pytorch datamodule to wrap dataset.
    """

    def __init__(
        self,
        args,
        word_2_vector_models,
        word_2_vector_dimensions,
        word_vectors,
        all_word_vectors,
        all_word_w2i,
        all_word_i2w,
        file_sequence,
        file_sequence_val,
        file_sequence_test,
    ) -> None:
        super().__init__()
        self.args = args
        self.word_vectors = word_vectors
        self.train_batch_size = args.train_batch_size
        self.all_word_vectors = all_word_vectors
        self.all_word_w2i = all_word_w2i
        self.all_word_i2w = all_word_i2w
        self.data_train_path = args.data_train_path
        self.data_test_path = args.data_test_path
        self.data_validation_path = args.data_validation_path
        self.file_sequence = file_sequence
        self.file_sequence_val = file_sequence_val
        self.file_sequence_test = file_sequence_test
        self.code_embeddings = args.code_embeddings
        self.batch_size = args.batch_size
        skip_empty_line = False
        self.word_vsize = args.vocab_size
        self.train_dataset = BlockDataset(
            self.file_sequence,
            args.max_blocks,
            args.max_length_block,
            args.max_length_sentence,
            word_2_vector_models,
            word_2_vector_dimensions,
            word_vectors,
            self.all_word_vectors,
            self.all_word_w2i,
            self.all_word_i2w,
        )
        self.pretrainedvecs = self.train_dataset.pretrained_vectors
        self.validation_dataset = BlockDataset(
            self.file_sequence_val,
            args.max_blocks,
            args.max_length_block,
            args.max_length_sentence,
            word_2_vector_models,
            word_2_vector_dimensions,
            word_vectors,
            all_word_vectors,
            all_word_w2i,
            all_word_i2w,
        )
        self.test_dataset = BlockDataset(
            self.file_sequence_test,
            args.max_blocks,
            args.max_length_block,
            args.max_length_sentence,
            word_2_vector_models,
            word_2_vector_dimensions,
            word_vectors,
            all_word_vectors,
            all_word_w2i,
            all_word_i2w,
        )
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_data_loader = DataLoader(
            self.validation_dataset, batch_size=1, shuffle=False, drop_last=True,
        )
        self.test_data_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, drop_last=True,
        )

    def get_val(self):
        return self.validation_dataset

    def get_train(self):
        return self.train_dataset

    def train_dataloader(self) -> DataLoader:
        return self.train_data_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_data_loader

    def test_dataloader(self) -> DataLoader:
        return self.test_data_loader
