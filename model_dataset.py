# encoding=utf-8
import sys
import pickle

import tensorflow as tf

from utils import unicode_open

__all__ = ["DatasetMaker"]


# full_to_half, replace_html.etc operations were done during preprocessing process to avoid redundant calculation
def _generator_maker(file_path, infer=False):
    """
    :rtype: generator
    """
    def _generator():
        for line in unicode_open(file_path):
            tokens = line.strip().split("\t")
            if not infer:
                if len(tokens) != 2:
                    continue
                sentence, tag_seq = tokens
                chars = sentence.strip().split("/")
                tags = tag_seq.strip().split("/")
                try:
                    assert len(chars) == len(tags)
                except AssertionError:
                    tf.logging.error("Chars num doesn't equal tags num")
                    sys.exit(0)

                yield chars, tags
            else:
                if len(tokens) != 1:
                    continue
                chars = tokens.strip().split("/")
                yield chars
    return _generator


class DatasetMaker(object):

    char_to_id = {u"<PAD>": 0, u"<UNK>": 1, u"<START>": 2, u"<END>": 3}
    id_to_char = {0: u"<PAD>", 1: u"<UNK>", 2: u"<START>", 3: u"<END>"}
    tag_to_id = {u"O": 0}
    id_to_tag = {0: u"O"}
    mapping_dict_ready = False

    char_mapping_tensor = None
    tag_mapping_tensor = None
    mapping_tensor_ready = False

    @classmethod
    def tag_ids_to_tags(cls, tag_ids):
        if not cls.mapping_dict_ready:
            tf.logging.error("Mapping dict isn't initialized!")
            sys.exit(0)

        tags = []
        for tag_id in tag_ids:
            tag_seq = [cls.id_to_tag[_] for _ in tag_id]
            tags.append(tag_seq)
        return tags

    @classmethod
    def generate_mapping(cls, file_path):
        char_freq = {}
        for char_list, tag_list in _generator_maker(file_path):
            for char in char_list:
                char_freq[char] = char_freq.get(char, 0) + 1
            for tag in tag_list:
                cls.tag_to_id[tag], cls.id_to_tag[len(cls.id_to_tag)] = len(cls.tag_to_id), tag

        sorted_items = sorted(char_freq.items(), key=lambda d: d[1], reverse=True)
        for key, value in sorted_items:
            if key not in cls.char_to_id:
                cls.char_to_id[key], cls.id_to_char[len(cls.id_to_char)] = len(cls.char_to_id), key

        cls.mapping_dict_ready = True
        tf.logging.info("Generated mapping dictionary with {} different chars and {} different tags!".format(len(cls.char_to_id), len(cls.tag_to_id)))

    @classmethod
    def save_mapping(cls, mapfile_path, vocabfile_path):
        if not cls.mapping_dict_ready:
            tf.logging.error("Error: mapping dict isn't initialized!")
            sys.exit(0)

        with tf.gfile.GFile(mapfile_path, "wb") as f_w:
            pickle.dump([cls.char_to_id, cls.id_to_char, cls.tag_to_id, cls.id_to_tag], f_w)
        tf.logging.info("Saved mapping dictionary in file {}".format(mapfile_path))
        with tf.gfile.GFile(vocabfile_path, "w") as f:
            f.write(u"\n".join(cls.char_to_id.keys()))
        tf.logging.info("Saved readable vocabulary in file {}".format(vocabfile_path))

    @classmethod
    def load_mapping(cls, mapfile_path):
        with tf.gfile.GFile(mapfile_path, "rb") as f:
            cls.char_to_id, cls.id_to_char, cls.tag_to_id, cls.id_to_tag = pickle.load(f)

        cls.mapping_dict_ready = True
        tf.logging.info("Loaded mapping dictionary from file {} with {} different chars and {} different tags!".format(mapfile_path, len(cls.char_to_id), len(cls.tag_to_id)))

    @classmethod
    def make_mapping_table_tensor(cls):
        if not cls.mapping_dict_ready:
            tf.logging.error("Error: mapping dict isn't initialized!")
            sys.exit(0)

        cls.char_mapping_tensor = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorIntializer(cls.char_to_id.keys(), cls.char_to_id.values()),
            cls.char_to_id.get(u"<UNK>")
        )
        cls.tag_mapping_tensor = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorIntializer(cls.tag_to_id.keys(), cls.tag_to_id.values()),
            cls.tag_to_id.get(u"O")
        )

        cls.mapping_tensor_ready = True
        tf.logging.info("Created mapping table tensor from exist mapping dict!")

    @classmethod
    def make_dataset(cls, file_path, batch_size, task_type, num_shards, worker_index):
        if not cls.mapping_tensor_ready:
            tf.logging.error("Error: mapping tensor isn't initialized!")
            sys.exit(0)

        if task_type == "infer":
            dataset = tf.data.Dataset.from_generator(_generator_maker(file_path, True), tf.string, None)
            dataset = dataset.shard(num_shards, worker_index)
            dataset = dataset.map(lambda chars: (cls.char_mapping_tensor.lookup(chars)))
            dataset = dataset.padded_batch(batch_size, padded_shapes=None)
        else:
            dataset = tf.data.Dataset.from_generator(_generator_maker(file_path, False), (tf.string, tf.string), (None, None))
            dataset = dataset.shard(num_shards, worker_index)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.map(lambda chars, tags: (cls.char_mapping_tensor.lookup(chars), cls.tag_mapping_tensor.lookup(tags)))
            # train
            if task_type == "train":
                dataset = dataset.padded_batch(batch_size, padded_shapes=(None, None)).repeat()
            # eval
            elif task_type == "eval":
                dataset = dataset.padded_batch(batch_size, padded_shapes=(None, None))
        return dataset








