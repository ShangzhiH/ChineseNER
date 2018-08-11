# encoding=utf-8
import sys

import tensorflow as tf


__all__ = ['entity_metric_collect']


def entity_metric_collect(real_tag_seqs, predict_tag_seqs, lengths, metrics):
    """"""
    try:
        assert len(real_tag_seqs) == len(predict_tag_seqs) == len(lengths)
    except AssertionError:
        tf.logging.error("Error: predict tag seq num doesn't equal real tag seq!")
        sys.exit(0)

    for real_tag_seq, predict_tag_seq, length in zip(real_tag_seqs, predict_tag_seqs, lengths):
        metrics = _entity_count(real_tag_seq, predict_tag_seq, length, metrics)
    return metrics


def _entity_count(real_tag_seq, predict_tag_seq, length, count):
    real_pos, real_type = _extract_entity_pos_and_type(real_tag_seq[0])
    predict_pos, predict_type = _extract_entity_pos_and_type(predict_tag_seq[0])

    last_real_entity_item_len = 1
    last_real_entity_type = real_type

    if predict_pos != "I":
        last_predict_entity_item_len = 1
        last_predict_entity_type = predict_type
    else:
        last_predict_entity_item_len = 0
        last_predict_entity_type = ""

    for real_tag, predict_tag in zip(real_tag_seq[1:length] + ["O"], predict_tag_seq[1:length] + ["O"]):
        real_pos, real_type = _extract_entity_pos_and_type(real_tag)
        predict_pos, predict_type = _extract_entity_pos_and_type(predict_tag)

        update_real = _update_entity_check(real_pos, real_type, last_real_entity_type)
        update_predict = _update_entity_check(predict_pos, predict_type, last_predict_entity_type)

        if update_real and update_predict:
            count = _entity_type_exist_check(last_real_entity_type, count)
            count = _entity_type_exist_check(last_predict_entity_type, count)
            if last_real_entity_item_len == last_predict_entity_item_len and last_real_entity_type == last_predict_entity_type:
                count[last_real_entity_type]["real"] += 1
                count[last_real_entity_type]["predict"] += 1
                count[last_real_entity_type]["correct"] += 1
            else:
                count[last_real_entity_type]["real"] += 1
                if last_predict_entity_item_len != 0:
                    count[last_predict_entity_type]["predict"] += 1
            last_real_entity_item_len = 1
            last_real_entity_type = real_type

            if predict_pos == "I":
                last_predict_entity_item_len = 0
                last_predict_entity_type = ""
            else:
                last_predict_entity_item_len = 1
                last_predict_entity_type = predict_type

        elif update_real and not update_predict:
            count = _entity_type_exist_check(last_real_entity_type, count)
            count[last_real_entity_type]["real"] += 1

            last_real_entity_item_len = 1
            last_real_entity_type = real_type

            last_predict_entity_item_len += 1

        elif not update_real and update_predict:
            count = _entity_type_exist_check(last_predict_entity_type, count)
            if last_predict_entity_item_len != 0:
                count[last_predict_entity_type]["predict"] += 1

            last_real_entity_item_len += 1

            if predict_pos == "I":
                last_predict_entity_item_len = 0
                last_predict_entity_type = ""
            else:
                last_predict_entity_item_len += 1
                last_predict_entity_type = predict_type

        else:
            last_real_entity_item_len += 1
            last_predict_entity_item_len += 1
    return count


def _update_entity_check(cur_pos, cur_type, last_type):
    """
    check if previous entity reaches the end
    when current tag
    - is O
    - begins with B
    - different entity type
    means previous tag is the end of entity
    """
    if cur_pos == "O":
        return True
    elif cur_pos == "B":
        return True
    elif cur_type != last_type:
        return True
    else:
        return False


def _extract_entity_pos_and_type(tag):
    """
    split tag in position(B, I, O etc.) and type(ORG, PER, O etc.)
    """
    if tag == "O":
        return "O", "O"
    else:
        return tag.split("-")


def _entity_type_exist_check(tag, info):
    if not tag:
        return info

    if tag not in info:
        info[tag] = {"real": 0, "predict": 0, "correct": 0}
    return info

if __name__ == "__main__":
    metrics_dict = {"ORG": {'real': 2, 'predict': 2, 'correct': 1}}
    real_seqs = [["O", "O", "B-ORG", "I-ORG", "O", "B-PER", "I-PER", "B-LOC", "B-LOC"],
                 ["B-ORG", "B-LOC", "I-LOC", "O", "O", "O", "O", "B-PER"]]
    predict_seqs = [["B-PER", "O", "B-ORG", "I-PER", "O", "B-PER", "I-PER", "B-LOC", "O"],
                    ["B-ORG", "B-LOC", "I-LOC", "O", "O", "O", "O", "B-PER"]]
    print(entity_metric_collect(real_seqs, predict_seqs, metrics_dict))








