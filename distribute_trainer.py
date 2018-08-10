# encoding=utf-8
import os
import sys
import traceback
import time

import tensorflow as tf
import numpy as np

from model import TrainModel, EvalModel
from model_dataset import DatasetMaker
from data_utils import line_num_count

flags = tf.app.flags
flags.DEFINE_string("ps_hosts", "", "Comma-separated list of ps hostname:port pairs")
flags.DEFINE_string("worker_hosts", "", "Comma-separated list of worker hostname:port pairs")
flags.DEFINE_string("chief_hosts", "", "Comma-separated list of chief hostname:port pairs")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker' or 'chief'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_boolean("is_sync", True, "Whether use sync strategy to update parameter")

flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("clip", 5.0, "gradient clipper value")
flags.DEFINE_float("max_epoch", 1000, "the max number of epochs")
flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("check_step", 100, "Check loss every N steps")
flags.DEFINE_integer("eval_step", 2000, "Eval model every N steps")
flags.DEFINE_string("root_path", "", "project root path")
flags.DEFINE_string("log_dir", "log/", "log directory")
flags.DEFINE_string("train_data", "", "training data source")
flags.DEFINE_string("valid_data", "", "validation data source")
flags.DEFINE_string("test_data", "", "test data source")
flags.DEFINE_string("N_best_model", 10, "models of top N accuracy")

flags.DEFINE_integer("char_dim", 300, "char embedding dimension")
flags.DEFINE_string("rnn_type", "LSTM", "rnn cell type")
flags.DEFINE_integer("rnn_dim", 300, "rnn hidden dimension")
flags.DEFINE_integer("rnn_layer", 1, "rnn layer number")
flags.DEFINE_string("loss_type", "crf", "type of loss layer")
flags.DEFINE_float("dropout", 0.5, "dropout rate during training")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


class Trainer(object):
    def __init__(self):
        self.ps_hosts = FLAGS.ps_hosts.split(",")
        self.worker_hosts = FLAGS.worker_hosts.split(",")
        self.chief_hosts = FLAGS.chief_hosts.split(",")
        tf.logging.info("PS hosts are: {}".format(self.ps_hosts))
        tf.logging.info("Worker hosts are: {}".format(self.worker_hosts))
        tf.logging.info("Chief hosts are: {}".format(self.chief_hosts))

        self.root_path = FLAGS.root_path
        self.log_dir = os.path.join(self.root_path, FLAGS.log_dir)
        tf.logging.info("Project root path is: {}".format(self.root_path))
        tf.logging.info("Log directory is: {}".format(self.log_dir))

        self.train_data = FLAGS.train_data
        self.valid_data = FLAGS.valid_data
        self.test_data = FLAGS.test_data
        self.train_data_num = line_num_count(self.train_data)
        self.valid_data_num = line_num_count(self.valid_data)
        self.test_data_num = line_num_count(self.test_data)
        tf.logging.INFO("{} / {} / {} sentences in train /dev / test".format(self.train_data_num, self.valid_data_num, self.test_data_num))
        self.num_steps = int(FLAGS.max_epoch * self.train_data_num / len(self.worker_hosts))

        self.map_file = os.path.join(self.root_path, "map.pkl")
        self.vocabulary_file = os.path.join(self.root_path, "vocabulary.csv")

        self.job_name = FLAGS.job_name
        self.task_index = FLAGS.task_index
        self.cluster = tf.train.ClusterSpec(
            {"ps": self.ps_hosts, "worker": self.worker_hosts, "chief": self.chief_hosts}
        )
        self.server = tf.train.Server(self.cluster, job_name=self.job_name, task_index=self.task_index)
        self.is_chief = (self.job_name == "chief" and self.task_index == 0)
        self.worker_prefix = '/job:%s/task:%s' % (self.job_name, self.task_index)
        self.num_ps = self.cluster.num_tasks("ps")
        self.num_worker = self.cluster.num_tasks("worker")
        FLAGS.worker_num = self.num_worker

        self.global_step = 0
        self.check_step = FLAGS.check_step
        self.eval_step = FLAGS.eval_step
        self.train_summary_op = None
        self.eval_summary_op = None
        self.summary_writer = tf.summary.FileWriter(self.log_dir)
        self.topN = FLAGS.N_best_model
        self.model_performance = dict.fromkeys(range(self.topN), 0.0)
        self.worst_valid_model_index = 0
        self.best_test_accuracy = 0.0

    @staticmethod
    def _print_flags(flags_to_print):
        for k, v in flags_to_print:
            tf.logging.info("{}:\t{}".format(k.ljust(15), v))

    @staticmethod
    def _eval_performance(self, session, model, name, iter_init_op):
        tf.logging.info("Evaluate:{}".format(name))
        session.run(iter_init_op)
        tf.logging.info("Iterator is switched to {}".format(name))

        metric_dict = model.evaluate(session)
        all_correct = sum([v["correct"] for v in metric_dict.values()])
        all_predict = sum([v["predict"] for v in metric_dict.values()])
        accuracy = all_correct / all_predict
        tf.logging.info("Processed: {} entities; found: {} entities; correct: {}; accuracy {:.2f}%"
                        .format(all_predict, all_predict, all_correct, 100.0 * accuracy))

        for key, value in sorted(metric_dict.items(), key=lambda d: d[0]):
            precision = 0.0 if value["predict"] == 0 else 100.0 * value["correct"] / value["predict"]
            recall = 0.0 if value["real"] == 0 else 100.0 * value["correct"] / value["real"]
            f1 = 0.0 if precision + recall == 0 else 200.0 * precision * recall / (precision + recall)
            tf.logging.info("Processed: {} {} entities: found: {}; correct: {};".
                            format(value["real"], key, value["predict"], value["correct"]))
            tf.logging.info(" ------------------------- precision: {:.2f}%; recall: {:.2f}%; f1: {:.2f}%"
                            .format(precision, recall, f1))
        if name == "validation":
            if accuracy > self.model_performance[self.worst_validate_model_index]:
                tf.logging.info("New best validation accuracy:{:.2f}".format(100.0 * accuracy))
                self.model_performance[self.worst_validate_model_index] = accuracy
                model.saver.save(session, self.log_dir, self.worst_validate_model_index)
                model.saver.save(session, os.path.join(self.log_dir, "best.ckpt"))
                self.worst_validate_model_index = sorted(self.model_performance.items(), key=lambda d: d[0])[0][0]
        elif name == "test":
            if accuracy > self.best_test_accuracy:
                tf.logging.info("New best test accuracy:{:.2f}".format(100.0 * accuracy))
        return accuracy

    def _init_dataset_maker(self, load=False):
        if not load:
            DatasetMaker.generate_mapping(self.train_data)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        else:
            DatasetMaker.load_mapping(self.map_file)
            DatasetMaker.save_mapping(self.map_file, self.vocabulary_file)
        FLAGS.char_num = len(DatasetMaker.char_to_id)
        FLAGS.tag_num = len(DatasetMaker.tag_to_id)

    def _create_session(self):
        session_config = tf.ConfigProto(allow_soft_placement=True,
                                        device_filters=["/job:ps", "/job:{}/task:{}".format(self.job_name, self.task_index)],
                                        log_device_placement=True)
        session_config.gpu_options.allow_growth = True
        stop_hook = tf.train.StopAtStepHook(num_steps=self.num_steps)
        summary_saver_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=self.log_dir, summary_op=self.train_summary_op)

        self.session = tf.train.MonitoredTrainingSession(
            master=self.server.target,
            is_chief=self.is_chief,
            checkpoint_dir=None,
            scaffold=None,
            hooks=[stop_hook],
            chief_only_hooks=[summary_saver_hook],
            save_checkpoint_secs=None,
            save_summaries_secs=None,
            save_summaries_steps=None,
            config=session_config,
            stop_grace_period_secs=120
        )
        return self.session

    def _create_session_wrapper(self, retries=10):
        if retries == 0:
            tf.logging.error("Creating the session is out of times!")
            sys.exit(0)
        try:
            return self._create_session()
        except Exception as e:
            tf.logging.info(e)
            tf.logging.info("Retry creating session:{}" % retries)
            try:
                if self.session is not None:
                    self.session.close()
                else:
                    tf.logging.info("Close session: session is None!")
            except Exception as e:
                exc_info = traceback.format_exc(sys.exc_info())
                msg = "Creating session exception:{}\n{}".format(e, exc_info)
                tf.logging.warn(msg)
            return self._create_session_wrapper(retries - 1)

    def train(self):
        if self.job_name == "ps":
            with tf.device("/cpu:0"):
                self.server.join()
                return

        self._init_dataset_maker(False)
        train_init_op = None
        valid_init_op = None
        test_init_op = None
        with tf.device(tf.train.replica_device_setter(worker_device=self.worker_prefix, cluster=self.cluster)):
            self.global_step = tf.train.get_or_create_global_step()
            if self.job_name == "worker":
                train_dataset = DatasetMaker.make_dataset(self.train_data, FLAGS.batch_size,
                                                          "train", self.num_worker, self.task_index)
                tf.logging.info("The part {}/{} Training dataset is prepared!".format(self.task_index+1, self.num_worker))
                train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
                train_init_op = train_iter.make_initializer(train_dataset)

                train_model = TrainModel(train_iter, FLAGS, self.global_step)

            elif self.job_name == "chief":
                # build same train graph to synchronize model parameters
                train_dataset = DatasetMaker.make_dataset(self.train_data, FLAGS.batch_size,
                                                          "train", self.num_worker, self.task_index)
                train_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
                train_model = TrainModel(train_iter, FLAGS, self.global_step)
                self.train_summary_op = train_model.merge_train_summary_op

                # build test graph of same structure but different name scope
                # restore model from train checkpoint, and avoid its updating during validation
                eval_graph = tf.Graph()
                with eval_graph.as_default():
                    valid_dataset = DatasetMaker.make_dataset(self.valid_data, FLAGS.batch_size, "eval", 1, 0)
                    tf.logging.info("The part 1/1 Validation dataset is prepared!")
                    test_dataset = DatasetMaker.make_dataset(self.test_data, FLAGS.batch_size, "eval", 1, 0)
                    tf.logging.info("The part 1/1 Test dataset is prepared!")

                    eval_iter = tf.data.Iterator.from_structure(valid_dataset.output_types, valid_dataset.output_shapes)
                    valid_init_op = eval_iter.make_initializer(valid_dataset)
                    test_init_op = eval_iter.make_initializer(test_dataset)
                    eval_model = EvalModel(eval_iter, FLAGS, "eval_graph")

        with self._create_session_wrapper(retries=10) as sess:
            try:
                if self.job_name == "worker":
                    DatasetMaker.init_mapping_table_tensor(sess)
                    sess.run(train_init_op)

                    step = 0
                    while not sess.should_stop():
                        global_step_val, loss_value = train_model.train(sess)
                        if (step + 1) % self.check_step == 0:
                            epoch = (global_step_val * FLAGS.batch_size) // self.train_data_num
                            tf.logging.info("Job-{}:Worker-{}-----Epoch:{}-Local_Step/Global_Step:{}/{}:Loss is {:.2f}".
                                            format(self.job_name, self.task_index, epoch, step, global_step_val, loss_value))
                        step += 1
                elif self.job_name == "chief":
                    tf.logging.info("Created model with fresh parameters.")
                    self._print_flags(FLAGS)
                    sess.run(tf.global_variables_initializer())
                    DatasetMaker.init_mapping_table_tensor(sess)
                    # record top N model's performance
                    while True:
                        time.sleep(2)
                        global_step_val = sess.run(self.global_step)
                        if (global_step_val + 1) % self.eval_step == 0:
                            tf.logging.info("Evaluate Validation Dataset and Test Dataset in step: {}".format(global_step_val))
                            train_model.saver.save(sess, self.log_dir, latest_filename="temp", global_step=self.global_step)
                            ckpt = tf.train.get_checkpoint_state(self.log_dir, latest_filename="temp")
                            tf.logging.info("Saving model parameters in {}".format(ckpt.model_checkpoint_path))

                            eval_model.saver.restore(sess, ckpt.model_checkpoint_path)
                            tf.logging.info("Loading model from {}".format(ckpt.model_checkpoint_path))
                            validation_accuracy = self._eval_performance(sess, EvalModel, "validation", valid_init_op)
                            test_accuracy = self._eval_performance(sess, EvalModel, "test", test_init_op)
                            eval_model.save_dev_test_summary(self.summary_writer, sess, validation_accuracy, test_accuracy, global_step_val)
            except tf.errors.OutOfRangeError as e:
                exc_info = traceback.format_exc(sys.exc_info())
                msg = 'Out of range error:{}\n{}'.format(e, exc_info)
                tf.logging.warn(msg)
                tf.logging.info('Done training -- step limit reached')


def main(_):
    if FLAGS.job_name == "chief" and FLAGS.task_index == 0:
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    trainer = Trainer()
    try:
        trainer.train()
    except Exception as e:
        exc_info = traceback.format_exc(sys.exc_info())
        msg = 'creating session exception:{}\n{}'.format(e, exc_info)
        tmp = 'Run called even after should_stop requested.'
        should_stop = type(e) == RuntimeError and str(e) == tmp
        if should_stop:
            tf.logging.warn(msg)
        else:
            tf.logging.error(msg)
        # 0 means 'be over', 1 means 'will retry'
        exit_code = 0 if should_stop else 1
        sys.exit(exit_code)

if __name__ == "__main__":
    tf.logging.info("----start----")
    tf.app.run()
