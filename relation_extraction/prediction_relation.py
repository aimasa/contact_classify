from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from relation_extraction import eval
import os
from model import modeling
from relation_extraction.pre import tokenization
import tensorflow as tf
from relation_extraction.bert_EL_muti_nobilstm import SemEvalProcessor
from relation_extraction import bert_EL_muti_nobilstm
from utils import normal_util
flags = tf.flags
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def pre_relation(mode):
    '''
    根据指定模型对文本中实体对应关系进行预测
    :param mode: 模型种类
    :return:
    '''
    if mode == "bert-EL_muti":
        ''''''
        bert_EL_muti()

def bert_EL_muti():
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "semeval": SemEvalProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)


    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels(FLAGS.data_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))
    model_fn = bert_EL_muti_nobilstm.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        labels_list=label_list)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    predict_examples, entity_T, all_num_sentences = processor.get_prediction_examples(FLAGS.data_dir)
    labels = processor.get_labels(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)


    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    bert_EL_muti_nobilstm.file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = bert_EL_muti_nobilstm.file_based_input_fn_builder(
        input_file=predict_file,

        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=True,
        relation_num=len(label_list)
    )

    result = estimator.predict(input_fn=predict_input_fn)
    length = 0

    evaluator = eval.chunkEvaluator(label_list)
    tf.logging.info("***** Predict results *****")
    num_no = 0
    list_dic = []
    last_txt_no = 0
    paths = []
    all_list_dic = []
    for prediction, example in zip(result, predict_examples):
        paths.append(example.path)
        results = evaluator.precision(prediction)
        contents = [i for i in example.text_a]
        if last_txt_no is not example.entity_T:
            last_txt_no = example.entity_T
            num_no = 0
            all_list_dic += list_dic
            write_ann(list_dic, paths[len(paths) - 2])
            list_dic = []
        index = example.entity_T
        for tmp in results:
            dic = {}
            entity_first = find_entity(entity_T[index], example, tmp[0], all_num_sentences[index])
            entity_last = find_entity(entity_T[index], example, tmp[-1], all_num_sentences[index])
            if entity_first is None or entity_last is None:
                continue
            gain_relation_dic(entity_first, entity_last, tmp[1], num_no, dic)
            list_dic.append(dic)
            num_no += 1
    write_ann(list_dic, paths[len(paths) - 2])
    all_list_dic += list_dic






def write_ann(list_dic, path):
    '''
    将关系写入对应路径中
    :param list_dic: 存在的关系序列列表
    :param example:
    :return:
    '''
    for dic in list_dic:
        relation_content = "\n%s\t%s Arg1:%s Arg2:%s\t"%(dic["No"], dic["relation"], dic["arg1"], dic["arg2"])
        normal_util.write_content(relation_content, path)


def gain_relation_dic(entity_first, entity_last, relation, relation_no, dic):
    dic["No"] = "%s%s" % ("R", relation_no)
    dic["relation"] = relation
    dic["arg1"] = entity_first
    dic["arg2"] = entity_last

def find_entity(entity_T, example, entity_index, num_sentences):
    '''
    找到实体对应的代号
    :param entity_T: 包含实体代号、实体对应位置的信息表
    :param example: 当前句子信息，包含了
    :param entity_index:
    :param num_sentences:
    :return:
    '''
    entity_first_index = int(num_sentences[example.guid]) + entity_index - 1
    # 根据偏移值找到句子中的实体下标在全文中的位置。
    for key in entity_T.keys():
        local = entity_T[key]
        if int(local["local"][0]) == entity_first_index:
            return key



if __name__ == "__main__":
    bert_EL_muti()