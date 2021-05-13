from entity_extraction import prediction_entity, normal_param,NER_pre_data, process_data_for_keras
from utils import normal_util, check_utils
import os
from tqdm import tqdm
from pro_data import eval_prodata as eval
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from sklearn.metrics import classification_report
def clean_txt(txts):
    '''
    将文件中的空格去除干净
    :param path: 文件路径
    '''
    # txts = normal_util.read_txt(path)
    list_txts = txts.split("\n")
    clear_contents = []
    for index, list_txt in enumerate(list_txts):
        list_txt = list_txt.replace(" ", "").replace("\u3000", "")
        if len(list_txt) == 0:
            continue
        clear_contents.append(list_txt)
    return clear_contents



def gen_label(path, write_path):
    labels = prediction_entity.prediction(path)
    txt_label = []
    for index, label in enumerate(labels):
        txt_label.append(" ".join(str(i) for i in label))
    return txt_label


def run(head_path, write_path):
    pro = eval.process_data()

    content = pro.load_data_docx(head_path)
    content = clean_txt(content)
    corr_write_path = os.path.join(check_utils.check_and_build(write_path), "label.txt")
    txt_label = gen_label(content, corr_write_path)
    return content, txt_label

def pre_score(head_path):

    paths = normal_util.concat_path(os.path.join(head_path, "txt"))
    label_paths = normal_util.concat_path(os.path.join(head_path, "label"))
    labels_all = []
    pre_labels_all = []

    for index in range(len(paths)):
        labels = prediction_entity.prediction(paths[index], mode="rnn")


        labels_entire =NER_pre_data.read_content(label_paths[index])
        for i in range(len(labels)):
            if len(labels[i]) != len(labels_entire[i]):
                continue
            pre_labels_all += labels[i]
            labels_all += labels_entire[i]
    print(classification_report(pre_labels_all, labels_all))



if __name__ == "__main__":
    # run("F:/contract/txt", "F:/contract/label")
    # gen_label("F:/contract/txt/17.txt", "F:/contract/label/17.txt")
    pre_score(normal_param.head_test_path)