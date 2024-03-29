import shutil
import os
import datetime
import random
from utils import check_utils
import re
import pickle
def copy_move(src_path, dest_path):
    '''
    :param src_path 源文件
    :param dest_path 需要剪切到的路径
    将文件剪切到另一个文件夹
    '''
    try:
        check_utils.check_path(dest_path)
        shutil.move(src_path, dest_path)
    except:
        pass

def copy_replace(src_path, dest_path):
    '''
    :param src_path 源文件
    :param dest_path 需要剪切到的路径
    将文件剪切到另一个文件夹
    '''
    try:
        check_utils.check_path(dest_path)
        shutil.copy(src_path, dest_path)
    except:
        pass



def get_file_name(path_name):
    '''对有后缀的文件名的后缀进行删除
    :param path_name 待处理文件名
    :return 已删除后缀名的文件名 '''
    if not check_utils.check_path_style(path_name, "path"):
        return path_name.split(".")[0]

def get_suffix(path_name):
    '''对有后缀的文件名的后缀进行获取
    :param path_name 待处理文件名
    :return 已获取文件的后缀名 '''
    return path_name.split(".")[1]

def names_filter(names,list_filts):
    '''根据条件过滤返回需要的文件名
    :param names 需要被过滤的文件名
    :param list_filt(list) 过滤条件
    :return names 过滤完成的文件名'''
    filted_name = []
    for name in names:
        for filt in list_filts:
            if re.search(filt, name):
                filted_name += [name]
    return filted_name

def concat_file_path(path, files_name):
    '''对所有的files_name进行拼接，返回该path文件夹下所有的文件（夹）的路径
    :param path 总路径
    :param files_name(list) 路径下文件（夹）名称列表
    :return 该path文件夹下所有的文件（夹）的路径'''
    files_path = []
    for file_name in files_name:
        if not check_utils.check_path_style(file_name, mode="suffix",is_flag=False):
            continue
        file_path = os.path.join(path, file_name)
        files_path += [file_path]
    return files_path

def get_all_path(path):
    return os.listdir(path)

def get_doc_file(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".doc"):  # 排除文件夹内的其它干扰文件，只获取".doc"后缀的word文件
            files.append(path + file)
    return files

def read_vocab(vocab_path):
    '''
    读取词表内容
    :param vocab_path: 词表路径
    :return: 词表dic文件
    '''
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def read_txt(txt_path):
    '''
    读取词表内容
    :param vocab_path: 词表路径
    :return: 词表dic文件
    '''
    with open(txt_path, 'r', encoding="utf-8") as f:
        content = f.read()
    return content

def shuffle(txt, label):
    length = len(txt)
    index = [i for i in range(length)]
    random.shuffle(index)
    txt = [txt[i] for i in index]
    label = [label[i] for i in index]
    return txt, label

def shuffle_(txt):
    length = len(txt)
    index = [i for i in range(length)]
    random.shuffle(index)
    txt = [txt[i] for i in index]
    return txt

def write_content(content, write_path):
    with open(write_path, 'a', encoding='utf-8') as f:
        f.write(content)

def write_content_renew(content, write_path):
    with open(write_path, 'w', encoding='utf-8') as f:
        f.write(content)

def concat_path(head_path):
    paths = [os.path.join(head_path, path_name) for path_name in os.listdir(head_path)]
    return paths

def gain_filename_from_path(path, mode):
    if mode == 'txt':
        name = re.search("(\\d+)\\.txt", path).group(1)
        return ("%s.txt"% name)

def get_file_path(base_path):
    '''将ann和txt文件分开；来获取路径列表
    :param base_path (str) 基础路径（包括ann和txt的）
    :return anns ann后缀的路径
    :return txt txt后缀的路径'''
    path_lists = os.listdir(base_path)
    ann_files = names_filter(path_lists, ["ann"])
    txt_files = names_filter(path_lists, ["txt"])
    anns = [os.path.join(base_path, i) for i in ann_files]
    txts = [os.path.join(base_path, i) for i in txt_files]
    return anns, txts, txt_files