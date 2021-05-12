from script.txt_to_ann import txt_to_ann
from entity_extraction import gen_label
'''
用于将txt预测得到对应的label并将其转换成ann格式。
'''
gen_label.run("E:/test_contract/txt", "E:/test_contract/label")
txt_to_ann.run("E:/test_contract", "E:/test_contract/ann")


