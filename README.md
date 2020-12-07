# contact_classify

常用文件所在位置

- 参数文件：contact_classify/normal_param.py
- 词汇文件：contact_classify/vocab

对合同数据进行分类，获取分类模型

- 训练数据进行分类方法入口：contact_classify/train.py
- 合同数据文本类型：docx


通过flask将分类模型封装成接口形式以供调用

- 使用分类模型：CNN
- 可调用模型分类接口所在文件：test_script/test_classify.py

