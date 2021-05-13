from flask_cors import CORS
from flask import Flask,render_template,request,jsonify,url_for
from contact_classify import pre_classfiy
from entity_extraction import pre_entity
from script.txt_to_ann import txt_to_ann
app = Flask(__name__)
CORS(app)




@app.route('/contact/classify', methods = ["POST"])
def classify():
    path = request.files['path']
    pro = pre_classfiy.run(path)
    print(pro)
    return {
        "accuracy" : pro.tolist()
    }

@app.route('/contact/entity', methods = ["POST"])
def entity_extract():
    path = request.files['path']
    content, txt_label = pre_entity.run(path, path)
    ann_info = txt_to_ann.run(content, txt_label)


    return {
        "ann_info" : ann_info,
        "content" : content
    }

@app.route('/contact/relation', methods = ["POST"])
def relation_extract():
    path = request.files['path']
    content, txt_label = pre_entity.run(path, path)
    ann_info = txt_to_ann.run(content, txt_label)


    return {
        "ann_info" : ann_info,
        "content" : content
    }



if __name__ == '__main__':
    app.run(debug=True,port =8899, host="0.0.0.0" )