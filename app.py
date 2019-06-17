from flask import Flask, render_template, request, redirect, url_for, flash
import os
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from ocrErry import main

app = Flask(__name__)
Bootstrap(app)

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'jpeg'])
app.secret_key = 'secret'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":

        image = request.files['file']
        nameBoolean = (request.form.get("nameBool") == "checked")
        nationBoolean = (request.form.get("nationBool") == "checked")
        idBoolean = (request.form.get("idBool") == "checked")
        genderBoolean = (request.form.get("genderBool") == "checked")
        birthdayBoolean = (request.form.get("birthdayBool") == "checked")
        expireBoolean = (request.form.get("expireBool") == "checked")

        if image:
           # image = request.files['file']
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
            img_src = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            (name, nationality, birthday, idNum, gender, time, expire) = main(img_src)
            return render_template("web.html", user_image=image.filename, msg="Upload succeed", nationBoolean=nationBoolean, idBoolean=idBoolean, genderBoolean=genderBoolean, birthdayBoolean=birthdayBoolean, nameBoolean=nameBoolean, name=name, nationality=nationality, idNum=idNum, birthday=birthday, time=time, gender=gender, expire=expire, expireBoolean=expireBoolean)
        else:
            return render_template('web.html', msg='No file selected')

    else:
        return render_template('web.html')

    

if __name__ == '__main__':
    app.run(debug=True)

