from flask import (
    Flask,
    request,
    render_template,
    send_file,
    jsonify,
    redirect,
    url_for,
    session,
)
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from io import BytesIO, StringIO
from base64 import b64encode
from hashlib import sha256
from ast import literal_eval
from datasets import Dataset, concatenate_datasets
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
from zipfile import ZipFile
import fitz
import csv
import os
import database
import model
import re
import json
import uuid
from bbox import get_answer_bbox, highlight
from worker import celery, background_train_and_save
from difflib import SequenceMatcher

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
app.config["SECRET_KEY"] = os.urandom(12).hex()
celery.conf.update(app.config)

login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    db = database.create_sess()
    user = database.findUser(db, user_id)
    db.close()
    return user


@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for("login"))


@app.route("/logout")
@login_required
def logout():
    # Remove session data if exists
    session_dict.pop(session["loginSession"], None)
    logout_user()
    return redirect(url_for("login"))


# Models will be stored in instance foler
MODELS_DIR = os.path.join(app.instance_path, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
BASE_ROBERTA = "deepset/roberta-base-squad2"
BASE_IMPIRA = "impira/layoutlm-document-qa"


class PDF:
    def __init__(self, fn, byt, con, bbx, ps, answers=None):
        self.filename = fn
        self.bytes = byt
        self.context = con
        self.bbox = bbx
        self.page_size = ps
        self.answers = answers


class AnnotateSession:
    def __init__(self):
        self.pdfs = []

    def add_pdf(self, pdf):
        self.pdfs.append(pdf)

    def len_pdfs(self):
        return len(self.pdfs)


class PredictSession:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.architecture = None
        self.pdfs = []

    def add_model(self, architecture, path):
        self.architecture = architecture
        self.model, self.tokenizer = model.load_model(architecture, path)

    def add_pdf(self, pdf):
        self.pdfs.append(pdf)

    def len_pdfs(self):
        return len(self.pdfs)


# Global dict
session_dict = {}


@app.after_request
def disable_caching(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    # Render Page
    if request.method == "GET":
        return render_template("login.html")

    db = database.create_sess()

    # Login User
    if request.form.get("type") == "login":
        user = database.findUser(db, request.form.get("username"))
        db.close()
        if user is None:
            app.logger.debug(f"Logging in user {user} - not found!")
            return error("User could not be logged in!")
        if user.password != request.form.get("password"):
            app.logger.debug(f"Logging in user {user} - incorrect creds!")
            return error("User could not be logged in!")
        login_user(user)
        session["loginSession"] = uuid.uuid4().hex
        app.logger.debug(
            f"Logging in user {user.username} - success w/ session: {session['loginSession']}"
        )
        return success("User logged in!")

    # Create Account
    if request.form.get("type") == "register":
        created = database.createUser(
            db, request.form.get("username"), request.form.get("password")
        )
        db.close()
        if created:
            app.logger.debug(f"Creating user {request.form.get('username')} - success")
            return success("User created!")
        app.logger.debug(f"Creating user {request.form.get('username')} - failure")
        return error("User could not be created!")


@app.route("/")
@login_required
def homepage():
    app.logger.debug(f"User {current_user.username} at homepage")
    return render_template("home.html")


@app.route("/annotate/render")
@login_required
def show_pdfs_for_annotation():
    return render_template("annotate/render.html")


@app.route("/annotate/upload", methods=["GET", "POST"])
@login_required
def upload_pdfs_for_annotation():
    if request.method == "GET":
        return render_template("annotate/upload.html")

    session_dict[session["loginSession"]] = AnnotateSession()
    app.logger.debug(
        f"User {current_user.username} starting annotation with session: {session['loginSession']}"
    )

    # PDFs needed to get bbox
    file = request.files["pdfs"]
    filename = secure_filename(file.filename)

    # Saves one or more PDFs to global dict
    bad_pdf_msg = ""
    if filename.endswith(".pdf"):
        file_bytes = file.read()
        file.close()
        cont, bb, ps = context_bbox_ps(file_bytes)
        if len(cont) == 0:
            del session_dict[session["loginSession"]]
            app.logger.debug("We couldn't read that PDF!")
            return error("We couldn't read that PDF!")
        session_dict[session["loginSession"]].add_pdf(
            PDF(filename, file_bytes, cont, bb, ps)
        )
        return success(f"{bad_pdf_msg}")

    elif filename.endswith(".zip"):
        with ZipFile(file, "r") as input_zip:
            for subfile in [x for x in input_zip.namelist() if "__MACOSX/" not in x]:
                if subfile.endswith(".pdf"):
                    with input_zip.open(subfile) as curr_pdf:
                        file_bytes = curr_pdf.read()
                        cont, bb, ps = context_bbox_ps(file_bytes)
                        if len(cont) == 0:
                            bad_pdf_msg += f"{secure_filename(subfile)}"
                        else:
                            session_dict[session["loginSession"]].add_pdf(
                                PDF(secure_filename(subfile), file_bytes, cont, bb, ps)
                            )
                else:
                    bad_pdf_msg += f"{subfile}"
        # Whole ZIP broken
        if session_dict[session["loginSession"]].len_pdfs() == 0:
            del session_dict[session["loginSession"]]
            app.logger.debug("We couldn't read any files from that ZIP!")
            return error("We couldn't read any files from that ZIP!")
        app.logger.debug(
            f"Success - Num pdfs: {session_dict[session['loginSession']].len_pdfs()}"
        )
        return success(f"{bad_pdf_msg}")

    del session_dict[session["loginSession"]]
    app.logger.debug("Incorrect file type! Please upload a PDF or ZIP.")
    return error("Incorrect file type! Please upload a PDF or ZIP.")


def hashPDF(pdf, username):
    return sha256(
        pdf.bytes + pdf.filename.encode("utf-8") + username.encode("utf-8")
    ).hexdigest()


@app.route("/annotate/<int:index>", methods=["GET", "POST"])
@login_required
def get_one_pdf_to_show_for_annotation(index):
    if request.method == "GET":
        if session["loginSession"] not in session_dict:
            return error("PDF doesn't exist!")

        if not isinstance(session_dict[session["loginSession"]], AnnotateSession):
            return error("PDF doesn't exist!")

        if 0 <= index < session_dict[session["loginSession"]].len_pdfs():
            db = database.create_sess()
            requestedPDF = session_dict[session["loginSession"]].pdfs[index]
            hash = hashPDF(requestedPDF, current_user.username)
            response = database.findPdfHash(db, hash)

            if response["oldSession"] is not None:
                database.decrement_docs_in_session(db, response["oldSession"])

            response["filename"] = requestedPDF.filename
            response["pdf_bytes"] = b64encode(requestedPDF.bytes).decode("utf-8")
            db.close()
            return jsonify(response), 200

        return error("PDF doesn't exist!")

    # POST
    app.logger.debug(
        f"Trying to save some annotations for user {current_user.username}"
    )
    if session["loginSession"] not in session_dict:
        return error("PDF doesn't exist!")

    if not isinstance(session_dict[session["loginSession"]], AnnotateSession):
        return error("PDF doesn't exist!")

    annotations = json.loads(request.form["annotations"])
    curr_pdf = session_dict[session["loginSession"]].pdfs[index]

    context = curr_pdf.context
    pymu_page_size = curr_pdf.page_size
    context_clean, mapping = create_clean_strings_and_mapping(context)
    bbox_list = curr_pdf.bbox

    question_list = []
    answer_list = []
    for annotation in annotations:
        try:
            match_span = find_span(
                context_clean, annotation, mapping, pymu_page_size, bbox_list
            )
            if match_span is not None:
                match = context[match_span[0] : match_span[1]]
                if SequenceMatcher(None, match, annotation["answer"]).ratio() >= 0.8:
                    app.logger.debug(
                        f"Selection from PyMUPDF context: {repr(f'{context[match_span[0]-10: match_span[0]]}[{context[match_span[0]: match_span[1]]}]{context[match_span[1]: match_span[1]+10]}')}"
                    )
                    question_list.append(annotation["question"])
                    answer_list.append(
                        {
                            "text": [context[match_span[0] : match_span[1]]],
                            "answer_start": [match_span[0]],
                        }
                    )
        except:
            continue

    app.logger.debug("Annotations that passed validation")
    app.logger.debug(question_list)
    app.logger.debug(json.dumps(answer_list, indent=2))

    pdf_hash = hashPDF(curr_pdf, current_user.username)
    app.logger.debug(f"Saving PDF with hash: {pdf_hash}")

    db = database.create_sess()
    saved = database.saveAnnotations(
        db,
        session_id=session["loginSession"],
        con=context,
        que=question_list,
        ans=answer_list,
        bbx=bbox_list,
        ps=pymu_page_size,
        pdf_h=pdf_hash,
        annot_meta=annotations,
    )

    if saved:
        database.create_or_increment_docs_in_session(
            db, session["loginSession"], current_user.username, datetime.today()
        )
        db.close()
        return success("Annotations saved!")
    db.close()
    return error("Annotation saving failed!")


@app.route("/annotate/get_all")
@login_required
def get_all_annotations_of_current_session():
    db = database.create_sess()
    csv_data = database.downloadCSV(db, session["loginSession"])
    db.close()
    if len(csv_data) <= 1:
        return error("There were no annotations!")
    with StringIO() as outf:
        w = csv.writer(outf)
        w.writerows(csv_data)
        outf.seek(0)
        binary = BytesIO(outf.getvalue().encode("utf-8"))
        return send_file(
            binary,
            as_attachment=True,
            mimetype="text/csv",
            download_name="annotations.csv",
        )


@app.route("/train", methods=["GET", "POST"])
@login_required
def train():
    db = database.create_sess()
    if request.method == "GET":
        train_table = database.get_train_table(db)
        model_names = database.get_model_names(db)
        db.close()
        app.logger.debug(f"User {current_user.username} at train page")
        app.logger.debug(train_table)
        # Using Jinja2 templating to fill in table
        return render_template("train.html", table=train_table, models=model_names)

    # Input: List of session_ids and architecture to do training
    data = request.get_json()
    session_ids = data.get("session_ids")
    model_id = data.get("model_id")
    architecture = data.get("architecture")

    try:
        existing = database.find_model(db, model_id)
        while existing is not None:
            model_id = increment_model_name(model_id)
            existing = database.find_model(db, model_id)

        path_saved = os.path.join(
            MODELS_DIR,
            f"{secure_filename(model_id)}__{int(datetime.now().timestamp())}",
        )

        task = background_train_and_save.delay(
            architecture, session_ids, path_saved, model_id, current_user.username
        )
        db.close()
        return success("Model training in background!")
    except Exception as e:
        db.close()
        app.logger.debug(f"Error while training: {e}")
        return error("Model training error!")


@app.route("/predict/select", methods=["GET", "POST"])
@login_required
def predict_select_model():
    db = database.create_sess()
    if request.method == "GET":
        app.logger.debug(f"User {current_user.username} entering prediction.")
        predict_table = database.get_predict_table(db)
        db.close()
        return render_template("predict/select.html", models=predict_table)

    try:
        session_dict[session["loginSession"]] = PredictSession()
        data = request.get_json()
        model_id = data.get("model_id")
        app.logger.debug(f"model_id for predict: {model_id}")
        model_info = database.find_model(db, model_id)
        db.close()
        app.logger.debug(f"model_path found: {model_info.save_path}")
        if not os.path.exists(model_info.save_path):
            raise Exception("Model path was not found - faulty entry in db")

        session_dict[session["loginSession"]].add_model(
            model_info.architecture, model_info.save_path
        )
        # Redirect to predict/upload page
        return success("Predict model selection success.")
    except Exception as e:
        db.close()
        del session_dict[session["loginSession"]]
        app.logger.debug(f"Model selection error: {e}")
        return error("Predict model selection failed.")


@app.route("/predict/upload", methods=["GET", "POST"])
@login_required
def predict_upload():
    # Page to let the user upload some prediction docs and input some questions
    if request.method == "GET":
        if session["loginSession"] not in session_dict:
            return redirect(url_for("predict_model_choice")), 301

        if not isinstance(session_dict[session["loginSession"]], PredictSession):
            return redirect(url_for("predict_model_choice")), 301

        if session_dict[session["loginSession"]].architecture is None:
            return redirect(url_for("predict_model_choice")), 301

        return render_template("predict/upload.html")

    if session["loginSession"] not in session_dict:
        return error("loginSession didnt exist!")

    if not isinstance(session_dict[session["loginSession"]], PredictSession):
        return error("You did not visit the /predict page!")

    if session_dict[session["loginSession"]].architecture is None:
        return error("You did not successfully select a model!")

    session_dict[session["loginSession"]].pdfs = []

    # PDFs needed for prediction
    file = request.files["pdfs"]
    filename = secure_filename(file.filename)
    questions = json.loads(request.form["questions"])

    # Saves one or more PDFs to global dict
    if filename.endswith(".pdf"):
        file_bytes = file.read()
        file.close()
        cont, bb, ps = context_bbox_ps(file_bytes)
        if len(cont) == 0:
            return error("Cannot read that PDF!")
        predictions = {}
        dataset = database.predict_dataset(filename, questions, cont, bb, ps)
        nbest = model.predict(
            session_dict[session["loginSession"]].architecture,
            session_dict[session["loginSession"]].model,
            session_dict[session["loginSession"]].tokenizer,
            dataset,
        )
        app.logger.debug("Nbest Predictions:")
        app.logger.debug(json.dumps(nbest, indent=2))

        for row in dataset:
            predictions[row["question"]] = get_answer_bbox(
                nbest[row["id"]], row["context"], literal_eval(row["bbox"])
            )
        highlighted_doc, answer_locations = highlight(file_bytes, predictions)
        session_dict[session["loginSession"]].add_pdf(
            PDF(filename, highlighted_doc, None, None, None, answer_locations)
        )
        return success("Prediction success.")

    elif filename.endswith(".zip"):
        bad_pdf_msg = ""
        predictions = {}
        bytes_list = {}
        dataset = Dataset.from_dict(
            {
                "id": [],
                "title": [],
                "question": [],
                "context": [],
                "bbox": [],
                "page_size": [],
            }
        )
        with ZipFile(file, "r") as input_zip:
            for subfile in [x for x in input_zip.namelist() if "__MACOSX/" not in x]:
                if subfile.endswith(".pdf"):
                    with input_zip.open(subfile) as curr_pdf:
                        file_bytes = curr_pdf.read()
                        cont, bb, ps = context_bbox_ps(file_bytes)
                        if len(cont) == 0:
                            bad_pdf_msg += f"{secure_filename(subfile)}"
                        else:
                            predictions[subfile] = {}
                            bytes_list[subfile] = file_bytes
                            subfile_dataset = database.predict_dataset(
                                subfile, questions, cont, bb, ps
                            )
                            dataset = concatenate_datasets([dataset, subfile_dataset])
                else:
                    bad_pdf_msg += f"{subfile}"
            # Whole ZIP broken
            if len(dataset) == 0:
                return error("Cannot read any files in ZIP!")

            nbest = model.predict(
                session_dict[session["loginSession"]].architecture,
                session_dict[session["loginSession"]].model,
                session_dict[session["loginSession"]].tokenizer,
                dataset,
            )

            for row in dataset:
                file_n = row["title"]
                predictions[file_n][row["question"]] = get_answer_bbox(
                    nbest[row["id"]], row["context"], literal_eval(row["bbox"])
                )

            for file_n in predictions.keys():
                highlighted_doc, answer_locations = highlight(
                    bytes_list[file_n], predictions[file_n]
                )
                app.logger.debug("answers")
                app.logger.debug(json.dumps(answer_locations, indent=2))
                session_dict[session["loginSession"]].add_pdf(
                    PDF(file_n, highlighted_doc, None, None, None, answer_locations)
                )
        return success(f"{bad_pdf_msg}")

    return error("Incorrect file type!")


@app.route("/predict/render")
@login_required
def predict_render():
    return render_template("predict/render.html")


@app.route("/predict/<int:index>", methods=["GET", "POST"])
@login_required
def predict_get_pdf(index):
    if request.method == "GET":
        if session["loginSession"] not in session_dict:
            return error("PDF doesn't exist!")
        if not isinstance(session_dict[session["loginSession"]], PredictSession):
            return error("PDF doesn't exist!")

        if 0 <= index < session_dict[session["loginSession"]].len_pdfs():
            pdf = session_dict[session["loginSession"]].pdfs[index]
            if pdf.answers is None:
                return error("PDF doesn't exist!")

            response = {}
            response["filename"] = pdf.filename
            response["pdf_bytes"] = b64encode(pdf.bytes).decode("utf-8")
            response["answers"] = pdf.answers
            return jsonify(response), 200
        return error("PDF doesn't exist!")


@app.route("/predict/download_all")
@login_required
def predict_get_all():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zip_file:
        for pdf in session_dict[session["loginSession"]].pdfs:
            zip_file.writestr(pdf.filename, pdf.bytes)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        mimetype="application/zip",
        download_name="highlighted.zip",
    )


# Helper functions
def error(message):
    return jsonify({"error": message}), 400


def success(message):
    return jsonify({"success": message}), 200


def round_decimals(coord, places):
    return (
        round(coord[0], places),
        round(coord[1], places),
        round(coord[2], places),
        round(coord[3], places),
        coord[4],
    )


# Get context, bbox, page size
def context_bbox_ps(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = ""
        word_info = []
        ps = [0, 0]
        for i, page in enumerate(doc):
            text += page.get_text(sort=False)
            ps[0] = max(ps[0], page.mediabox.x1)
            ps[1] = max(ps[1], page.mediabox.y1)
            word_info.extend(
                [
                    round_decimals(word[:5], places=2) + (i,)
                    for word in page.get_text("words", sort=False)
                ]
            )
    print(f"New annots added -- page_size: {repr(ps)}")
    return text, word_info, ps


def increment_model_name(filename):
    pattern = r"^(.+)\s+\((\d+)\)$"
    match = re.match(pattern, filename)
    if match:
        filename_before_integer = match.group(1)
        integer_group = match.group(2)
        if integer_group is not None:
            integer_value = int(integer_group) + 1
        return f"{filename_before_integer} ({integer_value})"
    return f"{filename} (1)"


def create_clean_strings_and_mapping(original):
    original_clean = re.sub(r"\s", "", original.lower())
    nonwhite_indexes = [j for j, s in enumerate(original) if not s.isspace()]
    mapping = {idx: i for idx, i in enumerate(nonwhite_indexes)}
    mapping[len(nonwhite_indexes)] = mapping[len(nonwhite_indexes) - 1]
    return original_clean, mapping


def closest_span(given_span, span_list):
    current_min = float("inf")
    closest = None
    for span in span_list:
        distance = min(abs(given_span[0] - span[0]), abs(given_span[1] - span[1]))
        if distance < current_min:
            current_min = distance
            closest = span
    return closest


def substring_occurrence(string, span):
    clean_string, mapping = create_clean_strings_and_mapping(string)
    clean_substring = re.sub(r"\s", "", string[span[0] : span[1]].lower())
    offset_matches = [
        (mapping[i.start()], mapping[i.end()])
        for i in re.finditer(re.escape(clean_substring), clean_string)
    ]
    return closest_span(span, offset_matches)


def find_anchor_span_from_bbox(bbox_list, anchor_bbox, anchor_text, page_num):
    anchor_clean = re.sub(r"\s", "", anchor_text.lower())
    starting_index = 0
    for bbox in bbox_list:
        if (page_num - 1) == bbox[5] and fitz.Rect(bbox[:4]).intersects(anchor_bbox):
            break
        else:
            starting_index += len(bbox[4].lower())
    return (starting_index, starting_index + len(anchor_clean))


# whitespace_mapping: whitespace span to original span
def find_span(
    original_clean, annotation, whitespace_mapping, pymu_page_size, bbox_list
):
    substring_clean = re.sub(r"\s", "", annotation["answer"].lower())
    app.logger.debug(f"Annotation to find: {repr(substring_clean)}")
    matches = [
        (i.start(), i.end())
        for i in re.finditer(re.escape(substring_clean), original_clean)
    ]
    if len(matches) == 0:
        app.logger.debug(f"Could not find annot: {repr(substring_clean)}\n")
        return None
    elif len(matches) == 1:
        clean_match = (matches[0][0], matches[0][1])
        app.logger.debug(f"One match for annot: {repr(substring_clean)}\n")
        app.logger.debug(original_clean[clean_match[0] : clean_match[1]])
        return (whitespace_mapping[clean_match[0]], whitespace_mapping[clean_match[1]])
    else:
        app.logger.debug(f"Multiple matches for substring: {repr(substring_clean)}\n")

        textlayer_page_size = literal_eval(annotation["page_size"])
        annotation["anchor"] = literal_eval(annotation["anchor"])
        annotation["focus"] = literal_eval(annotation["focus"])
        anchor_bbox = annotation["anchor"]["bbox"]
        focus_bbox = annotation["focus"]["bbox"]

        x_scale = pymu_page_size[0] / textlayer_page_size[0]
        y_scale = pymu_page_size[1] / textlayer_page_size[1]

        scaled_anchor_bbox = [
            anchor_bbox[0] * x_scale,
            anchor_bbox[1] * y_scale,
            anchor_bbox[2] * x_scale,
            anchor_bbox[3] * y_scale,
        ]
        anchor_span = find_anchor_span_from_bbox(
            bbox_list,
            scaled_anchor_bbox,
            annotation["anchor"]["text"],
            int(annotation["anchor"]["page_num"]),
        )
        app.logger.debug(
            f"Found anchor text: {original_clean[anchor_span[0]: anchor_span[1]]}"
        )

        if anchor_bbox == focus_bbox:
            app.logger.debug("Anchor was same as focus")
            subcontext_span = (anchor_span[0], anchor_span[1])
            if annotation["anchor"]["offset"] <= annotation["focus"]["offset"]:
                offset_span = (
                    annotation["anchor"]["offset"],
                    annotation["focus"]["offset"],
                )
            else:
                offset_span = (
                    annotation["focus"]["offset"],
                    annotation["anchor"]["offset"],
                )

        else:
            scaled_focus_bbox = [
                focus_bbox[0] * x_scale,
                focus_bbox[1] * y_scale,
                focus_bbox[2] * x_scale,
                focus_bbox[3] * y_scale,
            ]
            focus_span = find_anchor_span_from_bbox(
                bbox_list,
                scaled_focus_bbox,
                annotation["focus"]["text"],
                int(annotation["focus"]["page_num"]),
            )
            app.logger.debug(
                f"Found focus text: {original_clean[focus_span[0]: focus_span[1]]}"
            )
            if anchor_span[0] <= focus_span[0]:
                subcontext_span = (anchor_span[0], focus_span[1])
                offset_span = (
                    annotation["anchor"]["offset"],
                    annotation["focus"]["offset"]
                    + len(annotation["anchor"]["text"])
                    - 1,
                )
            else:
                subcontext_span = (focus_span[0], anchor_span[1])
                offset_span = (
                    annotation["focus"]["offset"],
                    annotation["anchor"]["offset"]
                    + len(annotation["focus"]["text"])
                    - 1,
                )

        subcontext = original_clean[subcontext_span[0] : subcontext_span[1]]
        app.logger.debug(f"Subcontext: {repr(subcontext)}")
        offset_matches = [
            (i.start(), i.end())
            for i in re.finditer(re.escape(substring_clean), subcontext)
        ]
        if len(offset_matches) == 0:
            app.logger.debug("Couldn't find in subcontext")
            return None
        elif len(offset_matches) == 1:
            clean_match = (
                offset_matches[0][0] + subcontext_span[0],
                offset_matches[0][1] + subcontext_span[0],
            )
            app.logger.debug(
                f"Singular match: {original_clean[clean_match[0]: clean_match[1]]}"
            )
            return (
                whitespace_mapping[clean_match[0]],
                whitespace_mapping[clean_match[1]],
            )

        else:
            app.logger.debug("Rely on textlayer char index")
            matches_from_clean = [
                (span[0] + subcontext_span[0], span[1] + subcontext_span[0])
                for span in offset_matches
            ]
            matches_from_whitespace = [
                (whitespace_mapping[span[0]], whitespace_mapping[span[1]])
                for span in matches_from_clean
            ]
            offset_span_from_whitespace = (
                offset_span[0] + whitespace_mapping[subcontext_span[0]],
                offset_span[1] + whitespace_mapping[subcontext_span[0]],
            )
            solution = closest_span(
                offset_span_from_whitespace, matches_from_whitespace
            )
            return solution


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8102, debug=True)
