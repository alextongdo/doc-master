from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from sqlalchemy.dialects.mysql import (
    LONGTEXT,
    TEXT,
    TINYTEXT,
    SMALLINT,
    INTEGER,
    DATETIME,
)
from sqlalchemy.exc import IntegrityError
from flask_login import UserMixin
from ast import literal_eval
from datasets import Dataset

Base = declarative_base()


class Users(UserMixin, Base):
    __tablename__ = "login"
    username = Column(String(128), primary_key=True, nullable=False)
    password = Column(String(128), nullable=False)

    def get_id(self):
        return str(self.username)


class TrainSession(Base):
    __tablename__ = "sessions"
    session_id = Column(String(64), primary_key=True, nullable=False)
    username = Column(String(128), nullable=False)
    datetime = Column(DATETIME, nullable=False)
    docs = Column(SMALLINT, nullable=False)


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(INTEGER, primary_key=True, autoincrement=True, nullable=False)
    session_id = Column(String(64), nullable=False)
    context = Column(LONGTEXT, nullable=False)
    question = Column(TEXT, nullable=False)
    answer = Column(TEXT, nullable=False)
    bbox = Column(LONGTEXT, nullable=False)
    page_size = Column(TINYTEXT, nullable=False)
    pdf_hash = Column(String(64), nullable=False)
    annotation_metadata = Column(LONGTEXT, nullable=False)


class Model(Base):
    __tablename__ = "models"
    model_id = Column(String(128), primary_key=True, nullable=False)
    architecture = Column(TINYTEXT, nullable=False)
    username = Column(String(128), nullable=False)
    datetime = Column(DATETIME, nullable=False)
    save_path = Column(TEXT, nullable=False)


engine = create_engine(
    "mysql+mysqlconnector://root:root@mysql-db:3306/ersp", echo=False
)
Base.metadata.bind = engine

# NOT the same as the session_id.
# Needed for database manipulation.
Session = scoped_session(sessionmaker(bind=engine))

# Creates tables if they don't exist
Base.metadata.create_all(engine)


def create_sess():
    return Session()


def findUser(sess, user):
    item = sess.query(Users).filter_by(username=user).first()
    return item


def createUser(sess, username, password):
    try:
        user = Users(username=username, password=password)
        sess.add(user)
        sess.commit()
        return True
    except IntegrityError as e:
        print("IntegrityError during createUser:", e)
        sess.rollback()
        return False
    except:
        return False


def saveAnnotations(sess, session_id, con, que, ans, bbx, ps, pdf_h, annot_meta):
    # While there may be multiple annotations to a
    # single PDF, this function can only be used for
    # one PDF at a time

    # questions: ['question1', 'question2', ...]
    # answers: [{'text': ['answer1'], 'answer_start': [10]}, ...]
    # annotation: [{}, {}, ...]

    entries = [
        Annotation(
            session_id=session_id,
            context=con,
            question=q,
            answer=repr(a),
            bbox=repr(bbx),
            page_size=repr(ps),
            pdf_hash=pdf_h,
            annotation_metadata=repr(m),
        )
        for q, a, m in zip(que, ans, annot_meta)
    ]

    try:
        sess.add_all(entries)
        sess.commit()
        return True
    except:
        return False


def save_model(sess, model_id, architecture, username, datetime, save_path):
    try:
        print("SAVING MODEL", flush=True)
        model = Model(
            model_id=model_id,
            architecture=architecture,
            username=username,
            datetime=datetime,
            save_path=save_path,
        )
        sess.add(model)
        sess.commit()
        return True
    except IntegrityError as e:
        print("IntegrityError during save_model:", e)
        sess.rollback()
        return False
    except Exception as e:
        print("Error:", e)
        return False


def downloadCSV(sess, session_id):
    data = sess.query(Annotation).filter_by(session_id=session_id).all()
    csv = [["context", "question", "answers", "bbox", "page_size"]]
    for item in data:
        csv.append(
            [item.context, item.question, item.answer, item.bbox, item.page_size]
        )
    return csv


def findPdfHash(sess, hash):
    annots = sess.query(Annotation).filter_by(pdf_hash=hash).all()
    if len(annots) == 0:
        return {"metadata": [], "oldSession": None}
    for row in annots:
        sess.delete(row)
    sess.commit()
    return {
        "metadata": [literal_eval(item.annotation_metadata) for item in annots],
        "oldSession": annots[0].session_id,
    }


def train_dataset(sess, session_ids, architecture):
    data = sess.query(Annotation).filter(Annotation.session_id.in_(session_ids)).all()

    if architecture == "roberta":
        dataset = [
            {
                "id": i,
                "title": "no title",
                "question": item.question,
                "context": item.context,
                "answers": literal_eval(item.answer),
            }
            for i, item in enumerate(data)
        ]
    elif architecture == "impira":
        dataset = [
            {
                "id": i,
                "title": "no title",
                "question": item.question,
                "context": item.context,
                "answers": literal_eval(item.answer),
                "bbox": item.bbox,  # Here the bbox is kept as a string instead of list because PyArrow doesn't play well with nesting
                "page_size": literal_eval(item.page_size),
            }
            for i, item in enumerate(data)
        ]
    dataset = Dataset.from_list(dataset)
    return dataset


def predict_dataset(filename, questions, context, bbox, ps):
    dataset = [
        {
            "id": f"{filename}__{i}",
            "title": filename,
            "question": ques,
            "context": context,
            "bbox": repr(bbox),
            "page_size": ps,
        }
        for i, ques in enumerate(questions)
    ]

    dataset = Dataset.from_list(dataset)
    return dataset


def get_model_names(sess):
    models = sess.query(Model).all()
    return [item.model_id for item in models]


def find_model(sess, model_id):
    existing_model = sess.query(Model).filter_by(model_id=model_id).first()
    return existing_model


def create_or_increment_docs_in_session(sess, session_id, username, datetime):
    item = sess.query(TrainSession).filter_by(session_id=session_id).first()
    if item:
        item.datetime = datetime
        item.docs = item.docs + 1
        sess.merge(item)
    else:
        sess.add(
            TrainSession(
                session_id=session_id, username=username, datetime=datetime, docs=1
            )
        )
    sess.commit()


def decrement_docs_in_session(sess, session_id):
    item = sess.query(TrainSession).filter_by(session_id=session_id).first()
    if item:
        item.docs = item.docs - 1
        sess.merge(item)
        sess.commit()


def get_train_table(sess):
    rows = sess.query(TrainSession).all()
    return [
        [item.session_id, item.username, item.datetime, item.docs]
        for item in rows
        if item.docs > 0
    ]


def get_predict_table(sess):
    models = sess.query(Model).with_for_update().all()
    predict_table = [
        [item.model_id, item.architecture if item.architecture != "impira" else "layoutlm", item.username, item.datetime]
        for item in models
    ]
    return predict_table
