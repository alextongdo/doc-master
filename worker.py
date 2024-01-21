from celery import Celery
from database import train_dataset, save_model, create_sess
from model import train, load_model
from datetime import datetime
# from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
# from sqlalchemy import create_engine
# from sqlalchemy.exc import IntegrityError

celery = Celery('worker', broker='redis://redis:6379/0', backend='redis://redis:6379/0')
BASE_ROBERTA = "deepset/roberta-base-squad2"
BASE_IMPIRA = "impira/layoutlm-document-qa"

@celery.task
def background_train_and_save(architecture, session_ids, path_saved, model_id, username):
    print("starting worker job", flush=True)
    if architecture == "roberta":
        model, tokenizer = load_model("roberta", BASE_ROBERTA)
    elif architecture == "impira":
        model, tokenizer = load_model("impira", BASE_IMPIRA)
    else:
        raise Exception("Invalid architecture!")
    
    db = create_sess()
    dataset = train_dataset(db, session_ids, architecture)
    train(
        architecture,
        model,
        tokenizer,
        dataset,
        path_saved,
    )

    saved = save_model(
        db, model_id, architecture, username, datetime.today(), path_saved
    )
    db.close()
    if not saved:
        raise Exception(f"Error while saving trained model")
    else:
        print(f"saved correctly {saved}")
    print("finished worker job", flush=True)
