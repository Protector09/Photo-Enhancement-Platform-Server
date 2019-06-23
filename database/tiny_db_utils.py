from tinydb import TinyDB
from config import DB_PATH


def create_db_and_main_tables(database_path):

    database = TinyDB(database_path)

    database.table(name="User")
    # username
    # password
    # occupation
    # credit_card_no
    # subscription_type

    database.table(name="Operation")
    # username
    # date
    # current_subscription_type
    # no_operations
    # price


def purge_db(database_path):

    database = TinyDB(database_path)

    database.purge_tables()



if __name__ == "__main__":

    initialized = True
    if not initialized:
        purge_db(DB_PATH)
        create_db_and_main_tables(DB_PATH)

    db = TinyDB(DB_PATH)

