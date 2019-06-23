import logging
from datetime import datetime

from tinydb import TinyDB

from config import DB_PATH
from monetary.pricesEnum import Prices
from utils.modelsEnums import Models


def charge_opperation(db_path, username, quantity):
    db = TinyDB(db_path)

    users = db.table("User")
    operations = db.table("Operation")

    subscription_type = Prices.DEFAULT

    for user in users:

        if user["username"] == username:

            for enum in Prices:
                if str(enum) == "Prices." + user["subscription_type"]:
                    subscription_type = enum
                    break
        if subscription_type != Prices.DEFAULT:
            break

    logging.info("User {} did {} operations for a total price of {} at {}".format(username,
                                                                                  quantity,
                                                                                  quantity * subscription_type.value,
                                                                                  datetime.now()))

    operations.insert(
        {'username': username,
         'date': str(datetime.now()),
         'current_subscription_type': subscription_type.name,
         'no_operations': quantity,
         'price': quantity * subscription_type.value
         })


def collect_earnings():
    db = TinyDB(DB_PATH)

    users = db.table("User")
    operations = db.table("Operation")

    list_of_payments = []
    usernames = {}

    for user in users:
        usernames[user["username"]] = 0.0

    for op in operations:
        usernames[op["username"]] += op["price"]


    operations.purge()


if __name__ == "__main__":
    # charge_opperation("david_dan", 10)
    # collect_earnings()
    pass
