import sqlite3

conn = sqlite3.connect('user_bank.db',  check_same_thread=False)
cursor = conn.cursor()


def update_by_account_number(acc_num, amount):
    sql_query = "UPDATE user_data set AMOUNT = ? where ACC_NUMBER = ?"
    data = (amount, acc_num)
    cursor.execute(sql_query, data)
    conn.commit()
    print("Record Updated successfully")


def retrieve_by_account_number(acc_num):
    sql_query = "select * from user_data where ACC_NUMBER = ?"
    data = (acc_num,)
    a = cursor.execute(sql_query, data)
    conn.commit()
    print("Record Updated successfully")
    return a


