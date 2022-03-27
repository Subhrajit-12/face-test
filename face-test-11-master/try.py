import sqlite3

conn = sqlite3.connect('user_bank.db')
print ("Opened database successfully")

conn.execute("INSERT INTO user_data (ID,NAME,ACC_NUMBER,AMOUNT) \
      VALUES (1, 'rrr', '76efbabe-3fd6-4622-bd2d-63f0bc222d76', '65000' )");

print ("Table created successfully")
conn.commit()
conn.close()