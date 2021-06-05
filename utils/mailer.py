import smtplib
from email.message import EmailMessage
import imghdr
import time
import os

def mail(name, from_, to, intruder_folder, password): #OR CSV Index OR Swipe Signal Input 
    print(from_)
    print(to)
    msg = EmailMessage()
    msg['Subject'] = 'Warning! Tailgating Detected'
    msg['From'] = from_
    msg['To'] = to
    msg.set_content(
'''Dear {},

A Tailgating incident was detected behind you at the time {}. We have attached an image of the possible intruder. You are requested to report at the security frontdesk immediately. 

If this was done knowingly, please refrain from doing so and follow the guidelines as laid down by the authority.

Stay Alert.

Regards,
Security Department'''.format(name,time.strftime("%H:%M:%S", time.localtime())) #User Name as Input + Local Time
    )
    with open(os.path.join(intruder_folder,os.listdir('Intruders')[-1]),'rb') as f:
       file_data = f.read()
       file_type = imghdr.what(f.name)
       file_name = f.name
    msg.add_attachment(file_data, maintype = 'image',subtype=file_type, filename = file_name)
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(from_, password)
        smtp.send_message(msg)