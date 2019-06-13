import time
import pymysql
import urllib
import Object_detection_webcam
import cv2
import numpy as np

con = pymysql.connect("185.210.95.81","arvedtk_root","Ankara@06.","arvedtk_hello_next_api")

iter=0

#rows = cur.fetchall()
urlGeneral = "http://arved.tk/uploads/"
while(True):
    cur = con.cursor()
    cur.execute("SELECT * FROM images")

    cur2 = con.cursor()

    flagForStaps = 0;
    rows = cur.fetchall()
    for row in rows:
        print("{0} {1} {2} {3} {4} {5}".format(row[0], row[1], row[2],row[3],row[4], row[5]))
        if(row[2]=="1"):
            url = urlGeneral+row[1]
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgNp, -1)
            #resized = cv2.resize(frame, None, fx=0.400, fy=0.400, interpolation=cv2.INTER_AREA)
            tcnum=Object_detection_webcam.ProcessingFunction(frame,flagForStaps)

            #cv2.imshow('Object detector', resized)
            #cv2.waitKey(0)
            sql_update_query_flag = 'Update images set flag = 2 where id = '
            sql_update_query_flag= sql_update_query_flag+str(row[0])
            cur.execute(sql_update_query_flag)
            if(tcnum=="okunamadi"):
                sql_update_query_message = 'Update images set message = "kotu fotograf" where id = '
                sql_update_query_message = sql_update_query_message + str(row[0])
                cur.execute(sql_update_query_message)

                sql_update_query_tc = 'Update images set tc = "' + tcnum + '" where id = '
                sql_update_query_tc = sql_update_query_tc + str(row[0])
                cur.execute(sql_update_query_tc)

            elif (tcnum=="crop_img error" or tcnum == "tc bulunamadi"):
                while(iter<3 and (tcnum=="crop_img error" or tcnum == "tc bulunamadi")):

                    rotated = np.rot90(frame)
                    tcnum = Object_detection_webcam.ProcessingFunction(rotated, flagForStaps)
                    iter +=1

                    if(tcnum=="okunamadi"):
                        sql_update_query_message = 'Update images set message = "kotu fotograf" where id = '
                        sql_update_query_message = sql_update_query_message + str(row[0])
                        cur.execute(sql_update_query_message)

                        sql_update_query_tc = 'Update images set tc = "' + tcnum + '" where id = '
                        sql_update_query_tc = sql_update_query_tc + str(row[0])
                        cur.execute(sql_update_query_tc)

                        break

                    elif(tcnum!="crop_img error" and tcnum != "tc bulunamadi"):
                        sql_update_query_message = 'Update images set message = "okundu xxx" where id = '
                        sql_update_query_message = sql_update_query_message + str(row[0])
                        cur.execute(sql_update_query_message)

                        sql_update_query_tc = 'Update images set tc = "' + tcnum + '" where id = '
                        sql_update_query_tc = sql_update_query_tc + str(row[0])
                        cur.execute(sql_update_query_tc)

                        cur2.execute("SELECT tc FROM inquiries WHERE tc = '%s'" %(tcnum))

                        if cur2.fetchone():

                            sql_update_query_inquiry_results = 'Update images set inquiry_results = "gercek kimlik" where id = '
                            sql_update_query_inquiry_results = sql_update_query_inquiry_results + str(row[0])
                            cur.execute(sql_update_query_inquiry_results)

                        else:
                            sql_update_query_inquiry_results = 'Update images set inquiry_results = "sahte kimlik" where id = '
                            sql_update_query_inquiry_results = sql_update_query_inquiry_results + str(row[0])
                            cur.execute(sql_update_query_inquiry_results)

                        break

                    else:
                        sql_update_query_message = 'Update images set message = "' + "hala kotu fotograf" + '" where id = '
                        sql_update_query_message = sql_update_query_message + str(row[0])
                        cur.execute(sql_update_query_message)

                        sql_update_query_tc = 'Update images set tc = "' + "yine okunamadi" + '" where id = '
                        sql_update_query_tc = sql_update_query_tc + str(row[0])
                        cur.execute(sql_update_query_tc)

                iter = 0
            else:
                sql_update_query_message = 'Update images set message = "okundu" where id = '
                sql_update_query_message = sql_update_query_message + str(row[0])
                cur.execute(sql_update_query_message)

                sql_update_query_tc = 'Update images set tc = "'+tcnum+'" where id = '
                sql_update_query_tc = sql_update_query_tc + str(row[0])
                cur.execute(sql_update_query_tc)

                cur2.execute("SELECT tc FROM inquiries WHERE tc = '%s'" %(tcnum))

                if cur2.fetchone():

                    sql_update_query_inquiry_results = 'Update images set inquiry_results = "gercek kimlik" where id = '
                    sql_update_query_inquiry_results = sql_update_query_inquiry_results + str(row[0])
                    cur.execute(sql_update_query_inquiry_results)

                else:
                    sql_update_query_inquiry_results = 'Update images set inquiry_results = "sahte kimlik" where id = '
                    sql_update_query_inquiry_results = sql_update_query_inquiry_results + str(row[0])
                    cur.execute(sql_update_query_inquiry_results)



    time.sleep(5)





