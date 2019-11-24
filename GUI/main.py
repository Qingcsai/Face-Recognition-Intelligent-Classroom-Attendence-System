# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:48:22 2019

@author: ASUS
"""

import sys
sys.path.append('F:/Face_Recognition/FaceNet/facenet-master/src')
sys.path.append('F:/Face_Recognition/FaceNet/facenet-master/contributed')
sys.path.append('F:/Face_Recognition/FaceNet/facenet-master/src/align')
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5 import uic,QtWidgets,QtSql,QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from random import randint
#from PIL import Image
from keras import backend as K
#import numpy as np
import cv2
import os
import face
import my_use_detect
import my_train_classifier
import time
import sqlite3

Ui_MainWindow, QtBaseClass = uic.loadUiType('read_road.ui')


class Student:
    def __init__(self):
        self.face = None
        self.name = None
#        self.image = None
        self.best_prediction = None
        self.face_time = None
        self.number = None
        self.classes = None
        
    def __str__(self):
        return '姓名:{} 班级:{} 准确率:{} 学号:{}'.format(self.name, self.classes, self.best_prediction, self.number)
        
class Class:
  
    def __init__(self,name):
        self.name = name 
        self.stu_list = []
        self.stu_name = []
        self.stu_count = 0  
    
    def add_stu(self, stu):
        if str(stu.name) in self.stu_name:
            pass
        else:
            self.stu_list.append(stu)
            self.stu_name.append(stu.name)
            self.stu_count += 1
        
    def show_stu(self):
        print("###")
        for s in self.stu_list:
            print(s)
        
    
    def count_stu(self):
        return self.stu_count
            
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        
        # Set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        
        self.slot_init()    #拍照
        self.detect_init()  #预处理
        self.train_init()   #训练分类器
        self.recog_init()
        self.Attend_init()
        self.sys_init()
        
    def sys_init(self):
        self.ui.btn_sys_close.clicked.connect(self.sys_close)
    
    def train_init(self):
        self.ui.btn_train_path.clicked.connect(self.setTrainPath)
        self.ui.btn_train.clicked.connect(self.btn_train_click)
        
    def detect_init(self): 
        self.ui.btn_detect_path.clicked.connect(self.setDetectPath)
        self.ui.btn_detect.clicked.connect(self.btn_detect_click)
#    def save_cam_init(self):
#        self.ui.btn_save_cam.clicked.connect(self.btn_save_cam_click)

    def slot_init(self):
        self.timer_camera1 = QTimer()
        self.ui.btn_path.clicked.connect(self.setSavePath)
        self.ui.btn_open_camera.clicked.connect(self.open_camera_slot)
        self.timer_camera1.timeout.connect(self.show_camera)
        self.ui.btn_insert.clicked.connect(self.btn_save_cam_click)

        
    def recog_init(self):
        self.timer_camera2 = QTimer()
        self.frame_interval = 3  # Number of frames after which to run face detection
        self.fps_display_interval = 5  # seconds
        self.frame_rate = 0
        self.frame_count = 0
        
        self.ui.btn_camera_recog.clicked.connect(self.open_camera_recog)
        self.timer_camera2.timeout.connect(self.show_recog)
        
    def Attend_init(self):
#        self.stu = Student()     #实例化
        self.C = Class('162')
        self.ui.btn_Attendance.clicked.connect(self.showtable)
        self.ui.btn_Attendance.clicked.connect(self.view_data)
        self.ui.btn_insert.clicked.connect(self.insert_data)  
        self.ui.btn_create_data.clicked.connect(self.create_regist_db)        
        self.ui.btn_save_Attendance.clicked.connect(self.create_attend_db)

        
    def sys_close(self):
        K.clear_session()
        print('系统已正常退出')
        
    def create_regist_db(self):
        try:
            # 调用输入框获取数据库名称
            self.db_text,db_action = QtWidgets.QInputDialog.getText(self,'保存数据库名称','请输入课程名称',QtWidgets.QLineEdit.Normal)
            
            self.ui.label_show_course.setText(self.db_text)
            if db_action is True:
                print(self.db_text)
                # 添加一个sqlite数据库连接并打开
#                self.db_tablename,db_action = QtWidgets.QInputDialog.getText(self,'保存数据表名称','请输入课时信息',QtWidgets.QLineEdit.Normal)
                self.db_tablename = 'name_list'
                self.conn = sqlite3.connect('{}.db'.format(self.db_text))
                # 创建一个Cursor:
                self.cursor = self.conn.cursor()
                # 执行一条SQL语句，创建user表:
                #先判断表是否存在
#                self.cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table' AND name = '{}'".format(self.db_tablename))
#                if self.cursor.getInt(0)==0:
                self.cursor.execute("create table if not exists {}(学号 int primary key,"
                               "姓名 varchar(20),"
                               "班级 varchar(100),"
                               "是否到勤 INTEGER DEFAULT (0))".format(self.db_tablename))
                print('创建数据库成功！')   

                # 关闭Cursor:
                self.cursor.close()
                # 提交事务:
                self.conn.commit()
                # 关闭Connection:
                self.conn.close()


#                self.db.close()
        except Exception as e:
            print(e)
    
    def insert_data(self):
        stu_name = str(self.ui.text_name.toPlainText())
        stu_classes = str(self.ui.text_classes.toPlainText())
        stu_num = self.ui.text_num.toPlainText()
        self.conn = sqlite3.connect('{}.db'.format(self.db_text))
                # 创建一个Cursor:
        self.cursor = self.conn.cursor()
#        self.cursor.execute("insert into {} values(152,'测试测试','00级宇宙班',1)".format(self.db_tablename))  #不能省略默认值
        self.cursor.execute("insert into {} values('{}','{}','{}',0)".format(self.db_tablename,stu_num,stu_name,stu_classes))
        print('添加成员成功！')
        self.cursor.close()
                # 提交事务:
        self.conn.commit()
                # 关闭Connection:
        self.conn.close()
        
    def view_data(self):
        # 实例化一个可编辑数据模型
        db = QtSql.QSqlDatabase.addDatabase('QSQLITE')
            
#        db_text,db_action = QtWidgets.QInputDialog.getText(self,'查询数据库名称','请输入课程名称',QtWidgets.QLineEdit.Normal)
#        db_tablename,db_action = QtWidgets.QInputDialog.getText(self,'查询数据表名称','请输入课时信息',QtWidgets.QLineEdit.Normal)
        db_text = self.ui.label_cla.text()
        db_tablename = self.ui.label_cla_time.text()
        db.setDatabaseName('{}.db'.format(db_text))
#        self.db = QtSql.QSqlDatabase.database('{}.db'.format(self.db_text))
        db.open()
#        db_text,db_action = QtWidgets.QInputDialog.getText(self,'打开数据表','请输入数据表名23称',QtWidgets.QLineEdit.Normal)
        self.model = QtSql.QSqlTableModel()
        
        self.model.setTable('{}'.format(db_tablename)) # 设置数据模型的数据表
        self.model.setEditStrategy(QtSql.QSqlTableModel.OnFieldChange) # 允许字段更改
#        self.model.select() # 查询所有数据
        # 设置表格头
        self.model.setHeaderData(0, QtCore.Qt.Horizontal,'学号')
        self.model.setHeaderData(1, QtCore.Qt.Horizontal, '姓名')
        self.model.setHeaderData(2, QtCore.Qt.Horizontal, '班级')
        self.model.setHeaderData(3, QtCore.Qt.Horizontal, '时间')
        self.model.setHeaderData(4, QtCore.Qt.Horizontal,'准确率')
        self.model.setHeaderData(5, QtCore.Qt.Horizontal, '是否到勤')

        self.model.select()
        view = self.ui.table_data_view
        view.setModel(self.model)
        view.setWindowTitle('数据表显示')
        db.close()
        
    def create_attend_db(self):
        try:
            # 调用输入框获取数据库名称
#            db_text,db_action = QtWidgets.QInputDialog.getText(self,'保存数据库名称','请输入课程名称',QtWidgets.QLineEdit.Normal)
            db_text = self.ui.label_show_course.text()
            db_action = True
            self.ui.label_show_course.setText(db_text)
            self.ui.label_cla.setText(db_text)
            if db_action is True:
                print(db_text)
                # 添加一个sqlite数据库连接并打开
                db_tablename,db_action = QtWidgets.QInputDialog.getText(self,'保存数据表名称','请输入课时信息',QtWidgets.QLineEdit.Normal)
                self.ui.label_cla_time.setText(db_tablename)
                self.conn = sqlite3.connect('{}.db'.format(db_text))
                # 创建一个Cursor:
                self.cursor = self.conn.cursor()
                # 执行一条SQL语句，创建user表:
                
                self.cursor.execute("create table {}(学号 int primary key,"
                               "姓名 varchar(20),"
                               "班级 varchar(100),"
                               "时间 varchar(50),"
                               "准确率 varchar(20),"
                               "是否到勤 INTEGER DEFAULT (0))".format(db_tablename))
        
                ####
                for i in range(len(self.C.stu_list)):
                    ID = str(self.C.stu_list[i].number)
                    db_stu_name = self.C.stu_list[i].name
                    classes = self.C.stu_list[i].classes
                    face_time = str(self.C.stu_list[i].face_time)
                    best_prediction = str(self.C.stu_list[i].best_prediction)
                    
                    sb_is_on_time = 1
                    face = i
                    
#                   face = QIcon(Image.fromarray(stu.face))
                    self.cursor.execute("insert into {} values('{}','{}','{}','{}','{}',1)".format(db_tablename, ID, db_stu_name, classes, face_time, best_prediction))
                # 关闭Cursor:
                self.cursor.close()
                # 提交事务:
                self.conn.commit()
                # 关闭Connection:
                self.conn.close()
                print('保存考勤记录成功！')
#                self.db.close()
        except Exception as e:
            print(e)        
        

    def showtable(self):
        list_rows = len(self.C.stu_list)
#        table_rows = self.ui.table_Attendance.rowCount()
#        self.ui.label_count.setText("共检测到 %d 同学到" + str(self.C.stu_name[:])+ "\n"  + str(self.C.stu_list[0].name) + " " + str(self.C.stu_list[1].name) + " " + str(self.C.stu_list[2].name) + "\n" + str(list_rows) +" " + str(table_rows) % self.C.count_stu)
        self.ui.label_count.setText("共检测到%d 同学" % list_rows)

        self.C.show_stu()
#        elif table_rows > 0 and list_rows > 0:
#            self.removeRows(table_rows)
            
    def setSavePath(self):
        save_path = os.path.join("F:/Face_Recognition/FaceNet/facenet-master/data/", str(self.ui.label_show_course.text()))
        self.ui.text_path.setText(save_path) 
        
    def setDetectPath(self): 
        detect_path = QFileDialog.getExistingDirectory(self,   
                                    "浏览",  
                                    "F:/Face_Recognition/FaceNet/facenet-master/data")   
        self.ui.text_detect_path.setText(detect_path)
        
    def setTrainPath(self): 
        '''
        train_path = QFileDialog.getExistingDirectory(self,  
                                    "浏览",  
                                    "F:/Face_Recognition/FaceNet/facenet-master/data")   
        '''
        train_path = self.ui.label_detect_output.text()  #Qlabel attri is text
        self.ui.text_train_path.setText(train_path)
        
    def btn_train_click(self):
        data_dir = self.ui.text_train_path.toPlainText()   
        classifier_filename = os.path.join('F:/Face_Recognition\FaceNet/facenet-master/pre-trained-model/','{}.pkl'.format(str(self.ui.label_show_course.text())))
        self.ui.label_train_output.setText(classifier_filename)
        my_train_classifier.train(data_dir, classifier_filename) 
        QMessageBox.information(self, "Information", self.tr("训练成功!"))
        
    def btn_detect_click(self):
        detect_path = self.ui.text_detect_path.toPlainText()
#        output_path = 'F:/Face_Recognition/FaceNet/facenet-master/data/scutdata_160'
        output_path = os.path.join("F:/Face_Recognition/FaceNet/facenet-master/data/", str(self.ui.label_show_course.text())+'_160')
        self.ui.label_detect_output.setText(output_path)
        my_use_detect.detect(detect_path, output_path)
        QMessageBox.information(self, "Information", self.tr("预处理成功!"))
        
    def btn_save_cam_click(self):
        save_path = self.ui.text_path.toPlainText()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
#        else:
        new_name_file = str(self.ui.text_name.toPlainText())
        print(new_name_file)
        photo_save_path = os.path.join(save_path,new_name_file)
        if not os.path.exists(photo_save_path):
            os.makedirs(photo_save_path)

        print(photo_save_path)
        dirs = os.listdir(photo_save_path)
        num_i = len(dirs) + 1
        print(dirs)
        show = cv2.resize(self.image, (self.image.shape[1],self.image.shape[0]))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB, self.image)  # 这里指的是显示原图
#            cv2.imwrite(os.path.join(photo_save_path,"%s_%d.jpg" % (new_name_file,num_i)),show)
        cv2.imencode('.jpg', show)[1].tofile(os.path.join(photo_save_path,"%s_%d.jpg" % (new_name_file,num_i)))  ##带有中文路径保存方法
            #self.showImage.save(photo_save_path +"%s.jpg" % new_name_file)
        QMessageBox.information(self, "Information", self.tr("保存成功!"))
        
    def open_camera_slot(self):
        self.cap = cv2.VideoCapture(0)      #开启摄像头

        self.cap_num = 0
        if self.timer_camera1.isActive() == False:
            flag = self.cap.open(self.cap_num)
            if flag == False:
                msg = QMessageBox.warning(self, "Warning", "请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
                if msg == QMessageBox.Cancel:
                    pass
            else:
                self.timer_camera1.start(20) #刷新时间
                self.ui.btn_open_camera.setText('关闭摄像头')
        else:
            self.timer_camera1.stop()
            self.cap.release()
            self.ui.label_camera.clear()
            self.ui.btn_open_camera.setText('开启摄像头')

    
    def open_camera_recog(self):
        classifier_model = os.path.join('F:/Face_Recognition/FaceNet/facenet-master/pre-trained-model/','{}.pkl'.format(str(self.ui.label_show_course.text())))
        print(classifier_model)
        print("正在打开识别系统...")
        self.face_recognition = face.Recognition(classifier_model)
#        self.cap = cv2.VideoCapture(0)      #开启摄像头
        
        self.cap = cv2.VideoCapture('F:/Face_Recognition/FaceNet/facenet-master/data/video/TDWY.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('F:/Face_Recognition/FaceNet/facenet-master/data/video/TD_result5.avi',fourcc, 20.0, (848,478))  #尺寸要和原视频一样,
#        self.cap_num = 0
        if self.timer_camera2.isActive() == False:
#            flag = self.cap.open(self.cap_num)
            flag = True
            if flag == False:
                msg = QMessageBox.warning(self, "Warning", "请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
                if msg == QMessageBox.Cancel:
                    pass
            else:
                self.timer_camera2.start(20) #刷新时间
                self.ui.btn_camera_recog.setText('停止识别')
        else:
            self.timer_camera2.stop()
            self.cap.release()
            self.ui.label_camera_recog.clear()
            self.ui.btn_camera_recog.setText('开始识别')
    
    def add_overlays(self,frame, faces, frame_rate):
        if faces is not None:
            for face1 in faces:
                face_bb = face1.bounding_box.astype(int)
                cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
                if face1.name is not None:
                    cv2.putText(frame, str(face1.name)+" "+str(float('%.2f' % face1.best_prediction)), (face_bb[0], face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                thickness=1, lineType=2)
       
#        cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
#                    thickness=2, lineType=2)     ##cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
          

    def show_recog(self):
######
        if self.cap.isOpened():
###
            ret, self.image = self.cap.read()
            if ret==True:
                start_time = time.time()
                if (self.frame_count % self.frame_interval) == 0:    # % 表示取余数  这行代码意思是每隔三帧运行一次人脸检测
                    self.faces = self.face_recognition.identify(self.image)
                    # Check our current fps
                    end_time = time.time() 
                    if (end_time - start_time) > self.fps_display_interval:
                        self.frame_rate = int(self.frame_count / (end_time - start_time))
                        start_time = time.time()
                        self.frame_count = 0
                    
                    for i, single_face in enumerate(self.faces):
                        self.stu = Student()     #实例化  必须放在循环内，否则每次list的值将被覆盖，详解见https://www.cnblogs.com/iqunqunqun/p/9249888.html
                        self.stu.name = single_face.name
                        self.stu.best_prediction = float('%.2f' % single_face.best_prediction)
                        self.stu.face = single_face.image
                        self.stu.face_time = time.asctime( time.localtime(time.time()) )
                        self.stu.classes = self.C.name
                        self.stu.number = randint(1,1000)
                        self.C.add_stu(self.stu)
                        self.C.show_stu()
                        
                self.add_overlays(self.image, self.faces, self.frame_rate)
                self.frame_count += 1
                
                self.out.write(self.image)
                
                show = cv2.resize(self.image, (self.image.shape[1],self.image.shape[0]))
                show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB, self.image)  # 这里指的是显示原图

                # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                # int height, Format format, QImageCleanupFunction cleanupFunction = 0, void *cleanupInfo = 0)
                self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                self.ui.label_camera_recog.setScaledContents(True)
                self.ui.label_camera_recog.setPixmap(QPixmap.fromImage(self.showImage))

            else:
                self.timer_camera2.stop()
                self.cap.release()
                self.ui.label_camera_recog.clear()
                self.out.release()
                self.ui.btn_camera_recog.setText('开始识别')
        else:        
            self.timer_camera2.stop()
            self.cap.release()
            self.ui.label_camera_recog.clear()
            self.out.release()
            self.ui.btn_camera_recog.setText('开始识别')
#### 
    def show_camera(self):
        ret, self.image = self.cap.read()
        if ret==True:
            show = cv2.resize(self.image, (self.image.shape[1],self.image.shape[0]))
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB, self.image)  # 这里指的是显示原图
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            # int height, Format format, QImageCleanupFunction cleanupFunction = 0, void *cleanupInfo = 0)
            self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
            self.ui.label_camera.setScaledContents(True)
            self.ui.label_camera.setPixmap(QPixmap.fromImage(self.showImage))
        else:
            self.timer_camera1.stop()
            self.cap.release()
            self.ui.label_camera.clear()

        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
