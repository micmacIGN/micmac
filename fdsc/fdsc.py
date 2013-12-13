#! /usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
#FDiSC v0.9                                                              #
#Fault DIsplacement Slip-Curve                                           #
#                                                                        #
#Copyright (C) (2013) Ana-Maria Rosu, IPGP-IGN project financed by CNES  #
#am.rosu@laposte.net                                                     #
#                                                                        #
#This software is governed by the CeCILL-B license under French law and  #
#abiding by the rules of distribution of free software.  You can  use,   #
#modify and/ or redistribute the software under the terms of the CeCILL-B#
#license as circulated by CEA, CNRS and INRIA at the following URL       #
#"http://www.cecill.info".                                               #
##########################################################################


import sys
import os
import os.path
import re
from pylab import *
from osgeo import gdal
from PyQt4 import QtGui, QtCore
from ConstructLine import *
from fonctions_cg import *


class MainWindow(QtGui.QMainWindow):
  def __init__(self, parent=None):
    super(MainWindow, self).__init__(parent)
    self.initUI()

    #Widgets
    self.groupBox_poly = QtGui.QGroupBox('Polyline describing the fault .')
    self.groupBox_poly.setCheckable(True)
    self.groupBox_poly.setChecked(False)

    self.groupBox_stack = QtGui.QGroupBox('Stacks of profiles')
    self.groupBox_stack.setCheckable(True)
    self.groupBox_stack.setChecked(False)

    self.groupBox_cg = QtGui.QGroupBox('Slip-curve')
    self.groupBox_cg.setCheckable(True)
    self.groupBox_cg.setChecked(False)

    layout_poly = QtGui.QGridLayout()
    layout_poly.setSpacing(10)

    layout_stack = QtGui.QGridLayout()
    layout_stack.setSpacing(10)

    layout_cg = QtGui.QGridLayout()
    layout_cg.setSpacing(10)

    self.groupBox_poly.setLayout(layout_poly)
    self.groupBox_stack.setLayout(layout_stack)
    self.groupBox_cg.setLayout(layout_cg)

    #Create central widget, add layout and set
    layout_global = QtGui.QGridLayout()
    central_widget = QtGui.QWidget()
    central_widget.setLayout(layout_global)
    self.setCentralWidget(central_widget)

    layout_global.addWidget(self.groupBox_poly)
    layout_global.addWidget(self.groupBox_stack)
    layout_global.addWidget(self.groupBox_cg)

    ## polyline section
    self.label_imageIn = QtGui.QLabel("Parallax image file:")
    self.button_imageIn = QtGui.QPushButton("...")
    self.edit_filename_imageIn = QtGui.QLineEdit()
        #--------
    self.label_traceOut= QtGui.QLabel("Fault trace Out:")
    self.button_traceOut= QtGui.QPushButton("...")
    self.edit_filename_traceOut = QtGui.QLineEdit()
        #--------
    self.button_ok=QtGui.QPushButton("Go drawing")
    self.button_ok.setEnabled(False)

    #stack section
    self.label_imageInStack = QtGui.QLabel("Parallax image file:")
    self.button_imageInStack = QtGui.QPushButton("...")
    self.edit_filename_imageInStack = QtGui.QLineEdit()
        #--------
    self.label_imageWeightStack = QtGui.QLabel("Weight image file:")
    self.button_imageWeightStack = QtGui.QPushButton("...")
    self.edit_filename_imageWeightStack = QtGui.QLineEdit()
        #--------
    self.label_traceFileStack = QtGui.QLabel("Fault trace file:")
    self.button_traceFileStack = QtGui.QPushButton("...")
    self.edit_traceFileStack = QtGui.QLineEdit()
        #--------
    self.label_methodStack = QtGui.QLabel("Method:")
    self.radio_medianStack=QtGui.QRadioButton("Median")
    self.radio_medianStack.setEnabled(True)
    self.radio_meanStack=QtGui.QRadioButton("Mean")
    self.radio_meanStack.setEnabled(True)
    #self.radio_meanStack.setChecked(True)
    self.radio_WmedianStack=QtGui.QRadioButton("Weighted median")
    self.radio_WmedianStack.setEnabled(True)
    self.radio_WmedianStack.setChecked(True)
    self.radio_WmeanStack=QtGui.QRadioButton("Weighted mean")
    self.radio_WmeanStack.setEnabled(True)
        #--------
    self.label_expOfWeightStack = QtGui.QLabel("Exponent of weights:")
    self.spin_expOfWeightStack=QtGui.QSpinBox()
    self.spin_expOfWeightStack.setValue(6)
    self.spin_expOfWeightStack.setMinimum(1)
    #self.spin_expOfWeightStack.setEnabled(True)
        #--------
    self.label_lengthStack = QtGui.QLabel("Length (odd):")
    self.spin_lengthStack=QtGui.QSpinBox()
    self.spin_lengthStack.setMinimum(1)
    self.spin_lengthStack.setMaximum(100000) #!!!!
    self.spin_lengthStack.setValue(61)
    self.spin_lengthStack.setEnabled(True)
        #--------
    self.label_widthStack = QtGui.QLabel("Width (odd):")
    self.spin_widthStack=QtGui.QSpinBox()
    self.spin_widthStack.setMinimum(1)
    self.spin_widthStack.setMaximum(100000) #!!!!
    self.spin_widthStack.setValue(41)
    self.spin_widthStack.setEnabled(True)
        #--------
    self.label_distStack = QtGui.QLabel("Distance between stacks:") #distance between central profiles of stacks
    self.spin_distStack=QtGui.QSpinBox()
    self.spin_distStack.setMinimum(1)
    self.spin_distStack.setMaximum(100000) #!!!!
    self.spin_distStack.setValue(200)
    self.spin_distStack.setEnabled(True)
        #--------
    self.check_saveStack=QtGui.QCheckBox("Save stacks") ## !!!! layout and connexion
    self.check_saveStack.setEnabled(True)
    self.check_saveStack.setChecked(True)
        #--------
    self.label_resolStack=QtGui.QLabel("Initial resolution (1px=?m):")
    self.label_resolStack.setToolTip("Resolution of the images used for correlation")
    self.spin_resolStack=QtGui.QDoubleSpinBox()
    self.spin_resolStack.setToolTip("Resolution of the images used for correlation")
    self.spin_resolStack.setMinimum(1)
        #--------
    self.label_offsetsOutStack = QtGui.QLabel("Offsets file Out:")
    self.button_offsetsOutStack = QtGui.QPushButton("...")
    self.edit_offsetsOutStack = QtGui.QLineEdit()
        #--------
    self.check_ValidView=QtGui.QCheckBox("View stacks for validation")
    self.check_ValidView.setEnabled(True)
    self.check_ValidView.setChecked(True)
        #--------
    self.check_ValidErrbars=QtGui.QCheckBox("View errorbars on the stacks")
    self.check_ValidErrbars.setEnabled(True)
    self.check_ValidErrbars.setChecked(True)
        #--------
    self.button_okStack=QtGui.QPushButton("OK")
    self.button_okStack.setEnabled(False)

    ## slip-curve section
    self.label_offsetsInCg=QtGui.QLabel("Offsets file:")
    self.edit_offsetsInCg = QtGui.QLineEdit()
    self.button_offsetsInCg = QtGui.QPushButton("...")
        #--------
    self.label_fileOutCg = QtGui.QLabel("Slip-curve file Out:")
    self.edit_fileOutCg = QtGui.QLineEdit()
    self.button_fileOutCg = QtGui.QPushButton("...")
        #--------
    self.button_okCg=QtGui.QPushButton("OK")
    self.button_okCg.setEnabled(False)

    ## polyline layout
    layout_poly.addWidget(self.label_imageIn,1,0,1,1)
    layout_poly.addWidget(self.edit_filename_imageIn,1,1,1,4)
    layout_poly.addWidget(self.button_imageIn,1,5,1,1)
        #--------
    layout_poly.addWidget(self.label_traceOut,2,0,1,1)
    layout_poly.addWidget(self.edit_filename_traceOut,2,1,1,4)
    layout_poly.addWidget(self.button_traceOut,2,5,1,1)
        #--------
    layout_poly.addWidget(self.button_ok,3,4)

    ## stack layout
    layout_stack.addWidget(self.label_imageInStack,6,0,1,1)
    layout_stack.addWidget(self.edit_filename_imageInStack,6,1,1,4)
    layout_stack.addWidget(self.button_imageInStack,6,5,1,1)
        #--------
    layout_stack.addWidget(self.label_traceFileStack,7,0,1,1)
    layout_stack.addWidget(self.edit_traceFileStack,7,1,1,4)
    layout_stack.addWidget(self.button_traceFileStack,7,5,1,1)
        #--------
    layout_stack.addWidget(self.label_methodStack,8,0,1,1)
    layout_stack.addWidget(self.radio_medianStack,8,1,1,1)
    layout_stack.addWidget(self.radio_WmedianStack,8,2,1,1)
    layout_stack.addWidget(self.radio_meanStack,8,3,1,1)
    layout_stack.addWidget(self.radio_WmeanStack,8,4,1,1)
        #--------
    layout_stack.addWidget(self.label_imageWeightStack,9,0,1,1)
    layout_stack.addWidget(self.edit_filename_imageWeightStack,9,1,1,4)
    layout_stack.addWidget(self.button_imageWeightStack,9,5,1,1)
    layout_stack.addWidget(self.label_expOfWeightStack,10,0,1,1)
    layout_stack.addWidget(self.spin_expOfWeightStack,10,1,1,1)
        #--------
    layout_stack.addWidget(self.label_lengthStack,11,0,1,1)
    layout_stack.addWidget(self.spin_lengthStack,11,1,1,1)
    layout_stack.addWidget(self.label_widthStack,11,3,1,1)
    layout_stack.addWidget(self.spin_widthStack,11,4,1,1)
    layout_stack.addWidget(self.label_distStack,12,0,1,1)
    layout_stack.addWidget(self.spin_distStack,12,1,1,1)
    layout_stack.addWidget(self.label_resolStack,13,0,1,1)
    layout_stack.addWidget(self.spin_resolStack,13,1,1,1)
    layout_stack.addWidget(self.label_offsetsOutStack,14,0,1,1)
    layout_stack.addWidget(self.edit_offsetsOutStack,14,1,1,4)
    layout_stack.addWidget(self.button_offsetsOutStack,14,5,1,1)
        #--------
    layout_stack.addWidget(self.check_ValidErrbars,15,4,1,1)
    layout_stack.addWidget(self.check_ValidView,16,4,1,1)
    layout_stack.addWidget(self.check_saveStack,17,4,1,1)
    layout_stack.addWidget(self.button_okStack,18,4)

    ## slip-curve layout
    layout_cg.addWidget(self.label_offsetsInCg,18,0,1,1)
    layout_cg.addWidget(self.edit_offsetsInCg,18,1,1,4)
    layout_cg.addWidget(self.button_offsetsInCg,18,5,1,1)
        #--------
    layout_cg.addWidget(self.label_fileOutCg,19,0,1,1)
    layout_cg.addWidget(self.edit_fileOutCg,19,1,1,4)
    layout_cg.addWidget(self.button_fileOutCg,19,5,1,1)
        #--------
    layout_cg.addWidget(self.button_okCg,20,4)

    #Connect signal
    self.connect(self.button_imageIn, QtCore.SIGNAL("clicked()"), self.ask_imageIn_file)
    self.connect(self.button_traceOut, QtCore.SIGNAL("clicked()"), self.ask_traceOut_file)
    self.connect(self.edit_filename_imageIn, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOk)
    self.connect(self.edit_filename_traceOut, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOk)
    self.connect(self.button_ok, QtCore.SIGNAL("clicked()"), self.click_ok)
      #-------
    self.connect(self.button_imageInStack,QtCore.SIGNAL("clicked()"), self.ask_imageInStack_file)
    self.connect(self.button_imageWeightStack,QtCore.SIGNAL("clicked()"), self.ask_imageWeightStack_file)
    self.connect(self.button_traceFileStack,QtCore.SIGNAL("clicked()"), self.ask_traceFileStack_file)
    self.connect(self.button_offsetsOutStack,QtCore.SIGNAL("clicked()"), self.ask_offsetsOutStack_file)
    self.connect(self.edit_filename_imageInStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_filename_imageInStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_filename_imageWeightStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_traceFileStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_offsetsOutStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.button_okStack, QtCore.SIGNAL("clicked()"), self.click_okStack)

    self.connect(self.radio_medianStack,QtCore.SIGNAL("clicked()"), self.update_Weight_input)
    self.connect(self.radio_meanStack,QtCore.SIGNAL("clicked()"), self.update_Weight_input)
    self.connect(self.radio_WmedianStack,QtCore.SIGNAL("clicked()"), self.update_Weight_input)
    self.connect(self.radio_WmeanStack,QtCore.SIGNAL("clicked()"), self.update_Weight_input)
    self.connect(self.groupBox_stack,QtCore.SIGNAL("clicked()"), self.update_Weight_input)
      #-------
    self.connect(self.button_offsetsInCg,QtCore.SIGNAL("clicked()"), self.ask_offsetsInCg_file)
    self.connect(self.button_fileOutCg,QtCore.SIGNAL("clicked()"), self.ask_Cg_file)
    self.connect(self.edit_offsetsInCg, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkCg)
    self.connect(self.edit_fileOutCg, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkCg)
    self.connect(self.button_okCg, QtCore.SIGNAL("clicked()"), self.click_okCg)

    #data
    self.filename_imageIn=""
    self.filename_traceOut=""
    self.filename_imageInStack=""
    self.filename_imageWeightStack=""
    self.filename_traceFaultStack=""
    self.filename_offsetsOutStack=""
    self.filename_offsetsInCg=""
    self.filename_Cg=""

    #self.update_Weight_input()

  def initUI(self):
    self.setGeometry(100, 100, 600, 500)
    self.setWindowTitle('FDSC')
    self.center()
    self.show()

  def center(self):
    qr = self.frameGeometry()
    cp = QtGui.QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    self.move(qr.topLeft())

  def ask_imageIn_file(self):
    self.filename_imageIn=QtGui.QFileDialog.getOpenFileName(self,"select image filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_imageIn!=""):
      self.edit_filename_imageIn.setText(self.filename_imageIn)

  def ask_traceOut_file(self):
    self.filename_traceOut=QtGui.QFileDialog.getSaveFileName(self,"select output filename","trace.txt","Text files (*.txt);;All files (*)")
    if (self.filename_traceOut!=""):
      self.edit_filename_traceOut.setText(self.filename_traceOut)

  def active_buttonOk(self):
    self.filename_imageIn=str(self.edit_filename_imageIn.text())
    self.filename_traceOut=str(self.edit_filename_traceOut.text())
    if (len(self.filename_imageIn)>0) and (len(self.filename_traceOut)>0):
      print 'Image in: ',self.filename_imageIn
      print 'Trace file out: ',self.filename_traceOut
      self.button_ok.setEnabled(True)
    else:
      self.button_ok.setEnabled(False)

  def click_ok(self):
    fig=figure()
    ax = fig.add_subplot(111)
    self.constrLine=ConstructLine(fig,ax,self.filename_imageIn,self.filename_traceOut)
    show()
    self.groupBox_poly.setChecked(False)
    self.groupBox_stack.setChecked(True)
    self.edit_traceFileStack.setText(self.filename_traceOut)

  #-------------
  def ask_imageInStack_file(self):
    self.filename_imageInStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Image Filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_imageInStack!=""):
      self.edit_filename_imageInStack.setText(self.filename_imageInStack)

  def ask_imageWeightStack_file(self):
    self.filename_imageWeightStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Weights Image Filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_imageWeightStack!=""):
      self.edit_filename_imageWeightStack.setText(self.filename_imageWeightStack)

  def ask_traceFileStack_file(self):
    self.filename_traceFaultStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Trace Fault Filename","trace.txt","Text files (*.txt);;All files (*)")
    if (self.filename_traceFaultStack!=""):
      self.edit_traceFileStack.setText(self.filename_traceFaultStack)

  def ask_offsetsOutStack_file(self):
    self.filename_offsetsOutStack=QtGui.QFileDialog.getSaveFileName(self,"Stacks: Select Offsets Output Filename","offsets.txt","Text files (*.txt);;All files (*)")
    if (self.filename_offsetsOutStack!=""):
      self.edit_offsetsOutStack.setText(self.filename_offsetsOutStack)


  def active_buttonOkStack(self):
    self.filename_imageInStack=str(self.edit_filename_imageInStack.text())
    self.filename_imageWeightStack=str(self.edit_filename_imageWeightStack.text())
    self.filename_traceFaultStack=str(self.edit_traceFileStack.text())
    self.filename_offsetsOutStack=str(self.edit_offsetsOutStack.text())
    self.compare_fileSize()
    if (len(self.filename_imageInStack)>0) and (len(self.filename_traceFaultStack)>0) and (len(self.filename_offsetsOutStack)>0):
      self.button_okStack.setEnabled(True)
    else:
      self.button_okStack.setEnabled(False)


  def click_okStack(self):
    if (self.spin_lengthStack.value()%2!=0) and (self.spin_widthStack.value()%2!=0):
      showStack=False
      showErr=False
      startName_stack=""
      if self.check_ValidErrbars.isChecked():
        showErr=True
      if self.check_ValidView.isChecked():
        showStack=True
      if self.check_saveStack.isChecked():
        name_abs=os.path.abspath(self.filename_offsetsOutStack)
        separator="." #on cree un objet String (pour pouvoir appeler la methode join() de string)
        name_tmp=os.path.split(name_abs)[1].split(separator)
        startName_stack=separator.join(name_tmp[0:-1])
        startName_stack=os.path.join(os.path.split(name_abs)[0],startName_stack)

      if self.radio_WmedianStack.isChecked():
        ##activate the fields for weights' file and exponent of weights
        stackPerp(self.filename_imageInStack, self.filename_imageWeightStack, self.spin_resolStack.value(), self.spin_expOfWeightStack.value(), self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_mediane_pond,self.filename_traceFaultStack,self.filename_offsetsOutStack,'', 'r', 'distance along profile (px)', 'offset (px)','Weighted Median Stack Profile',startName_stack,showErr,showStack)
      elif self.radio_WmeanStack.isChecked():
        ##activate the fields for weights' file and exponent of weights
        stackPerp(self.filename_imageInStack, self.filename_imageWeightStack, self.spin_resolStack.value(), self.spin_expOfWeightStack.value(), self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_coef_correl,self.filename_traceFaultStack,self.filename_offsetsOutStack,'', 'r', 'distance along profile (px)', 'offset (px)','Weighted Mean Stack Profile',startName_stack,showErr,showStack)
      elif self.radio_medianStack.isChecked():
        stackPerp(self.filename_imageInStack, "",  self.spin_resolStack.value(), 1, self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_mediane_pond,self.filename_traceFaultStack,self.filename_offsetsOutStack,'', 'r', 'distance along profile (px)', 'offset (px)','Median Stack Profile',startName_stack,showErr,showStack)
      elif self.radio_meanStack.isChecked():
        stackPerp(self.filename_imageInStack, "", self.spin_resolStack.value(), 1, self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_coef_correl,self.filename_traceFaultStack,self.filename_offsetsOutStack,'', 'r', 'distance along profile (px)', 'offset (px)','Mean Stack Profile',startName_stack,showErr,showStack)
      self.groupBox_stack.setChecked(False)
      self.groupBox_cg.setChecked(True)
      self.edit_offsetsInCg.setText(self.filename_offsetsOutStack)
    else:
      QtGui.QMessageBox.warning(self, 'Error detected',"The length and the width of stack must be odd numbers!", QtGui.QMessageBox.Ok)


  def compare_fileSize(self):
    if (len(self.filename_imageInStack)>0) and (len(self.filename_imageWeightStack)>0):
      ds_Px=gdal.Open(self.filename_imageInStack, gdal.GA_ReadOnly)
      nb_col_Px=ds_Px.RasterXSize
      nb_lig_Px=ds_Px.RasterYSize
      ds_w=gdal.Open(self.filename_imageWeightStack, gdal.GA_ReadOnly)
      nb_col_w=ds_w.RasterXSize
      nb_lig_w=ds_w.RasterYSize
      if (nb_col_Px!=nb_col_w) or (nb_lig_Px!=nb_lig_w):
        QtGui.QMessageBox.warning(self, 'Error detected',"Different file size for parallax and weight files!!!", QtGui.QMessageBox.Ok)


  def update_Weight_input(self):
    with_weight= self.groupBox_stack.isChecked() and (self.radio_WmedianStack.isChecked() or self.radio_WmeanStack.isChecked())
    self.edit_filename_imageWeightStack.setEnabled(with_weight)
    self.spin_expOfWeightStack.setEnabled(with_weight)

  #---------------
  def ask_offsetsInCg_file(self):
    self.filename_offsetsInCg=QtGui.QFileDialog.getOpenFileName(self,"Select Offsets Info File",".","Text files (*.txt);;All files (*)")
    if (self.filename_offsetsInCg!=""):
      self.edit_offsetsInCg.setText(self.filename_offsetsInCg)

  def ask_Cg_file(self):
    self.filename_Cg=QtGui.QFileDialog.getSaveFileName(self,"Select name of file for slip-curve","slip_curve.png","Images (*.png *.tif *.tiff  *.jpg);;All files (*)")
    if (self.filename_Cg!=""):
      self.edit_fileOutCg.setText(self.filename_Cg)

  def active_buttonOkCg(self):
    self.filename_offsetsInCg=str(self.edit_offsetsInCg.text())
    self.filename_Cg=str(self.edit_fileOutCg.text())
    #~ print self.filename_offsetsInCg,self.filename_Cg
    if (len(self.filename_offsetsInCg)>0) and (len(self.filename_Cg)>0):
      self.button_okCg.setEnabled(True)
    else:
      self.button_okCg.setEnabled(False)

  def click_okCg(self):
    drawCG(self.filename_offsetsInCg, self.filename_Cg)
    self.groupBox_cg.setChecked(False)


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
