#! /usr/bin/python
# -*- coding: utf-8 -*-

##########################################################################
# Fault Displacement Slip-Curve (FDSC) v1.0                              #
#                                                                        #
# Copyright (C) (2013-2014) Ana-Maria Rosu am.rosu@laposte.net           #
# IPGP-ENSG/IGN project financed by TOSCA/CNES                           #
#                                                                        #
#                                                                        #
# This software is governed by the CeCILL-B license under French law and #
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
    print "********************************************************"
    print "*****  Fault Displacement Slip-Curve (FDSC) v1.0   *****"
    print "*** Copyright (C) (2013-2014) Ana-Maria Rosu         ***"
    print "** IPGP-ENSG/IGN project financed by TOSCA/CNES       **"
    print "** This software is governed by the CeCILL-B license. **"
    print "********************************************************"
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
    self.label_Px1imageInStack = QtGui.QLabel("Px1 Parallax image file:")
    self.label_Px1imageInStack.setToolTip("Parallax image in the epipolar direction")
    self.button_Px1imageInStack = QtGui.QPushButton("...")
    self.edit_filename_Px1imageInStack = QtGui.QLineEdit()
    self.label_Px2imageInStack = QtGui.QLabel("Px2 Parallax image file:")
    self.label_Px2imageInStack.setToolTip("Parallax image in the transverse direction")
    self.button_Px2imageInStack = QtGui.QPushButton("...")
    self.edit_filename_Px2imageInStack = QtGui.QLineEdit()
        #---------
    self.label_dirDispStack = QtGui.QLabel("Offsets output direction:")
    #~ self.radio_dirDispCLStack = QtGui.QRadioButton("Column/Line")
    #~ self.radio_dirDispCLStack.setChecked(True)
    #~ self.radio_dirDispPPStack = QtGui.QRadioButton("Parallel/Perpendicular")
    self.cbox_dirDispColStack = QtGui.QCheckBox("Column")
    #~ self.cbox_dirDispColStack.setChecked(True)
    self.cbox_dirDispLineStack = QtGui.QCheckBox("Line")
    self.cbox_dirDispParalStack = QtGui.QCheckBox("Parallel")
    self.cbox_dirDispPerpStack = QtGui.QCheckBox("Perpendicular")
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
    self.spin_resolStack.setMinimum(0.00000001)
    self.spin_resolStack.setValue(1)
    self.spin_resolStack.setSingleStep(0.01)
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
    layout_stack.addWidget(self.label_Px1imageInStack,6,0,1,1)
    layout_stack.addWidget(self.edit_filename_Px1imageInStack,6,1,1,4)
    layout_stack.addWidget(self.button_Px1imageInStack,6,5,1,1)
    layout_stack.addWidget(self.label_Px2imageInStack,7,0,1,1)
    layout_stack.addWidget(self.edit_filename_Px2imageInStack,7,1,1,4)
    layout_stack.addWidget(self.button_Px2imageInStack,7,5,1,1)
        #--------
    layout_stack.addWidget(self.label_traceFileStack,8,0,1,1)
    layout_stack.addWidget(self.edit_traceFileStack,8,1,1,4)
    layout_stack.addWidget(self.button_traceFileStack,8,5,1,1)
        #--------
    layout_stack.addWidget(self.label_methodStack,9,0,1,1)
    layout_stack.addWidget(self.radio_medianStack,9,1,1,1)
    layout_stack.addWidget(self.radio_WmedianStack,9,2,1,1)
    layout_stack.addWidget(self.radio_meanStack,9,3,1,1)
    layout_stack.addWidget(self.radio_WmeanStack,9,4,1,1)
        #--------
    layout_stack.addWidget(self.label_imageWeightStack,10,0,1,1)
    layout_stack.addWidget(self.edit_filename_imageWeightStack,10,1,1,4)
    layout_stack.addWidget(self.button_imageWeightStack,10,5,1,1)
    layout_stack.addWidget(self.label_expOfWeightStack,11,0,1,1)
    layout_stack.addWidget(self.spin_expOfWeightStack,11,1,1,1)
        #--------
    layout_stack.addWidget(self.label_lengthStack,12,0,1,1)
    layout_stack.addWidget(self.spin_lengthStack,12,1,1,1)
    layout_stack.addWidget(self.label_widthStack,12,3,1,1)
    layout_stack.addWidget(self.spin_widthStack,12,4,1,1)
    layout_stack.addWidget(self.label_distStack,13,0,1,1)
    layout_stack.addWidget(self.spin_distStack,13,1,1,1)
    layout_stack.addWidget(self.label_dirDispStack,14,0,1,1)
    layout_stack.addWidget(self.cbox_dirDispColStack,14,1,1,1)
    layout_stack.addWidget(self.cbox_dirDispLineStack,14,2,1,1)
    layout_stack.addWidget(self.cbox_dirDispParalStack,14,3,1,1)
    layout_stack.addWidget(self.cbox_dirDispPerpStack,14,4,1,1)
    layout_stack.addWidget(self.label_resolStack,15,0,1,1)
    layout_stack.addWidget(self.spin_resolStack,15,1,1,1)
    layout_stack.addWidget(self.label_offsetsOutStack,16,0,1,1)
    layout_stack.addWidget(self.edit_offsetsOutStack,16,1,1,4)
    layout_stack.addWidget(self.button_offsetsOutStack,16,5,1,1)
        #--------
    layout_stack.addWidget(self.check_ValidErrbars,17,4,1,1)
    layout_stack.addWidget(self.check_ValidView,18,4,1,1)
    layout_stack.addWidget(self.check_saveStack,19,4,1,1)
    layout_stack.addWidget(self.button_okStack,20,4)

    ## slip-curve layout
    layout_cg.addWidget(self.label_offsetsInCg,21,0,1,1)
    layout_cg.addWidget(self.edit_offsetsInCg,21,1,1,4)
    layout_cg.addWidget(self.button_offsetsInCg,21,5,1,1)
        #--------
    layout_cg.addWidget(self.label_fileOutCg,22,0,1,1)
    layout_cg.addWidget(self.edit_fileOutCg,22,1,1,4)
    layout_cg.addWidget(self.button_fileOutCg,22,5,1,1)
        #--------
    layout_cg.addWidget(self.button_okCg,23,4)

    #Connect signal
    self.connect(self.button_imageIn, QtCore.SIGNAL("clicked()"), self.ask_imageIn_file)
    self.connect(self.button_traceOut, QtCore.SIGNAL("clicked()"), self.ask_traceOut_file)
    self.connect(self.edit_filename_imageIn, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOk)
    self.connect(self.edit_filename_traceOut, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOk)
    self.connect(self.button_ok, QtCore.SIGNAL("clicked()"), self.click_ok)
      #-------
    self.connect(self.button_Px1imageInStack,QtCore.SIGNAL("clicked()"), self.ask_Px1imageInStack_file)
    self.connect(self.button_Px2imageInStack,QtCore.SIGNAL("clicked()"), self.ask_Px2imageInStack_file)
    self.connect(self.button_imageWeightStack,QtCore.SIGNAL("clicked()"), self.ask_imageWeightStack_file)
    self.connect(self.button_traceFileStack,QtCore.SIGNAL("clicked()"), self.ask_traceFileStack_file)
    self.connect(self.button_offsetsOutStack,QtCore.SIGNAL("clicked()"), self.ask_offsetsOutStack_file)
    self.connect(self.edit_filename_Px1imageInStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_filename_Px2imageInStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_filename_imageWeightStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_traceFileStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.edit_offsetsOutStack, QtCore.SIGNAL("textChanged(QString)"),self.active_buttonOkStack)
    self.connect(self.cbox_dirDispColStack, QtCore.SIGNAL("clicked()"),self.select_typeOutputStack)
    self.connect(self.cbox_dirDispLineStack, QtCore.SIGNAL("clicked()"),self.select_typeOutputStack)
    self.connect(self.cbox_dirDispParalStack, QtCore.SIGNAL("clicked()"),self.select_typeOutputStack)
    self.connect(self.cbox_dirDispPerpStack, QtCore.SIGNAL("clicked()"),self.select_typeOutputStack)
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
    self.filename_Px1imageInStack=""
    self.filename_Px2imageInStack=""
    self.filename_imageWeightStack=""
    self.filename_traceFaultStack=""
    self.filename_offsetsOutStack=""
    self.filename_offsetsInCg=""
    self.filename_Cg=""
    self.type_output=[]
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
  def dimIm(self, filepath_im):
    ds = gdal.Open(str(filepath_im), gdal.GA_ReadOnly)
    nb_col=ds.RasterXSize #number of columns of image
    nb_lig=ds.RasterYSize #number of lines of image
    nb_b=ds.RasterCount #number of bands of image
    #~ print 'Image: ',filepath_im, '   ','nb_col: ',nb_col,' nb_lines: ',nb_lig,' nb_b:',nb_b
    return nb_col, nb_lig, nb_b

  def ask_Px1imageInStack_file(self):
    self.filename_Px1imageInStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Px1 Image Filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_Px1imageInStack!=""):
      self.edit_filename_Px1imageInStack.setText(self.filename_Px1imageInStack)
      print "Px1 image: ", self.filename_Px1imageInStack

  def ask_Px2imageInStack_file(self):
    self.filename_Px2imageInStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Px2 Image Filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_Px2imageInStack!=""):
      self.edit_filename_Px2imageInStack.setText(self.filename_Px2imageInStack)
      print "Px2 image: ", self.filename_Px2imageInStack

  def verif_Pximages(self):
    self.filename_Px1imageInStack=str(self.edit_filename_Px1imageInStack.text())
    self.filename_Px2imageInStack=str(self.edit_filename_Px2imageInStack.text())
    if (self.filename_Px1imageInStack!=""):
      if (self.filename_Px2imageInStack!=""):
        finfo1=QtCore.QFileInfo(self.filename_Px1imageInStack)
        finfo2=QtCore.QFileInfo(self.filename_Px2imageInStack)
        if (finfo1.fileName()!=finfo2.fileName()):
          nb_col_im1, nb_lig_im1, nb_b_im1 = self.dimIm(self.filename_Px1imageInStack)
          nb_col_im2, nb_lig_im2, nb_b_im2 = self.dimIm(self.filename_Px2imageInStack)
          if (nb_col_im1!=nb_col_im2 or nb_lig_im1!=nb_lig_im2 or nb_b_im1!=nb_b_im2):
            QtGui.QMessageBox.warning(self, 'Error detected',"Different dimensions for the two parallax images!", QtGui.QMessageBox.Ok)
        else:
          QtGui.QMessageBox.warning(self, 'Warning',"Same parallax image given twice?!", QtGui.QMessageBox.Ok)
    else:
        QtGui.QMessageBox.warning(self, 'Warning',"Two parallax images are expected!", QtGui.QMessageBox.Ok)


  def ask_imageWeightStack_file(self):
    self.filename_imageWeightStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Weights Image Filename",".","Images (*.tif *.tiff *.png *.jpg);;All files (*)")
    if (self.filename_imageWeightStack!=""):
      self.edit_filename_imageWeightStack.setText(self.filename_imageWeightStack)
      print "Weights images: ", self.filename_imageWeightStack


  def ask_traceFileStack_file(self):
    self.filename_traceFaultStack=QtGui.QFileDialog.getOpenFileName(self,"Stacks: Select Trace Fault Filename","trace.txt","Text files (*.txt);;All files (*)")
    if (self.filename_traceFaultStack!=""):
      self.edit_traceFileStack.setText(self.filename_traceFaultStack)
      print "Trace fault file: ", self.filename_traceFaultStack


  def ask_offsetsOutStack_file(self):
    self.filename_offsetsOutStack=QtGui.QFileDialog.getSaveFileName(self,"Stacks: Select Offsets Output Filename","offsets.txt","Text files (*.txt);;All files (*)")
    if (self.filename_offsetsOutStack!=""):
      self.edit_offsetsOutStack.setText(self.filename_offsetsOutStack)
      print "Offsets filename: ", self.filename_offsetsOutStack

  def select_typeOutputStack(self):
    self.type_output=[]
    if self.cbox_dirDispColStack.isChecked():
      self.type_output.append(1)
    if self.cbox_dirDispLineStack.isChecked():
      self.type_output.append(2)
    if self.cbox_dirDispParalStack.isChecked():
      self.type_output.append(3)
    if self.cbox_dirDispPerpStack.isChecked():
      self.type_output.append(4)
    #~ print "type_output: ", self.type_output, "  len: ", len(self.type_output)
    self.active_buttonOkStack()

  def active_buttonOkStack(self):
    self.filename_Px1imageInStack=str(self.edit_filename_Px1imageInStack.text())
    self.filename_Px2imageInStack=str(self.edit_filename_Px2imageInStack.text())
    self.filename_imageWeightStack=str(self.edit_filename_imageWeightStack.text())
    self.filename_traceFaultStack=str(self.edit_traceFileStack.text())
    self.filename_offsetsOutStack=str(self.edit_offsetsOutStack.text())
    if (len(self.filename_Px1imageInStack)>0) and (len(self.filename_Px2imageInStack)>0) and (len(self.filename_traceFaultStack)>0) and (len(self.filename_offsetsOutStack)>0) and (len(self.type_output)>0):
      self.button_okStack.setEnabled(True)
    else:
      self.button_okStack.setEnabled(False)
    self.verif_Pximages()


  def click_okStack(self):
    if (self.spin_lengthStack.value()%2!=0) and (self.spin_widthStack.value()%2!=0):
      showStack=False
      showErr=False
      startName_stack=""
      if self.check_ValidErrbars.isChecked():
        showErr=True
        #~ print "show error bars!"
      if self.check_ValidView.isChecked():
        showStack=True
        #~ print "show stacks for validation!"
      for typeDir in self.type_output:
        typeDir_name=("","dirCol","dirLine","dirParal","dirPerp")[typeDir]
        tmp_abs=os.path.abspath(self.filename_offsetsOutStack)
        #~ print "Offsets abspath: ", tmp_abs
        tmp=os.path.split(tmp_abs)[1].split('.')
        fname_offsetsOutStack=os.path.split(tmp_abs)[0]+'/'+'.'.join(tmp[0:-1])+"_"+typeDir_name+'.'+tmp[-1]
        #~ print "### type dir: ", typeDir, " offsets filename: ", fname_offsetsOutStack
        if self.check_saveStack.isChecked():
          startName_stack=os.path.split(tmp_abs)[0]+'/'+'.'.join(tmp[0:-1])+"_"+typeDir_name
          #~ print "startName_stack: ", startName_stack
        if self.radio_WmedianStack.isChecked():
          ##activate the fields for weights' file and exponent of weights
          stackPerp(self.filename_Px1imageInStack,self.filename_Px2imageInStack, self.filename_imageWeightStack, self.spin_resolStack.value(), self.spin_expOfWeightStack.value(), self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_mediane_pond, self.filename_traceFaultStack,fname_offsetsOutStack,typeDir,'', 'r', 'distance along profile (px)', 'offset (m)','Weighted Median Stack Profile',startName_stack,showErr,showStack)
        elif self.radio_WmeanStack.isChecked():
          ##activate the fields for weights' file and exponent of weights
          stackPerp(self.filename_Px1imageInStack, self.filename_Px2imageInStack, self.filename_imageWeightStack, self.spin_resolStack.value(), self.spin_expOfWeightStack.value(), self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_coef_correl,self.filename_traceFaultStack,fname_offsetsOutStack,typeDir,'', 'r', 'distance along profile (px)', 'offset (m)','Weighted Mean Stack Profile',startName_stack,showErr,showStack)
        elif self.radio_medianStack.isChecked():
          stackPerp(self.filename_Px1imageInStack, self.filename_Px2imageInStack, "",  self.spin_resolStack.value(), 1, self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_mediane_pond,self.filename_traceFaultStack,fname_offsetsOutStack,typeDir,'', 'r', 'distance along profile (px)', 'offset (m)','Median Stack Profile',startName_stack,showErr,showStack)
        elif self.radio_meanStack.isChecked():
          stackPerp(self.filename_Px1imageInStack, self.filename_Px2imageInStack, "", self.spin_resolStack.value(), 1, self.spin_lengthStack.value(), self.spin_widthStack.value(), self.spin_distStack.value(),calc_profil_coef_correl,self.filename_traceFaultStack,fname_offsetsOutStack,typeDir,'', 'r', 'distance along profile (px)', 'offset (m)','Mean Stack Profile',startName_stack,showErr,showStack)
        self.groupBox_stack.setChecked(False)
        self.groupBox_cg.setChecked(True)
        #~ self.edit_offsetsInCg.setText(self.filename_offsetsOutStack)
    else:
      QtGui.QMessageBox.warning(self, 'Error detected',"Length and width of stack must be odd numbers!", QtGui.QMessageBox.Ok)


  def update_Weight_input(self):
    with_weight= self.groupBox_stack.isChecked() and (self.radio_WmedianStack.isChecked() or self.radio_WmeanStack.isChecked())
    self.edit_filename_imageWeightStack.setEnabled(with_weight)
    self.spin_expOfWeightStack.setEnabled(with_weight)

  #---------------
  def ask_offsetsInCg_file(self):
    self.edit_offsetsInCg.clear()
    self.edit_fileOutCg.clear()
    self.filename_offsetsInCg=QtGui.QFileDialog.getOpenFileName(self,"Select Offsets Info File",".","Text files (*.txt);;All files (*)")
    if (self.filename_offsetsInCg!=""):
      self.edit_offsetsInCg.setText(self.filename_offsetsInCg)


  def ask_Cg_file(self):
    self.edit_fileOutCg.clear()
    self.filename_offsetsInCg=str(self.edit_offsetsInCg.text())
    tmp_abs=os.path.abspath(self.filename_offsetsInCg)
    tmp=os.path.split(tmp_abs)[1].split('.')
    fname="slip_curve_"+".".join(tmp[0:-1])+".png"
    self.filename_Cg=QtGui.QFileDialog.getSaveFileName(self,"Select name of file for slip-curve",fname,"Images (*.png *.tif *.tiff  *.jpg);;All files (*)")
    if (self.filename_Cg!=""):
      self.edit_fileOutCg.setText(self.filename_Cg)

  def active_buttonOkCg(self):
    self.filename_offsetsInCg=str(self.edit_offsetsInCg.text())
    self.filename_Cg=str(self.edit_fileOutCg.text())
    if (len(self.filename_offsetsInCg)>0) and (len(self.filename_Cg)>0):
      self.button_okCg.setEnabled(True)
    else:
      self.button_okCg.setEnabled(False)

  def click_okCg(self):
    print "OffsetsInCg file: ", self.filename_offsetsInCg
    drawCG(self.filename_offsetsInCg, self.filename_Cg)
    self.groupBox_cg.setChecked(False)


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


#"http://www.cecill.info".
if __name__ == '__main__':
    main()

