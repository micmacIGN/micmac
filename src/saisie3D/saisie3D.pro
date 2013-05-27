#-------------------------------------------------
#
# Project created by QtCreator 2013-03-28T11:24:12
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = saisie3D
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp\
        GLWidget.cpp \
    Cloud.cpp \
    ../src/poisson/plyfile.cpp

HEADERS  += mainwindow.h\
            GLWidget.h \
    util.h \
    mmVector3.h

FORMS    += \
    mainwindow.ui

RESOURCES += \
    icones/icones.qrc
	
#Don't warn about sprintf, fopen etc being 'unsafe'
DEFINES += _CRT_SECURE_NO_WARNINGS

