#-------------------------------------------------
#
# Project created by QtCreator 2013-07-16T14:32:41
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Saisie2D
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
        Engine2D.cpp

HEADERS  += mainwindow.h \
    Engine2D.h

FORMS    += mainwindow.ui

RESOURCES += \
    ../icones/icones.qrc

DEFINES += _CRT_SECURE_NO_WARNINGS

macx: LIBS+= -L/usr/X11R6/lib/ -lX11 -lglut
else:unix: LIBS += -lGLU -lGLEW -lglut
unix:!macx: QMAKE_CXXFLAGS += -Wall -Wno-ignored-qualifiers -Wno-unused-parameter

