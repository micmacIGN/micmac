#-------------------------------------------------
#
# Project created by QtCreator 2013-03-28T11:24:12
#
#-------------------------------------------------

#CONFIG(release, release|debug)
#{
#	LIBS += -L../build/src/Release -lelise
#} else {
#	LIBS += -L../bin -lelise
#}

QT       += core gui opengl xml


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets concurrent

TARGET = saisie3D
TEMPLATE = app

DEFINES += TWEAK



SOURCES += main.cpp\
        mainwindow.cpp\
        GLWidget.cpp \
        Cloud.cpp \
        ../poisson/plyfile.cpp \
        Data.cpp \
        Engine.cpp

HEADERS  += mainwindow.h\
            GLWidget.h \
            Data.h \
            Engine.h \
		3DTools.h \
		Cloud.h

FORMS    += \
    mainwindow.ui

RESOURCES += \
    icones/icones.qrc
	
#Don't warn about sprintf, fopen etc being 'unsafe'
DEFINES += _CRT_SECURE_NO_WARNINGS
win32: DEFINES += ELISE_windows

INCLUDEPATH += $$PWD/../../include
DEPENDPATH += $$PWD/../../include

#comment to run debug
CONFIG(release)
{
unix|win32: LIBS += -L$$PWD/../../lib -lelise

macx: LIBS+= -L/usr/X11R6/lib/ -lX11 -lglut
else:unix: LIBS += -lGLU -lGLEW -lglut
unix:!macx: QMAKE_CXXFLAGS += -Wall -Wno-ignored-qualifiers -Wno-unused-parameter

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}
#end of section to comment

CONFIG(debug)
{
unix|win32: LIBS += -L$$PWD/../../bin -lelise

macx: LIBS+= -L/usr/X11R6/lib/ -lX11 -lglut
else:unix: LIBS += -lGLU -lGLEW -lglut
unix:!macx: QMAKE_CXXFLAGS += -Wall -Wno-ignored-qualifiers -Wno-unused-parameter

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}
