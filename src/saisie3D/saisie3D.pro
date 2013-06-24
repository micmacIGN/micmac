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

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

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
unix: LIBS += -lGLU -lGLEW

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}
#end of section to comment

CONFIG(debug)
{
unix|win32: LIBS += -L$$PWD/../../bin -lelise
unix: LIBS += -lGLU -lGLEW

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}
