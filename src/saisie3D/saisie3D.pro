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
			util.h \
			mmVector3.h \
    Data.h \
    Engine.h

FORMS    += \
    mainwindow.ui

RESOURCES += \
    icones/icones.qrc
	
#Don't warn about sprintf, fopen etc being 'unsafe'
DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH += $$PWD/../../include
DEPENDPATH += $$PWD/../../include

CONFIG(release, release|debug)
{
unix|win32: LIBS += -L$$PWD/../../lib -lelise

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}else{
unix|win32: LIBS += -L$$PWD/../../bin -lelise

win32: PRE_TARGETDEPS += $$PWD/../../bin/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../bin/libelise.a
}
