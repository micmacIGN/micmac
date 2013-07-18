#-------------------------------------------------
#
# Project created by QtCreator 2013-03-28T11:24:12
#
#-------------------------------------------------

#CONFIG(release, release|debug)
#{
#	LIBS += -L../../build/src/Release -lelise
#} else {
#	LIBS += -L../../bin -lelise
#}

QT       += core gui opengl xml


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets concurrent

TARGET = saisie
TEMPLATE = app

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
    ./icones/icones.qrc
	
DEFINES += TWEAK
#Don't warn about sprintf, fopen etc being 'unsafe'
DEFINES += _CRT_SECURE_NO_WARNINGS
win32: DEFINES += ELISE_windows

INCLUDEPATH += $$PWD/../../include
DEPENDPATH += $$PWD/../../include
DEPENDPATH += ./translations

#comment to run debug

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

CONFIG(release)
{
unix|win32: LIBS += -L$$PWD/../../lib -lelise

macx: LIBS+= -L/usr/X11R6/lib/ -lX11 -lglut
else:unix: LIBS += -lGLU -lGLEW -lglut
unix:!macx: QMAKE_CXXFLAGS += -Wall -Wno-ignored-qualifiers -Wno-unused-parameter

win32: PRE_TARGETDEPS += $$PWD/../../lib/elise.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../lib/libelise.a
}
# Tell Qt Linguist that we use UTF-8 strings in our sources
TRANSLATIONS += ./translations/saisie_fr.ts
CODECFORTR = UTF-8
CODECFORSRC = UTF-8
#include(./translations/locale.pri)

INSTALLS += translations

translations.path = ./translations
translations.files += $$DESTDIR/locale
