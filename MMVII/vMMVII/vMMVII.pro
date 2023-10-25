QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    actionbox.cpp \
    cmdconfigurewidget.cpp \
    cmdselectwidget.cpp \
    commandspec.cpp \
    ellipsiscombobox.cpp \
    global.cpp \
    inputwidget.cpp \
    main.cpp \
    mainwindow.cpp \
    processwidget.cpp \
    settings.cpp \
    spinboxdefault.cpp \
    workingdirwidget.cpp

HEADERS += \
    actionbox.h \
    cmdconfigurewidget.h \
    cmdselectwidget.h \
    commandspec.h \
    ellipsiscombobox.h \
    global.h \
    inputwidget.h \
    mainwindow.h \
    processwidget.h \
    settings.h \
    spinboxdefault.h \
    workingdirwidget.h

TARGET=vMMVII
# Default rules for deployment.
unix: target.path = ../bin
!isEmpty(target.path): INSTALLS += target

FORMS += \
    actionbox.ui \
    cmdSelect.ui \
    processwidget.ui \
    settings.ui \
    workingdirwidget.ui

DISTFILES += \
    TODO
