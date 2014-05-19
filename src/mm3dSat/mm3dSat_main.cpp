#include <QApplication>
#include "mm3dSat_mainwindow.h"
#include "StdAfx.h"


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
