#include <QApplication>
//#include <QtOpenGL/QGLWidget>
#include "mainwindow.h"

int main(int argc, char *argv[]) {

    QApplication::setStyle("fusion");

    QApplication app(argc, argv);
	
    MainWindow w;
    w.show();

    w.checkForLoadedEntities();

    return app.exec();
}
