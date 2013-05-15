#include <QtGui/QApplication>
#include <QtOpenGL/QGLWidget>
#include "mainwindow.h"

int main(int argc, char *argv[]) {

    QApplication app(argc, argv);
	
    MainWindow w;
    w.show();

    w.checkForLoadedEntities();

    return app.exec();
}
