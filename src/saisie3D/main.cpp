#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication::setStyle("fusion");

    QApplication app(argc, argv);
	
    app.setOrganizationName("IGN");
    app.setApplicationName("Saisie3D");

    MainWindow w;
    w.show();
    w.checkForLoadedEntities();

    return app.exec();
}

