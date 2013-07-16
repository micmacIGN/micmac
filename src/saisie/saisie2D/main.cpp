#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication::setStyle("fusion");

    QApplication app(argc, argv);

    app.setOrganizationName("IGN");
    app.setApplicationName("Saisie2D");

    MainWindow w;
    w.show();

    QStringList cmdline_args = QCoreApplication::arguments();

    if (cmdline_args.size() > 1)
    {
        cmdline_args.pop_front();
        w.addFiles(cmdline_args);
    }
    
    return app.exec();
}
