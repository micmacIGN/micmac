#include <QtGui>
#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication::setStyle("fusion");

    QApplication app(argc, argv);
	
    app.setOrganizationName("IGN");
    app.setApplicationName("saisie");

    const QString locale = QLocale::system().name().section('_', 0, 0);

    // qt translations
    QTranslator qtTranslator;
    qtTranslator.load(app.applicationName() + "_" + locale);
    app.installTranslator(&qtTranslator);

    MainWindow w;

    QStringList cmdline_args = QCoreApplication::arguments();

    if (cmdline_args.size() > 1)
    {
        for (int i=0; i< cmdline_args.size(); ++i)
        {
            if (cmdline_args[i] == "mode2D")
            {
                cmdline_args[i] = cmdline_args.back();
                cmdline_args.pop_back();

                w.setMode2D(true);
                break;
            }
        }
    }

    w.show();

    if (cmdline_args.size() > 1)
        w.addFiles(cmdline_args);

    w.checkForLoadedData();

    return app.exec();
}

