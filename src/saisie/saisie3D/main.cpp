//#include <QApplication>
#include <QtGui>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication::setStyle("fusion");

    QApplication app(argc, argv);
	
    app.setOrganizationName("IGN");
    app.setApplicationName("Saisie3D");

    const QString locale = QLocale::system().name();

     // qt translations
     QTranslator qtTranslator;
     qtTranslator.load("qt_" + locale,
                       QLibraryInfo::location(QLibraryInfo::TranslationsPath));
     app.installTranslator(&qtTranslator);

     // app translations
   #ifdef PKGDATADIR
     QString dataDir = QLatin1String(PKGDATADIR);
   #else
     QString dataDir = "";
   #endif

#if defined(Q_OS_OS2) //|| defined(Q_OS_WIN) ->this isn't checked
     QString    localeDir = qApp->applicationDirPath() + QDir::separator() + "locale";
#else
     QString    localeDir = dataDir + QDir::separator() + "locale";
#endif

  QTranslator translator;
  translator.load(locale, localeDir);
  app.installTranslator(&translator);

    MainWindow w;
    w.show();

    QStringList cmdline_args = QCoreApplication::arguments();

    if (cmdline_args.size() > 1)
    {
        cmdline_args.pop_front();
        w.addFiles(cmdline_args);
    }

    w.checkForLoadedData();

    return app.exec();
}

