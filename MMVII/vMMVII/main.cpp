#include "global.h"
#include "mainwindow.h"

#include <QFileInfo>
#include <QApplication>
#include <QMessageBox>
#include <QCommandLineParser>
#include <QTextStream>
#include <QSettings>

#define SPEC_FILENAME "MMVII_argsspec.json"

// TODOCM: Gerer FDP et/ou DP : changer repertoire courant tempo ?
// TODOCM: MPF ?
// TODOCM: Liste commande pas affichee (ou affichee ?)

static void parseArgs(QString& mmviiPath, QString& specPath,QStringList& command)
{
    QCommandLineParser parser;

    parser.setApplicationDescription(QObject::tr("GUI front end for MMVII","main"));
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("command", QObject::tr("MMVII command to execute","main"));
    parser.addOptions({
        {{"d", "debug"},
         QObject::tr("Show some debug information.","main")},
        {{"m", "mmvii"},
         QObject::tr("Full path MMVII executable.","main"),
         QObject::tr("MMVII","main")},
        {{"s", "specs"},
         QObject::tr("Full path name of argument specification file.","main"),
         QObject::tr("SPECS.JSON","main")},
      });

    parser.process(QCoreApplication::arguments());

    showDebug = parser.isSet("d");
    mmviiPath = parser.value("m");
    specPath = parser.value("s");
    command = parser.positionalArguments();
}


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QApplication::setOrganizationName(APP_ORGANIZATION);
    QApplication::setApplicationName(APP_NAME);
    QApplication::setApplicationVersion(APP_VERSION);

    QSettings::setDefaultFormat(QSettings::IniFormat);

    QLocale::setDefault(QLocale::C);

    QString mmviiPath, specPath;
    QStringList command;
    parseArgs(mmviiPath, specPath, command);


    MainWindow w(mmviiPath, specPath, command);
    if (! w.initOk())
        return EXIT_FAILURE;
    w.show();
    return app.exec();
}
