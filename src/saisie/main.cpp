#include <QtGui>
#include <QApplication>
#include "mainwindow.h"

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hinstance, HINSTANCE hPrevInstance,LPSTR lpCmdLine, int nCmdShow)
#else
int main(int argc, char *argv[])
#endif
{
    QApplication::setStyle("fusion");

#ifdef WIN32
	LPWSTR *argv;
	int argc;
	int i;

	argv = CommandLineToArgvW(GetCommandLineW(), &argc);

	QApplication app(argc, 0);
#else
    QApplication app(argc, argv);
#endif
	
    app.setOrganizationName("IGN");
    app.setApplicationName("saisie");

    const QString locale = QLocale::system().name().section('_', 0, 0);

    // qt translations
    QTranslator qtTranslator;
    qtTranslator.load(app.applicationName() + "_" + locale);
    app.installTranslator(&qtTranslator);

    MainWindow w;

    QStringList cmdline_args = QCoreApplication::arguments();
    QString str;

    if (cmdline_args.size() > 1)
    {
        for (int i=0; i< cmdline_args.size(); ++i)
        {
            bool removeArg = false;

            str = cmdline_args[i];

            if (str == "mode2D")
            {
                w.setMode2D(true);

                removeArg = true;
            }
            else
            {
                if (str.contains("Post="))
                {
                    w.setPostFix(str.mid(str.indexOf("Post=")+5, str.size()));

                    removeArg = true;
                }

                if (str.contains("SzW="))
                {
                    QString arg = str.mid(str.indexOf("SzW=")+4, str.size());
                    int szW = arg.toInt();

                    int szH = szW * w.height() / w.width();

                    w.resize( szW, szH );

                    removeArg = true;
                }

                if (str.contains("Name="))
                {
                    QString arg = str.mid(str.indexOf("Name=")+5, str.size());

                    w.getEngine()->setFilenameOut(arg);

                    removeArg = true;
                }

				if (str.contains("Gama="))
                {
                    QString strGamma = str.mid(str.indexOf("Gama=")+5, str.size());

					float aGamma = strGamma.toFloat();

                    w.setGamma(aGamma);

                    removeArg = true;
                }
            }

            if (removeArg)
            {
                cmdline_args[i] = cmdline_args.back();
                cmdline_args.pop_back();
                i--;
            }
        }
    }


    w.show();

    if (cmdline_args.size() > 1)
    {
        cmdline_args[0] = cmdline_args.back();
        cmdline_args.pop_back();

        w.addFiles(cmdline_args);
    }

    w.checkForLoadedData();

#ifdef WIN32
	LocalFree(argv);
#endif

    return app.exec();
}

