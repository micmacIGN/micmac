#include "saisieQT_main.h"

int saisieMasqQT_main(QApplication &app)
{
    app.setApplicationName("SaisieMasqQT");

    Pt2di NbFen(1,1);
    Pt2di SzWin(800,600);

    MainWindow w(SzWin, NbFen);

    QStringList cmdline_args = QCoreApplication::arguments();
    QString str;

    if (cmdline_args.size() > 1)
    {
        cmdline_args.pop_front();

        for (int i=0; i< cmdline_args.size(); ++i)
        {
            bool removeArg = false;

            str = cmdline_args[i];
#ifdef _DEBUG
            cout << "arguments : " << str.toStdString().c_str() << endl;
#endif

            if (str.contains("help"))
            {
                QString help =  app.applicationName() +" [filename] [option=]\n\n"
                                "* [filename] string\t: open file (image or ply or camera xml)\n\n"
                                "Options\n\n"
                                "* [Name=SzW] integer\t: set window width (default=800)\n"
                                "* [Name=Post] string\t: change postfix output file (default=_Masq)\n"
                                "* [Name=Name] string\t: set output filename (default=input+_Masq)\n"
                                "* [Name=Gama] REAL\t: apply gamma to image\n\n"
                                "Example: " + app.applicationName() + " IMG.tif SzW=1200 Name=PLAN Gama=1.5\n\n"
                                "NB: " + app.applicationName() + " can be run without any argument\n\n";

                return helpMessage(app, help);
            }
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
                    QString arg = str.mid(str.indexOf("Gama=")+5, str.size());

                    float aGamma = arg.toFloat();

                    w.setGamma(aGamma);

                    removeArg = true;
                }

                if (str.contains(app.applicationName()))
                    removeArg=true;
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

    if (cmdline_args.size() > 0)
        w.addFiles(cmdline_args);

    return app.exec();
}
