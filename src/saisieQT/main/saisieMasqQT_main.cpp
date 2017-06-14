#include "saisieQT_main.h"

void saisieMasq_ElInitArgMain(int argc,char ** argv, std::string &aFullName, std::string &aPost, std::string &aNameMasq, std::string &aAttr, Pt2di &aSzW, double &aGama,bool & aForceTif);


int saisieMasqQT_main(int argc, char *argv[])
{
	QApplication &app = getQApplication();

    Pt2di SzWP = Pt2di(900,700);
    std::string aFullName ="";
    std::string aPost("_Masq");
    std::string aNameMasq ="";
    std::string aAttr="";
    double aGama=1.0;
    bool aForceTif;

    if (argv[1][0] == 'v')
    {
        argv++;
        argc--;

        argv[0] = (char*) "SaisieMasqQT";

        MMVisualMode = true;
        saisieMasq_ElInitArgMain(argc, argv, aFullName, aPost, aNameMasq, aAttr, SzWP, aGama,aForceTif);
        MMVisualMode = false;

        return EXIT_SUCCESS;
    }
    else
    {
        //Set application name before SaisieQtWindow creation to read settings
        app.setApplicationName("SaisieMasqQT");
        app.setOrganizationName("Culture3D");

        QStringList cmdline_args = QCoreApplication::arguments();

        loadTranslation(app);

        SaisieQtWindow win;
        cQT_Interface::connectDeviceElise(win);

#ifdef _DEBUG
        for (int aK=0; aK < cmdline_args.size();++aK)
            cout << "arguments : " << cmdline_args[aK].toStdString().c_str() << endl;
#endif

        if (cmdline_args.back().contains("help"))
        {
            QString help =  app.applicationName() +" [filename] [option=]\n\n"
                    "* [filename] string\t: open file (image or ply or camera xml)\n\n"
                    "Options\n\n"
                    "* [Name=SzW] Pt2di\t: set window size (default=[900,700])\n"
                    "* [Name=Post] string\t: change postfix output file (default=_Masq)\n"
                    "* [Name=Name] string\t: set output filename (default=input+_Masq)\n"
                    "* [Name=Gama] REAL\t: apply gamma to image\n"
                    "* [Name=Attr] string\t: string to add to postfix\n\n"
                    "Example: mm3d " + app.applicationName() + " IMG.tif SzW=[1200,800] Name=PLAN Gama=1.5\n\n"
                    "NB: \n"
                    + app.applicationName() + " can be run without any argument\n"
                    "Visual interface for argument edition available with command: mm3d v" + app.applicationName() + "\n\n";

            return helpMessage(app, help);
        }
        else if (cmdline_args.size() != 2)
        {
            argc--;
            argv++;

            saisieMasq_ElInitArgMain(argc, argv, aFullName, aPost, aNameMasq, aAttr, SzWP, aGama,aForceTif);

            if (EAMIsInit(&aPost))
                win.setPostFix(QString(aPost.c_str()));
            else
                win.setPostFix("_Masq");

            if (EAMIsInit(&aAttr))
                win.setPostFix(win.getPostFix() + QString(aAttr.c_str()));

            if (EAMIsInit(&aGama))
                win.setGamma(aGama);

            win.resize(SzWP.x,SzWP.y);
        }

        win.show();

        if (aFullName != "")
        {
            QStringList filenames;

            if (aFullName == "-l")
            {
                QSettings settings;
                QStringList files = settings.value("recentFileList").toStringList();

                filenames.push_back(files.first());
            }
            else filenames.push_back(QString(aFullName.c_str()));

            win.addFiles(filenames,true);

            if(EAMIsInit(&aNameMasq))
                win.getEngine()->setFilenameOut(QString(aNameMasq.c_str()));
        }

        return app.exec();
    }
}
