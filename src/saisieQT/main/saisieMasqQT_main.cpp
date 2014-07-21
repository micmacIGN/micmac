#include "saisieQT_main.h"

int saisieMasqQT_main(QApplication &app, int argc, char *argv[])
{
    MMD_InitArgcArgv(argc,argv);

    Pt2di SzWP = Pt2di(900,700);
    std::string aFullName ="";
    std::string aPost("Masq");
    std::string aNameMasq ="";
    std::string aAttr="";
    double aGama=1.0;

    if (argv[1][0] == 'v')
    {
        argv++;
        argc--;

        argv[0] = (char*) "SaisieMasqQT";

        MMVisualMode = true;
        saisieMasq_ElInitArgMain(argc, argv, aFullName, aPost, aNameMasq, aAttr, SzWP, aGama);
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

        SaisieQtWindow w;

#ifdef _DEBUG
        for (int aK=0; aK < cmdline_args.size();++aK)
            cout << "arguments : " << cmdline_args[aK].toStdString().c_str() << endl;
#endif

        if ((cmdline_args.size() == 3) && (cmdline_args.back().contains("help")))
        {
            QString help =  app.applicationName() +" [filename] [option=]\n\n"
                    "* [filename] string\t: open file (image or ply or camera xml)\n\n"
                    "Options\n\n"
                    "* [Name=SzW] integer\t: set window width (default=800)\n"
                    "* [Name=Post] string\t: change postfix output file (default=_Masq)\n"
                    "* [Name=Name] string\t: set output filename (default=input+_Masq)\n"
                    "* [Name=Gama] REAL\t: apply gamma to image\n"
                    "* [Name=Attr] string\t: string to add to postfix\n\n"
                    "Example: mm3d " + app.applicationName() + " IMG.tif SzW=1200 Name=PLAN Gama=1.5\n\n"
                    "NB: \n"
                    "1: "+ app.applicationName() + " can be run without any argument\n"
                    "2: visual interface for argument edition available with v" + app.applicationName() + "\n\n";

            return helpMessage(app, help);
        }
        else if (cmdline_args.size() != 2)
        {
            argc--;
            argv++;

            saisieMasq_ElInitArgMain(argc, argv, aFullName, aPost, aNameMasq, aAttr, SzWP, aGama);

            if (EAMIsInit(&aPost))
                w.setPostFix(QString(aPost.c_str()));
            else
                w.setPostFix("Masq");

            if (EAMIsInit(&aAttr))
                w.setPostFix(w.getPostFix() + QString(aAttr.c_str()));

            if (EAMIsInit(&aGama))
                w.setGamma(aGama);

            if(EAMIsInit(&aNameMasq));
                w.getEngine()->setFilenameOut(QString(aNameMasq.c_str()));

            w.resize(SzWP.x,SzWP.y);
        }

        w.show();

        if (aFullName != "")
        {
            QStringList filenames;
            filenames.push_back(QString(aFullName.c_str()));

            w.addFiles(filenames,true);
        }

        return app.exec();
    }
}
