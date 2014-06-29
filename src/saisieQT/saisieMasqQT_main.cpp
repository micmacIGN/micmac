#include "saisieQT_main.h"

int saisieMasqQT_main(QApplication &app, int argc, char *argv[])
{
    MMD_InitArgcArgv(argc,argv);

    app.setApplicationName("SaisieMasqQT");
    app.setOrganizationName("Culture3D");

    SaisieQtWindow w;

    if ((argc==2) && (argv[1][0] == 'v') )
    {
        char* nargv[] = {(char*) "SaisieMasqQT"};
        argc = 1;

        MMVisualMode = true;

        Pt2di SzWP = Pt2di(900,700);
        std::string aFullName;
        std::string aPost("Masq");
        std::string aNameMasq ="";
        double aGama=1.0;
        ElInitArgMain
        (
               argc,nargv,
               LArgMain() << EAMC(aFullName,"Name of input image", eSAM_IsExistFile) ,
               LArgMain() << EAM(SzWP,"SzW",true)
                          << EAM(aPost,"Post",true)
                          << EAM(aNameMasq,"Name",true,"Name of result, default toto->toto_Masq.tif")
                          << EAM(aGama,"Gama",true)
        );

        MMVisualMode = false;
        return EXIT_SUCCESS;
    }
    else
    {
        QStringList cmdline_args = QCoreApplication::arguments();
        QString str;

        if (cmdline_args.size() > 1)
        {
            cmdline_args.pop_front(); //remove "SaisieQT"

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
                                    "* [Name=Attr] string\t: string to add to postfix\n\n"
                                    "Example: " + app.applicationName() + " IMG.tif SzW=1200 Name=PLAN Gama=1.5\n\n"
                                    "NB: " + app.applicationName() + " can be run without any argument\n\n";

                    return helpMessage(app, help);
                }

                if (str.contains("Post="))
                {
                    w.setPostFix(str.mid(str.indexOf("Post=")+5, str.size()));

                    removeArg = true;
                }
                else
                    w.setPostFix("Masq");

                if (str.contains("SzW="))
                {
                    Pt2di SzWP = Pt2di(900,700);

                    //TODO:  load SzW
                    w.resize(SzWP.x,SzWP.y);

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

                if (str.contains("Attr="))
                {
                    QString arg = str.mid(str.indexOf("Attr=")+5, str.size());

                    w.setPostFix(w.getPostFix() + arg);

                    removeArg = true;
                }

                if (str.contains(app.applicationName()))
                    removeArg=true;


                if (removeArg)
                {
                    cmdline_args[i] = cmdline_args.back();
                    cmdline_args.pop_back();
                    i--;
                }
            }

            w.show();

            if (cmdline_args.size() > 0)
                w.addFiles(cmdline_args, true);
        }

        return app.exec();
    }
}
