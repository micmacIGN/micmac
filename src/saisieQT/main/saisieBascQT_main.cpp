#include "saisieQT_main.h"

int saisieBascQT_main(int argc, char *argv[])
{
	QApplication &app = getQApplication();

    QStringList cmdline_args = QCoreApplication::arguments();

    QString aMode = "0";
    if (argc>1)
    {
        string str = string(argv[argc-1]);
        size_t pos = str.find("Mode=");
        if(pos != string::npos)
        {
            aMode=QString::fromStdString(str.substr(pos + sizeof("Mode=") -1, 1));
            argc--;
        }
    }

    MMD_InitArgcArgv(argc,argv);

    if (aMode == "0")
        app.setApplicationName("SaisieBascQT");
    else if (aMode == "1")
        app.setApplicationName("SaisieCylQT");

    app.setOrganizationName("Culture3D");

    if (cmdline_args.back().contains("help"))
    {
        QString help = "Mandatory unnamed args :\n"
                 "* string :: {Full Name (Dir+Pattern)}\n"
                 "* string :: {Orientation, NONE if unused}\n"
                 "* string :: {Output File}\n\n"
                "Named args :\n"
                "* [Name=SzW] Pt2di :: {Sz of window}\n"
                "* [Name=NbF] Pt2di :: {Number of Windows}\n"
                "* [Name=ForceGray] bool :: {Force gray image, def=false}\n\n"

                "Example:\nmm3d " + app.applicationName() + " IMG_558{0-9}[1].tif RadialBasic basc.xml\n\n"
                "NB: visual interface for argument edition available with command\n\n mm3d v" + app.applicationName();

        return helpMessage(app, help);
    }

    loadTranslation(app);

    if ((argc>0)&&(string(argv[0]).find("SaisieQT")!= string::npos))
    {
        argv++;
        argc--;
    }

    Pt2di aSzWin(800,800);
    Pt2di aNbFen(-1,-1);
    std::string aFullName,anOri,anOut, aDir, aName;
    bool aForceGray = false;

    QSettings settings(QApplication::organizationName(), QApplication::applicationName());

    settings.beginGroup("Drawing settings");
    aForceGray = settings.value("forceGray", false       ).toBool();
    settings.endGroup();

    if (argv[0][0] == 'v')
    {
        MMVisualMode = true;
        argv[0] = (char*) "SaisieBascQT";
    }

    SaisieBasc(argc, argv, aFullName, aDir, aName, anOri, anOut, aSzWin, aNbFen, aForceGray);

    if (!MMVisualMode)
    {
        list<string> aNamelist = RegexListFileMatch(aDir, aName, 1, false);
        QStringList filenames;

        for
        (
            list<string>::iterator itS=aNamelist.begin();
            itS!=aNamelist.end();
            itS++
        )
            filenames.push_back( QString((aDir + *itS).c_str()));

        updateSettings(settings, aSzWin, aNbFen, aForceGray);

        QStringList input;
        input   << QString(MMDir().c_str()) + QString("bin/SaisiePts")
                << QString(MMDir().c_str()) + QString("include/XML_MicMac/SaisieLine.xml")
                << QString("DirectoryChantier=") + QString(aDir.c_str())
                << QString("+Image=") +  QString(aName.c_str())
                << QString("+Ori=")  + QString(anOri.c_str())
                << QString("+Sauv=") + QString(anOut.c_str())
                << QString("+SzWx=") + QString::number(aSzWin.x)
                << QString("+SzWy=") + QString::number(aSzWin.y)
                << QString("+NbFx=") + QString::number(aNbFen.x)
                << QString("+NbFy=") + QString::number(aNbFen.y)
                << QString("+Mode=") + aMode;

        char **output;

        // Copy input to output
        output = new char*[input.size() + 1];
        for (int i = 0; i < input.size(); i++) {
            output[i] = new char[strlen(input.at(i).toStdString().c_str())+1];
            memcpy(output[i], input.at(i).toStdString().c_str(), strlen(input.at(i).toStdString().c_str())+1);
        }
        output[input.size()] = ((char*)NULL);

        cResultSubstAndStdGetFile<cParamSaisiePts> aP2(
                input.size()-2,output+2,
                output[1],
                StdGetFileXMLSpec("ParamSaisiePts.xml"),
                "ParamSaisiePts",
                "ParamSaisiePts",
                "DirectoryChantier",
                "FileChantierNameDescripteur" );

        cAppli_SaisiePts anAppli (aP2,false);

        SaisieQtWindow w(BASC);

        QAction* actionBascule = w.addCommandTools("Bascule");

        cQT_Interface* interQT = new cQT_Interface(anAppli,&w);

        interQT->connect(actionBascule,	SIGNAL(triggered()), interQT,SLOT(cmdBascule()));

        w.show();

        w.addFiles(filenames, false); // false = aSetGLData

        return app.exec();
    }
    else
        return EXIT_SUCCESS;
}

