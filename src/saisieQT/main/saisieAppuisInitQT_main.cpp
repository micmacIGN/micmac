#include "saisieQT_main.h"

extern void SaisieAppuisInit(int argc, char ** argv,
                      Pt2di &aSzW,
                      Pt2di &aNbFen,
                      std::string &aFullName,
                      std::string &aDir,
                      std::string &aName,
                      std::string &aNamePt,
                      std::string &anOri,
                      std::string &aModeOri,
                      std::string &anOut,
                      std::string &aNameAuto,
                      std::string &aPrefix2Add,
                      bool &aForceGray,
                      double &aZMoy,
                      double &aZInc,
                      std::string & aInputSec,
                      bool & WithMaxMin, 
		      double & aGama
                      );


//extern const char * theNameVar_ParamApero[];

int SaisiePts_main2(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);
  // cAppliApero * anAppli = cAppliMICMAC::Alloc(argc,argv,eAllocAM_STD);

  //if (0) delete anAppli;

   ELISE_ASSERT(argc>=2,"Not enough arg");

   cElXMLTree aTree(argv[1]);


   cResultSubstAndStdGetFile<cParamSaisiePts> aP2
                                          (
                                               argc-2,argv+2,
                                              //0,0,
                                      argv[1],
                                   StdGetFileXMLSpec("ParamSaisiePts.xml"),
                                  "ParamSaisiePts",
                                  "ParamSaisiePts",
                                              "DirectoryChantier",
                                              "FileChantierNameDescripteur"
                                          );

   //cAppli_SaisiePts   anAppli (aP2);
   //((cX11_Interface*)anAppli.Interface())->BoucleInput();

   //SaisiePts_Banniere();
   return 0;
}

int saisieAppuisInitQT_main(int argc, char *argv[])
{
	QApplication &app = getQApplication();

    app.setApplicationName("SaisieAppuisInitQT");
    app.setOrganizationName("Culture3D");

    QStringList cmdline_args = QCoreApplication::arguments();

    if (cmdline_args.back().contains("help"))
    {
        QString help = "Mandatory unnamed args :\n"
                 "* string :: {Full name (Dir+Pattern)}\n"
                 "* string :: {Orientation ; NONE if not used}\n"
                 "* string :: {Point name, or point file name}\n"
                 "* string :: {Output}\n\n"
                "Named args :\n"
                "* [Name=SzW] Pt2di :: {Sz of window}\n"
                "* [Name=NbF] Pt2di :: {Nb of sub window}\n"
                "* [Name=NameAuto] string :: {Prefix for automatic point creation}\n"
                //"* [Name=Pref2Add] string :: {Prefix to add during import (for bug correction ?)}\n"
                "* [Name=ForceGray] bool :: {Force gray image, def=false}\n"
                "* [Name=Gama] double :: {Gama ,def = 1.0}\n"
                "* [Name=OriMode] string :: {Orientation type (GRID) (Def=Std)}\n"
                "* [Name=ZMoy] REAL :: {Average Z, Mandatory in PB}\n"
                "* [Name=ZInc] REAL :: {Incertitude on Z, Mandatory in PB}\n\n"

                "Example:\nmm3d " + app.applicationName() + " IMG_558{0-9}[1].tif RadialBasic 100 measures.xml\n\n"
                "NB: visual interface for argument edition available with command:\n\n mm3d v" + app.applicationName() + "\n\n";

        return helpMessage(app, help);
    }

    loadTranslation(app);

    QSettings settings(QApplication::organizationName(), QApplication::applicationName());

    if ((argc>0)&&(string(argv[0]).find("SaisieQT")!= string::npos))
    {
        argv++;
        argc--;
    }

    Pt2di aSzWin(800,800);
    Pt2di aNbFen(-1,-1);

    string aFullName, aDir, aName, aNamePt, aNameOut;   //mandatory arguments
    string aNameOri, aModeOri, aNameAuto, aPrefix2Add;  //named args
    aPrefix2Add = "";
    bool aForceGray = false;
    double aGama = 1.0;

    settings.beginGroup("Misc");
    aNameAuto = settings.value("defPtName", QString("100")).toString().toStdString();
    settings.endGroup();

    settings.beginGroup("Drawing settings");
    aForceGray = settings.value("forceGray", false       ).toBool();
    settings.endGroup();

    double aZInc, aZMoy;

    if (argv[0][0] == 'v')
    {
        MMVisualMode = true;
        argv[0] = (char*) "SaisieAppuisInitQT";
    }
    
    std::string aInputSec;
    bool  WithMaxMin=false;

    SaisieAppuisInit(argc, argv, aSzWin, aNbFen, aFullName, aDir, aName, aNamePt, aNameOri, aModeOri, aNameOut, aNameAuto, aPrefix2Add, aForceGray, aZMoy, aZInc,aInputSec,WithMaxMin, aGama);

    if (!MMVisualMode)
    {
        if (!checkNamePt( QString (aNamePt.c_str()))) return -1;

        QStringList filenames = getFilenames(aDir, aName);

        int aNbW = aNbFen.x * aNbFen.y;
        if (filenames.size() < aNbW)
        {
             aNbW = filenames.size();

             cVirtualInterface::ComputeNbFen(aNbFen, aNbW);
        }

        updateSettings(settings, aSzWin,aNbFen, aForceGray);

        settings.beginGroup("Misc");
        settings.setValue("defPtName", QString(aNameAuto.c_str()));
        settings.endGroup();

        QStringList input;
        input   << QString(MMDir().c_str()) + QString("bin/SaisiePts")
                << QString(MMDir().c_str()) + QString("include/XML_MicMac/SaisieInitiale.xml")
                << QString("DirectoryChantier=") + QString(aDir.c_str())
                << QString("+Image=") +  QString(aName.c_str())
                << QString("+Ori=") + QString(aNameOri.c_str())
                << QString("+NamePt=") + QString(aNamePt.c_str())
                << QString("+NameAuto=") + QString(aNameAuto.c_str())
                << QString("+Sauv=") + QString(aNameOut.c_str())
                << QString("+SzWx=") + QString::number(aSzWin.x)
                << QString("+SzWy=") + QString::number(aSzWin.y)
                << QString("+UseMinMaxPt=") + QString(ToString(WithMaxMin).c_str())

                << QString("+NbFx=") + QString::number(aNbFen.x)
                << QString("+NbFy=") + QString::number(aNbFen.y);

/*
if (MPD_MM())
{
    std::cout << input << "\n";
getchar();
}
*/

        if (aModeOri == "GRID")
        {
            input << QString("+ModeOriIm=eGeomImageGrille")
                  << QString("+Conik=false")
                  << QString("+ZIncIsProp=false")
                    //<< QString(+PostFixOri=GRIBin")
                  << QString("+Px1Inc=") + QString::number(aZInc)
                  << QString("+Px1Moy=") + QString::number(aZMoy);

            //<< QString("+Geom=eGeomMNTFaisceauIm1ZTerrain_Px1D");
        }

        if (EAMIsInit(&aForceGray))
           input << QString("+ForceGray=") + QString(((string)(ToString(aForceGray))).c_str());

        if (EAMIsInit(&aPrefix2Add))
           input << QString("+Pref2Add=") + QString(aPrefix2Add.c_str());

	if (EAMIsInit(&aGama))
           input << QString("+Gama=") + QString::number(aGama);


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

        cAppli_SaisiePts   anAppli (aP2,false);

        SaisieQtWindow w(POINT2D_INIT);

        new cQT_Interface(anAppli,&w);

        w.show();

        w.addFiles(filenames, false);

        return app.exec();
    }
    else
        return EXIT_SUCCESS;
}
