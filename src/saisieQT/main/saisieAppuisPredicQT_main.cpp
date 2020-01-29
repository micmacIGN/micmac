#include "saisieQT_main.h"
void SaisieAppuisPredic(int argc, char ** argv,
                      Pt2di &aSzW,
                      Pt2di &aNbFen,
                      std::string &aFullName,
                      std::string &aDir,
                      std::string &aName,
                      std::string &aNamePt,
                      std::string &anOri,
                      std::string &aModeOri,
                      std::string &aNameMesure,
                      std::string &aTypePts,
                      std::string &aMasq3D,
                      std::string &PIMsFilter,
                      double &aFlou,
                      bool &aForceGray,
                      double &aZMoy,
                      double &aZInc,
                      std::string & aInputSec,
                      bool & WithMaxMinPt,
		      double & aGama,
                      std::string & aPatFilter,
                      double & aDistMax
       );


using namespace std;

int saisieAppuisPredicQT_main(int argc, char *argv[])
{
	QApplication &app = getQApplication();

    app.setApplicationName("SaisieAppuisPredicQT");
    app.setOrganizationName("Culture3D");

    QStringList cmdline_args = QCoreApplication::arguments();

    if (cmdline_args.back().contains("help"))
    {
        QString help = "Mandatory unnamed args :\n"
                 "* string :: {Full name (Dir+Pattern)}\n"
                 "* string :: {Orientation ; NONE if not used}\n"
                 "* string :: {File for Ground Control Points}\n"
                 "* string :: {File for Image Measurements}\n\n"
                "Named args :\n"
                "* [Name=SzW] Pt2di :: {Sz of window}\n"
                "* [Name=NbF] Pt2di :: {Nb of sub window}\n"
                "* [Name=WBlur] REAL :: {Size IN GROUND GEOMETRY of bluring for target}\n"
                "* [Name=Type] string :: {in [MaxLoc,MinLoc,GeoCube]}\n"
                "* [Name=ForceGray] bool :: {Force gray image, def=false}\n\n"
                "* [Name=Gama] double :: {gama, def = 1.0}\n\n"

                "Example:\nmm3d " + app.applicationName() + " IMG_558{0-9}[1].tif RadialBasic gcp.xml measures.xml\n\n"
                "NB: visual interface for argument edition available with command\n\n mm3d v" + app.applicationName() + "\n\n";

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

    string aFullName, aDir, aName, aNamePt;   //mandatory arguments
    string aNameOri, aModeOri, aNameMesure;   //named args
    string aTypePts="Pts";
    std::string aMasq3D,aPIMsFilter;
    double aFlou=0.;
    double aZMoy, aZInc;
    std::string aInputSec;

    bool aForceGray = false;
    double aGama = 1.0;

    settings.beginGroup("Misc");
    aNameMesure = settings.value("defPtName", QString("100")).toString().toStdString();
    settings.endGroup();

    settings.beginGroup("Drawing settings");
    aForceGray = settings.value("forceGray", false       ).toBool();
    settings.endGroup();

    if (argv[0][0] == 'v')
    {
        MMVisualMode = true;
        argv[0] = (char*) "SaisieAppuisPredicQT";
    }

    bool WithMaxMinPt=false;
    std::string aPatF;
    double aDistMax;
    SaisieAppuisPredic(argc, argv, aSzWin, aNbFen, aFullName, aDir, aName, aNamePt, aNameOri, aModeOri, aNameMesure, aTypePts, aMasq3D,aPIMsFilter,aFlou, aForceGray, aZMoy, aZInc, aInputSec,WithMaxMinPt, aGama,aPatF,aDistMax);

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
        settings.setValue("defPtName", QString(aNameMesure.c_str()));
        settings.endGroup();

        QStringList input;
        input   << QString(MMDir().c_str()) + QString("bin/SaisiePts")
                << QString(MMDir().c_str()) + QString("include/XML_MicMac/SaisieAppuisPredic.xml")
                << QString("DirectoryChantier=") + QString(aDir.c_str())
                << QString("+Images=") +  QString(aName.c_str())
                << QString("+Ori=") + QString(aNameOri.c_str())
                << QString("+LargeurFlou=") + QString::number(aFlou)
                << QString("+Terrain=") + QString(aNamePt.c_str())
                << QString("+Sauv=") + QString(aNameMesure.c_str())
                << QString("+SzWx=") + QString::number(aSzWin.x)
                << QString("+SzWy=") + QString::number(aSzWin.y)
                << QString("+UseMinMaxPt=") + QString(ToString(WithMaxMinPt).c_str())

                << QString("+NbFx=") + QString::number(aNbFen.x)
                << QString("+NbFy=") + QString::number(aNbFen.y)
                << QString("+TypePts=") + QString(aTypePts.c_str());

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

        if (EAMIsInit(&aMasq3D))
        {
            input << QString("+WithMasq3D=true");
            input << QString("+Masq3D=") + QString(aMasq3D.c_str());
        }

        if (EAMIsInit(&aPIMsFilter))
        {
            input << QString("+WithPIMsFilter=true");
            input << QString("+PIMsFilter=") + QString(aPIMsFilter.c_str());
        }

        if (EAMIsInit(&aFlou))
            input << QString("+FlouSpecified=") + QString::number(aFlou);

        if (EAMIsInit(&aTypePts))
            input << QString("+TypeGlobEcras=") + QString(aTypePts.c_str());

        if (EAMIsInit(&aForceGray))
            input << QString("+ForceGray=") + QString(aForceGray ? "1" : "0");

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

        SaisieQtWindow w(POINT2D_PREDIC);

        new cQT_Interface(anAppli,&w);

        w.show();

        w.addFiles(filenames, false);

        return app.exec();
    }
    else
        return EXIT_SUCCESS;
}

