#include "saisieQT_main.h"
#include "../uti_phgrm/SaisiePts/cParamSaisiePts.h"
#include "QT_interface_Elise.h"
#include "private/files.h"

using namespace std;

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


int saisieAppuisInitQT_main(QApplication &app, int argc, char *argv[])
{
    app.setApplicationName("SaisieAppuisInitQT");
    app.setOrganizationName("IGN");

    QSettings settings(QApplication::organizationName(), QApplication::applicationName());

    if ((argc>0)&&(string(argv[0]).find("SaisieQT")!= string::npos))
    {
        argv++;
        argc--;
    }

    Pt2di aSzWin(800,800);
    Pt2di aNbFen(-1,-1);

    string aFullName, aDir, aName, aNamePt, aNameOut;   //mandatory arguments
    string aNameOri, aNameAuto, aPrefix2Add;            //named args
    settings.beginGroup("Misc");
    aNameAuto = settings.value("defPtName", QString("100")).toString().toStdString();
    settings.endGroup();
    aPrefix2Add = "";
    bool aForceGray = false;

    SaisieAppuisInit(argc, argv, aSzWin, aNbFen, aFullName, aDir, aName, aNamePt, aNameOri, aNameOut, aNameAuto, aPrefix2Add, aForceGray);

    list<string> aNamelist = RegexListFileMatch(aDir, aName, 1, false);
    QStringList filenames;

    for
    (
        list<string>::iterator itS=aNamelist.begin();
        itS!=aNamelist.end();
        itS++
    )
        filenames.push_back( QString((aDir + *itS).c_str()));

    int aNbW = aNbFen.x * aNbFen.y;
    if (filenames.size() < aNbW)
    {
         aNbW = filenames.size();

         cVirtualInterface::ComputeNbFen(aNbFen, aNbW);
    }

    bool init = settings.contains("MainWindow/size");

    settings.beginGroup("MainWindow");
    if (aSzWin.x > 0)
        settings.setValue("size", QSize(aSzWin.x, aSzWin.y));
    else if (!init)
    {
        settings.setValue("size", QSize(800, 800));
         aSzWin.x = aSzWin.y = 800;
    }
    settings.setValue("NbFen", QPoint(aNbFen.x, aNbFen.y));
    settings.endGroup();

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
            << QString("+NbFx=") + QString::number(aNbFen.x)
            << QString("+NbFy=") + QString::number(aNbFen.y) ;

    if (EAMIsInit(&aForceGray))
       input << QString("+ForceGray=") + QString(((string)(ToString(aForceGray))).c_str());

    if (EAMIsInit(&aPrefix2Add))
       input << QString("+Pref2Add=") + QString(aPrefix2Add.c_str());


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
