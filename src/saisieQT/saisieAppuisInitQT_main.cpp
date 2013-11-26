#include "saisieQT_main.h"

int saisieAppuisInitQT_main(QApplication &app, int argc, char *argv[])
{
    app.setApplicationName("SaisieAppuisInitQT");

    if ((argc>0)&&(string(argv[0]).find("SaisieQt")>0))
    {
        argv++;
        argc = argc -1;
    }

    Pt2di aSzW(800,800);
    Pt2di aNbFen(-1,-1);
    std::string aFullName, aDir, aName, aNamePt,anOri,anOut, aNameAuto = "NONE", aPrefix2Add;
    bool aForceGray = false;

    SaisieAppuisInit(argc, argv, aSzW, aNbFen, aFullName, aDir, aName, aNamePt, anOri, anOut, aNameAuto, aPrefix2Add, aForceGray);

    MainWindow w;
    w.setMode2D(true);
    w.resize( aSzW.x, aSzW.y );

    w.show();

   /*
        w.addFiles(aName);

    w.checkForLoadedData();*/

    return app.exec();
}
