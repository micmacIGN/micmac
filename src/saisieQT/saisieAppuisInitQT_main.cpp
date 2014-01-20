#include "saisieQT_main.h"

using namespace std;

int saisieAppuisInitQT_main(QApplication &app, int argc, char *argv[])
{
    app.setApplicationName("SaisieAppuisInitQT");

    if ((argc>0)&&(string(argv[0]).find("SaisieQT")!= string::npos))
    {
        argv++;
        argc--;
    }

    Pt2di aSzW(800,600);
    Pt2di aNbFen(1,1);
    string aFullName, aDir, aName, aNamePt, aNameOri, aNameOut, aNameAuto, aPrefix2Add;
    aNameAuto = "NONE";
    aPrefix2Add = "";
    bool aForceGray = false;

    SaisieAppuisInit(argc, argv, aSzW, aNbFen, aFullName, aDir, aName, aNamePt, aNameOri, aNameOut, aNameAuto, aPrefix2Add, aForceGray);

    MainWindow w(aSzW, aNbFen, POINT2D_INIT, QString(aNamePt.c_str()));

    w.show();

    list<string> aNamelist = RegexListFileMatch(aDir, aName, 1, false);
    QStringList filenames;

    for
    (
        list<string>::iterator itS=aNamelist.begin();
        itS!=aNamelist.end();
        itS++
    )
        filenames.push_back( QString((aDir + *itS).c_str()));

    w.addFiles(filenames);

    return app.exec();
}
