#include "saisieQT_main.h"

using namespace std;

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

    Pt2di aSzWin(-1,-1);
    Pt2di aNbFen(-1,-1);

    string aFullName, aDir, aName, aNamePt, aNameOri, aNameOut, aNameAuto, aPrefix2Add;
    aNameAuto = "NONE";
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

    // dans InitWindows de SaisiePts A FACTORISER (methode virtuelle de Virtual_Interface...)
    int aNbW = aNbFen.x * aNbFen.y;
    if (filenames.size() < aNbW)
    {
         aNbW = filenames.size();
         aNbFen.x = round_up(sqrt(aNbW-0.01));
         aNbFen.y = round_up((double(aNbW)-0.01)/aNbFen.x);
    }

    bool init = settings.contains("MainWindow/size");

    settings.beginGroup("MainWindow");
    if (aSzWin.x > 0)
        settings.setValue("size", QSize(aSzWin.x, aSzWin.y));
    else if (!init)
        settings.setValue("size", QSize(800, 600));
    settings.setValue("NbFen", QPoint(aNbFen.x, aNbFen.y));
    settings.endGroup();

    settings.beginGroup("Misc");
    settings.setValue("defPtName", QString(aNamePt.c_str()));
    settings.endGroup();

    MainWindow w(POINT2D_INIT);

    w.show();

    w.addFiles(filenames);

    return app.exec();
}
