#include "saisieQT_main.h"

int saisieAppuisInitQT_main(QApplication &app, int argc, char *argv[])
{
    app.setApplicationName("SaisieAppuisInitQT");

    const QString locale = QLocale::system().name().section('_', 0, 0);

    // qt translations
    QTranslator qtTranslator;
    qtTranslator.load(app.applicationName() + "_" + locale);
    app.installTranslator(&qtTranslator);

    MainWindow w;

    MMD_InitArgcArgv(argc,argv);
    Pt2di aSzW(800,800);
    Pt2di aNbFen(-1,-1);
    std::string aFullName,aNamePt,anOri,anOut;
    std::string aNameAuto = "NONE";
    std::string aPrefix2Add = "";
    bool aForceGray = true;

    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aFullName,"Full Name (Dir+Pattern)")
                      << EAMC(anOri,"Orientation ; NONE if not used")
                      << EAMC(aNamePt,"Name point")
                      << EAMC(anOut,"Output"),
          LArgMain()  << EAM(aSzW,"SzW",true,"Sz of Window")
                      << EAM(aNbFen,"NbF",true,"Nb Of Sub window (Def depends of number of images with max of 2x2)")
                      << EAM(aNameAuto,"NameAuto",true," Prefix or automatic point creation")
                      << EAM(aPrefix2Add,"Pref2Add",true," Prefix to add during import (for bug correction ?)")
                      << EAM(aForceGray,"ForceGray",true," Force gray image, def =true")
    );

    std::string aDir,aName;
    SplitDirAndFile(aDir,aName,aFullName);


    cInterfChantierNameManipulateur * aCINM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    const cInterfChantierNameManipulateur::tSet  *  aSet = aCINM->Get(aName);

    std::cout << "Nb Image =" << aSet->size() << "\n";
    ELISE_ASSERT(aSet->size()!=0,"No image found");

    if (aNbFen.x<0)
    {
       if (aSet->size() == 1)
       {
           aNbFen = Pt2di(1,2);
       }
       else if (aSet->size() == 2)
       {
           Tiff_Im aTF = Tiff_Im::StdConvGen(aDir+(*aSet)[0],1,false,true);
           Pt2di aSzIm = aTF.sz();
           aNbFen = (aSzIm.x>aSzIm.y) ? Pt2di(1,2) : Pt2di(2,1);
       }
       else
       {
           aNbFen = Pt2di(2,2);
       }
    }

    cResulMSO aRMSO = aCINM->MakeStdOrient(anOri,true);

    if (0)
    {
       std::cout  << "RMSO; Cam "  << aRMSO.Cam()
                  << " Nuage " <<  aRMSO.Nuage()
                  << " Ori " <<  aRMSO.IsKeyOri()
                  << "\n";
       getchar();
    }



    /*
    if (cmdline_args.size() > 1)
    {
        cmdline_args.pop_front();

        for (int i=0; i< cmdline_args.size(); ++i)
        {
            bool removeArg = false;

            str = cmdline_args[i];

            if (str.contains("help"))
            {
                QString help =  app.applicationName() + " [filename] [option=]\n\n"
                                "* [filename] string\t: open image\n\n"
                                "Options\n\n"
                                "* [Name=SzW] integer\t: set window width (default=800)\n"
                                "* [Name=Post] string\t: change postfix output file (default=_Masq)\n"
                                "* [Name=Name] string\t: set output filename (default=input+_Masq)\n"
                                "* [Name=Gama] REAL\t: apply gamma to image\n\n"
                                "Example: " + app.applicationName() + " IMG.tif SzW=1200 Name=PLAN Gama=1.5\n\n"
                                "NB: " + app.applicationName() + " can be run without any argument\n\n";

                return helpMessage(app, help);
            }

            w.setMode2D(true);

            if (str.contains("Post="))
            {
                w.setPostFix(str.mid(str.indexOf("Post=")+5, str.size()));

                removeArg = true;
            }

            if (str.contains("SzW="))
            {
                QString arg = str.mid(str.indexOf("SzW=")+4, str.size());
                int szW = arg.toInt();

                int szH = szW * w.height() / w.width();

                w.resize( szW, szH );

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
                QString strGamma = str.mid(str.indexOf("Gama=")+5, str.size());

                float aGamma = strGamma.toFloat();

                w.setGamma(aGamma);

                removeArg = true;
            }

            if (str.contains("SaisieAppuisInitQT"))
                removeArg=true;

            if (removeArg)
            {
                cmdline_args[i] = cmdline_args.back();
                cmdline_args.pop_back();
                i--;
            }
        }
    }*/

    w.show();

   /* if (cmdline_args.size() > 0)
        w.addFiles(cmdline_args);

    w.checkForLoadedData();*/

    return app.exec();
}
