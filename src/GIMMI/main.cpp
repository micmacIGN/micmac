
#include "NewProject.h"
#include "MainWindow.h"
#include "Tapioca.h"
#include "Console.h"
#include "StdAfx.h"





int main(int argc, char *argv[])
{


    QApplication app(argc, argv);
    MainWindow fenetre;
    fenetre.show();
    return app.exec();
}

