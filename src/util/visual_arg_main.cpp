/*Header-MicMac-eLiSe-25/06/2007

MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
eLiSe  : ELements of an Image Software Environnement

www.micmac.ign.fr


Copyright : Institut Geographique National
Author : Marc Pierrot Deseilligny
Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
"A multiresolution and optimization-based image matching approach:
An application to surface reconstruction from SPOT5-HRS stereo imagery."
In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
(With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
d'images, adapte au contexte geograhique" to appears in
Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

MicMac est un logiciel de mise en correspondance d'image adapte
au contexte de recherche en information geographique. Il s'appuie sur
la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

MicMac is an open source software specialized in image matching
for research in geographic information. MicMac is built on the
eLiSe image library. MicMac is governed by the  "Cecill-B licence".
See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "general/CMake_defines.h"

#if ELISE_QT
	#include "general/visual_mainwindow.h"
#else
	#include "StdAfx.h"
#endif

void ShowEnum(const cMMSpecArg & anArg)
{
    list<string>::const_iterator itS = anArg.EnumeratedValues().begin();
    for (; itS != anArg.EnumeratedValues().end(); itS++ )
        cout << "     " << *itS << "\n";
}

list<string> listPossibleValues(const cMMSpecArg & anArg)
{
    list<string> list_enum;

    if (anArg.IsBool() || anArg.Type() == AMBT_bool)
    {
        list_enum.push_back("True");
        list_enum.push_back("False");
    }
    else
    {
        list<string>::const_iterator itS = anArg.EnumeratedValues().begin();
        for (; itS != anArg.EnumeratedValues().end(); itS++ )
        {
            list_enum.push_back(*itS);
        }
    }
    return list_enum;
}


// ====================================================================
//
//      Lecture d'un argument optionnel; utilise par MMRunVisualMode
//      retourne false si on specifie la fin de lecture;
//
//=====================================================================


/*
bool ContinuerReadOneArg(std::vector<cMMSpecArg> & aVAO, bool Prems)
{
    // la premiere fois on imprime toute l'info sur tous les arguments
    if (Prems)
    {
        for (int aK=0 ; aK<int(aVAO.size()) ; aK++)
        {
            std::cout <<  aVAO[aK].NameArg()  << " ; "  << aVAO[aK].NameType();
            std::string aCom = aVAO[aK].Comment();
            if (aCom != "") std::cout << " ; " << aCom ;
            std::cout  << "\n";
            ShowEnum(aVAO[aK]);
        }
    }

    // Lecture du nom et de la valeur
    cout << "Enter Name + Val of optional arg, NONE if finish\n";
    std::string aName,aVal;

    std::cin >> aName >> aVal;


    // Si on veut signifier la fin il faut taper NONE  xxx
    if (aName=="NONE") return false;

    // Si on trouve le bon nom, on initialise et on retourne true
    for (int aK=0 ; aK<int(aVAO.size()) ; aK++)
    {
        if (aVAO[aK].NameArg() ==aName)
        {
            aVAO[aK].Init(aVal);
            return true;
        }
    }

    // Sinon un message d'insulte (light) et on continue
    std::cout << "Name is not valid !!! (got " << aName << ")\n";
    return true;
}*/

#if ELISE_QT
void showErrorMsg(QApplication &app, std::vector <std::string> vStr)
{
    QString str("In visual mode, possible values are:\n");

    QString msg;
    for (int aK=0; aK < (int)vStr.size(); ++aK)
        msg += QString("\nv")+ app.applicationDisplayName() + QString(" ") + QString(vStr[aK].c_str());
    setStyleSheet(app);
    QMessageBox::critical(NULL, "Error", str + msg);
}
#endif

int MMRunVisualMode
(
        int argc,char ** argv,
        std::vector<cMMSpecArg> & aVAM,  // Vector Arg Mandatory
        std::vector<cMMSpecArg> & aVAO,  // Vector Arg Optional
        std::string aFirstArg            // Command name
        )
{

#if ELISE_QT
    bool deleteAPp = QApplication::instance() == NULL;

    QApplication *app = (QApplication::instance() == NULL)   ? new QApplication(argc, argv) : static_cast<QApplication *>(QApplication::instance());

    app->setApplicationName("MMVisualMode");
    app->setOrganizationName("Culture3D");

    QSettings settings(QApplication::organizationName(), QApplication::applicationName());

    settings.beginGroup("FilePath");
    QString lastDir = settings.value( "Path", QDir::currentPath() ).toString();
    settings.endGroup();

    setStyleSheet(*app);

    visual_MainWindow * w = new visual_MainWindow(aVAM, aVAO, aFirstArg, lastDir);

    w->set_argv_recup(string(argv[0]));

    w->setAttribute(Qt::WA_DeleteOnClose);

    w->show();

    int appReturn = app->exec();

    if(deleteAPp)
        delete app;

    return appReturn;

#endif // ELISE_QT
    return EXIT_SUCCESS;
}


/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
