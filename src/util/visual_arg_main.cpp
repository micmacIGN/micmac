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
#include "StdAfx.h"

#if(ELISE_QT5)
#ifdef Int
#undef Int
#endif
#include <QApplication>
#include <QString>
#include "general/visual_mainwindow.h"
#endif

void ShowEnum(const cMMSpecArg & anArg)
{
    const std::list<std::string>  & aLEnum = anArg.EnumeratedValues();
    for
            (
             std::list<std::string>::const_iterator itS = aLEnum.begin();
             itS != aLEnum.end();
             itS++
             )
        std::cout << "     " << *itS << "\n";
}

std::list<std::string> listPossibleValues(const cMMSpecArg & anArg)
{
    std::list<std::string> list_enum;

    if (anArg.IsBool())
    {
        list_enum.push_back("True");
        list_enum.push_back("False");
    }
    else
    {
        std::list<std::string>::const_iterator itS = anArg.EnumeratedValues().begin();
        for (; itS != anArg.EnumeratedValues().end(); itS++ )
        {
            //std::cout << "     " << *itS << "\n";
            list_enum.push_back(*itS);
            //i++;
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
    std::cout << "Enter Name + Val of optional arg, NONE if finish\n";
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
}

void MMRunVisualMode
(
        int argc,char ** argv, // A priori inutile, mais peut-etre cela evoluera-t-il ?
        std::vector<cMMSpecArg> & aVAM,  // Vector Arg Mandatory
        std::vector<cMMSpecArg> & aVAO   // Vector Arg Optional
        )
{

#if(ELISE_QT5)

    QApplication app(argc, argv);

    QFile file(app.applicationDirPath() + "/../src/uti_qt/style.qss");
    if(file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        app.setStyleSheet(file.readAll());
        file.close();
    }

    visual_MainWindow w(aVAM, aVAO);

    std::string arg_eff="";
    for (int i=0;i<argc;i++)
    {
        //std::cout<<argv[i]<<std::endl;

        arg_eff += std::string(argv[i]);
    }
    w.set_argv_recup(arg_eff);

    w.show();
    app.exec();

#endif //ELISE_QT5



    // On lit tous les arguments obligatoires
    //     for (int aK=0 ; aK<int(aVAM.size()) ; aK++)
    //     {
    //         // On imprime un peu d'info
    //         std::cout << "Enter Mandatory Arg " << aK << " ; Type is " << aVAM[aK].NameType() << "\n";
    //         std::string aCom = aVAM[aK].Comment();
    //         if (aCom != "") std::cout << "Comment=" << aCom << "\n";
    //         ShowEnum(aVAM[aK]);

    //         // on lit une chaine de caractere
    //         std::string aVal;
    //         std::cin >> aVal;
    //         // on initialise la variable a partir de la chaine
    //         aVAM[aK].Init(aVal);
    //     }

    //     // On lit autant d'arguments optionnels que l'utilisateur souhaite en passer
    //     bool FirstCall = true;
    //     while (ContinuerReadOneArg(aVAO,FirstCall))
    //     {
    //         FirstCall=false;
    //     }
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
