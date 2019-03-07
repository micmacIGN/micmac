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

struct Moy4Pts
{
    bool All4Pts;
    Pt2dr MoyPt;
};

std::vector<int> ReadPtFile(string & aDir, string aPtFile)
{
    std::vector<int> aVPt;
    ifstream aFichier((aDir + aPtFile).c_str());
    if(aFichier)
    {
        std::string aLine;
        while(!aFichier.eof())
        {
            getline(aFichier,aLine,'\n');
            if(aLine.size() != 0)
            {
                char *aBuffer = strdup((char*)aLine.c_str());
                std::string aName = strtok(aBuffer,"	");
                int aPt = atof(aName.c_str());

                aVPt.push_back(aPt);
            }
        }
        aFichier.close();
    }
    else
    {
        std::cout<< "Error While opening file" << '\n';
    }
    return aVPt;
}

//verify the existence of all 4 points
Moy4Pts CalMoy(int aPt, std::list<cOneMesureAF1I> aLOneMAF1I)
{

    bool Pt1(false),Pt2(false),Pt3(false),Pt4(false),Pt5(false),Pt6(false),Pt7(false),Pt8(false);
    Moy4Pts aMoy4Pts;

    Pt2dr MoyPt(0.0,0.0);
    for (auto iT=aLOneMAF1I.begin();iT!=aLOneMAF1I.end();iT++)
    {
        int aMPt = atoi(iT->NamePt().c_str());
        if(aMPt==aPt+1) {Pt1=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+2) {Pt2=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+3) {Pt3=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+4) {Pt4=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+5) {Pt5=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+6) {Pt6=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+7) {Pt7=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}
        if(aMPt==aPt+8) {Pt8=1;MoyPt.x+=iT->PtIm().x;MoyPt.y+=iT->PtIm().y;}

    }
        aMoy4Pts.All4Pts = 0;
    if ((Pt1&&Pt2&&Pt3&&Pt4)||(Pt5&&Pt6&&Pt7&&Pt8))
        aMoy4Pts.All4Pts = 1;
    std::cout << "All4Pts=" << aMoy4Pts.All4Pts << endl;
    if(aMoy4Pts.All4Pts) aMoy4Pts.MoyPt = MoyPt/4;
    std::cout << "MoyPt=" << aMoy4Pts.MoyPt << endl;

    return aMoy4Pts;
}

int MoyMAF_main(int argc,char ** argv)
{
    string aMAFFile, aPtFile, aOut, aDir, aMAF;
    ElInitArgMain
     (
          argc, argv,
          LArgMain() << EAMC(aMAFFile,"MAF file", eSAM_IsExistFile)
                     << EAMC(aPtFile, "Pt file containing names of central pt and corresponding edge pts", eSAM_IsExistFile),
          LArgMain() << EAM(aOut,"Out",false,"Output MAF file")
     );

     SplitDirAndFile(aDir, aMAF, aMAFFile);

     //read MAF file
     cSetOfMesureAppuisFlottants aMAFset=StdGetFromPCP(aMAF,SetOfMesureAppuisFlottants);
     std::list<cMesureAppuiFlottant1Im> aLMAF = aMAFset.MesureAppuiFlottant1Im();
     std::cout << "Nb of Im: " << aLMAF.size() << endl;

     //read name point
     std::vector<int> aVPt = ReadPtFile(aDir,aPtFile);
     std::cout << "size of pt:" << aVPt.size() << endl;

     //create new MAF set
     std::list<cMesureAppuiFlottant1Im> aNewLMAF;


     //find corrending MAF of listed points and calculate the average
     for (auto iT=aLMAF.begin();iT!=aLMAF.end(); iT++)
     {
         std::list<cOneMesureAF1I> aLOneMAF1I = iT->OneMesureAF1I();
         std::cout << iT->NameIm() << endl;

         cMesureAppuiFlottant1Im aNewLOneMAF1I;
         //look for corresponding points of listed points
         for (uint iV=0; iV<aVPt.size();iV++)
         {
             cOneMesureAF1I aOneMAFMoy;
             Moy4Pts aMoy4Pts = CalMoy(aVPt.at(iV),aLOneMAF1I);
             if (aMoy4Pts.All4Pts)
             {
                 int aPt = aVPt.at(iV);
                 for (auto iT1=aLOneMAF1I.begin();iT1!=aLOneMAF1I.end();iT1++)
                 {
                     int aMPt = atoi(iT1->NamePt().c_str());
                     if ((aMPt>=aPt+1) && (aMPt<=aPt+8))
                     {
                         iT1 = aLOneMAF1I.erase(iT1);
                         iT1--;
                     }

                     std::string aNamePtMoy = std::to_string(aPt);
                     aOneMAFMoy.NamePt() = aNamePtMoy;
                     aOneMAFMoy.PtIm() = aMoy4Pts.MoyPt;
                 }
                 aLOneMAF1I.push_back(aOneMAFMoy);

                 aNewLOneMAF1I.NameIm()=iT->NameIm();
                 aNewLOneMAF1I.OneMesureAF1I()=aLOneMAF1I;
             }
         }
         aNewLMAF.push_back(aNewLOneMAF1I);
    }
    cSetOfMesureAppuisFlottants aNewMAFset;
    aNewMAFset.MesureAppuiFlottant1Im()=aNewLMAF;

    aOut = "Moy-"+aMAF;
    MakeFileXML(aNewMAFset,aOut);
    return EXIT_SUCCESS;
}

/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
associés au chargement,  a  l'utilisation,  a  la modification et/ou au
développement et a  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe a
manipuler et qui le réserve donc a  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités a  charger  et  tester  l'adéquation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
a  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
