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

#include "schnaps.h"

/**
 * Homol2GND: Creates fake ground points for aerotriangulation wedge
 *
 * */


//----------------------------------------------------------------------------

int Homol2GND_main(int argc,char ** argv)
{
    std::string aFullPattern;//images pattern
    std::string aInHomolDirName="";//input Homol dir suffix
    //std::string mOri;//images orientation dir
    int aNbPts=4;//Nb points
    //double aPts3DSigma=0.01;//3d points sigma (m)
    std::string outName="homol2gnd_out";
    std::string ptsName="fakePt";
    bool ExpTxt=false;//Homol are in dat or txt

    ElInitArgMain
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  <<  EAMC(aFullPattern, "Pattern images",  eSAM_IsPatFile),
                //<<  EAMC(mOri,"Orientation",eSAM_IsDir),

    //optional arguments
    LArgMain()  << EAM(aInHomolDirName, "SH", true, "Input Homol directory suffix (without \"Homol\")")
                << EAM(aNbPts, "nbPts", true, "Number of out points (default=4)")
                //<< EAM(aPts3DSigma, "3dSigma", true, "Sigma for 3d points (default 0.01m)")
                << EAM(outName, "out", true, "out filename base (defaut \"homol2gnd_out\")")
                << EAM(ptsName, "ptsName", true, "fake points basename (defaut \"fakePt\")")
                << EAM(ExpTxt,"ExpTxt",true,"Ascii format for in and out, def=false")
    );

    std::cout<<"Homol2GND: Creates fake ground points for aerotriangulation wedge"<<std::endl;
    if (MMVisualMode) return EXIT_SUCCESS;

    //search for good homol points on best image
    int nbCells=4*aNbPts-sqrt(aNbPts)*2*2+1; //works for aNbPts=x², to have one used cell for 4 cells (to force dispersion)

    std::cout<<"Looking for "<<aNbPts<<" points.\n";
    // Initialize name manipulator & files
    std::string aDirImages,aPatIm;
    SplitDirAndFile(aDirImages,aPatIm,aFullPattern);
    std::cout<<"Working dir: "<<aDirImages<<std::endl;
    std::cout<<"Images pattern: "<<aPatIm<<std::endl;

    StdCorrecNameHomol(aInHomolDirName,aDirImages);

    cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    const std::vector<std::string> aSetIm = *(aICNM->Get(aPatIm));

    // Init Keys for homol files
    std::list<cHomol> allHomols;
    std::string anExt = ExpTxt ? "txt" : "dat";
    std::string aKHIn =   std::string("NKS-Assoc-CplIm2Hom@")
            +  std::string(aInHomolDirName)
            +  std::string("@")
            +  std::string(anExt);

    CompiledKey2 aCKin(aICNM,aKHIn);

    //create pictures list, and pictures size list
    std::map<std::string,cPic*> allPics;

    std::cout<<"Found "<<aSetIm.size()<<" pictures."<<endl;

    computeAllHomol(aDirImages,aPatIm,aSetIm,allHomols,aCKin,allPics,false,nbCells);

    std::cout<<"Found "<<allHomols.size()<<" homols."<<endl;


    //select homols on central pic
    cout<<"Central pic: "<<aSetIm[aSetIm.size()/2]<<std::endl;
    cPic* centralPic=0;

    std::map<std::string,cPic*>::iterator itPic1;
    /*for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic *pic1=(*itPic1).second;
        std::cout<<" Picture "<<pic1->getName()<<std::endl;
    }*/

    //create new homols ------------------------------------------------
    std::cout<<"Create new homol..\n";
    for (itPic1=allPics.begin();itPic1!=allPics.end();++itPic1)
    {
        cPic* aPic=(*itPic1).second;
        if (aPic->getName()==aSetIm[aSetIm.size()/2])
        {
            centralPic=aPic;
            aPic->selectHomols();
            std::cout<<"Central picture: "<<aPic->getName()<<": "<<aPic->getAllSelectedPointsOnPicSize()<<std::endl;
        }
    }
    std::cout<<"Done!"<<endl;


    //keep only points on even/even windows (to avoid points too close)
    std::vector<cHomol*> allFinalHomols;

    std::cout<<"Homol points on central pic:\n";
    std::map<double,cPointOnPic*>::iterator itHomolPoint;
    for (itHomolPoint=centralPic->getAllSelectedPointsOnPic()->begin();
         itHomolPoint!=centralPic->getAllSelectedPointsOnPic()->end();
         ++itHomolPoint)
    {
        cPointOnPic* aPoP=(*itHomolPoint).second;
        if (aPoP->getHomol()->isBad()) continue;
        int x=aPoP->getPt().x/centralPic->getPicSize()->getWinSz().x;
        int y=aPoP->getPt().y/centralPic->getPicSize()->getWinSz().y;
        //std::cout<<aPoP->getPt().x<<" "<<aPoP->getPt().y<<" "<<aPoP->getHomol()->getPointOnPicsSize()<<" ("<<x<<","<<y<<")"<<std::endl;
        if ((x%2==0)&&(y%2==0))
        {
            allFinalHomols.push_back(aPoP->getHomol());
            static char  buf[200];
            sprintf(buf,"%s_%02d",ptsName.c_str(),(int)allFinalHomols.size()-1);
            std::cout<<"  * Selected homol: "<<buf<<" "<<aPoP->getPt().x<<" "<<aPoP->getPt().y
                    <<" (multiplicity "<<aPoP->getHomol()->getPointOnPicsSize()<<").\n";
        }
    }
    if ((int)allFinalHomols.size()<aNbPts)
    {
        for (itHomolPoint=centralPic->getAllSelectedPointsOnPic()->begin();
             itHomolPoint!=centralPic->getAllSelectedPointsOnPic()->end();
             ++itHomolPoint)
        {
            cPointOnPic* aPoP=(*itHomolPoint).second;
            if (aPoP->getHomol()->isBad()) continue;
            int x=aPoP->getPt().x/centralPic->getPicSize()->getWinSz().x;
            int y=aPoP->getPt().y/centralPic->getPicSize()->getWinSz().y;
            //std::cout<<aPoP->getPt().x<<" "<<aPoP->getPt().y<<" "<<aPoP->getHomol()->getPointOnPicsSize()<<" ("<<x<<","<<y<<")"<<std::endl;
            if ((x%2==1)&&(y%2==1)&&(((int)allFinalHomols.size()<aNbPts)))
            {
                allFinalHomols.push_back(aPoP->getHomol());
                static char  buf[200];
                sprintf(buf,"%s_%02d",ptsName.c_str(),(int)allFinalHomols.size()-1);
                std::cout<<"  * Selected homol: "<<buf<<" "<<aPoP->getPt().x<<" "<<aPoP->getPt().y
                        <<" (multiplicity "<<aPoP->getHomol()->getPointOnPicsSize()<<").\n";
            }
        }
    }

    std::cout<<"Homol points on central pic:\n";
    //create out 2D files
    cSetOfMesureAppuisFlottants aMes2dList;

    for (unsigned int i=0;i<allFinalHomols.size();i++)
    {
        cHomol* aHomol=allFinalHomols[i];
        static char  buf[200];
        sprintf(buf,"%s_%02d",ptsName.c_str(),i);
        for (unsigned int j=0;j<aHomol->getPointOnPicsSize();j++)
        {
            cPointOnPic *aPoP=aHomol->getPointOnPic(j);
            cMesureAppuiFlottant1Im aListMes1Im;
            aListMes1Im.NameIm()=aPoP->getPic()->getName();

            cOneMesureAF1I aOneMesIm;
            aOneMesIm.NamePt()=buf;
            aOneMesIm.PtIm()=aPoP->getPt();
            aListMes1Im.OneMesureAF1I().push_back( aOneMesIm );
            aMes2dList.MesureAppuiFlottant1Im().push_back( aListMes1Im );
        }
    }
    std::cout<<"Written to "<<aDirImages + outName+".xml\n";
    MakeFileXML(aMes2dList, aDirImages + outName+".xml");



    std::cout<<"Finished!"<<std::endl;

     return EXIT_SUCCESS;
}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systèmes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
