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

#include "general/all.h"
#include "private/all.h"
#include "XML_GEN/all.h"


/*
     Algorithmie de Conu
*/


void Banniere_Conu()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     Information chantier      *\n";
   std::cout <<  " *********************************\n";

}


using namespace NS_ParamChantierPhotogram;
using namespace NS_SuperposeImage;



namespace NS_InfoChantier
{

class cAppliInfoChantier;


/****************************************************/
/*                                                  */
/*          HEADER                                  */
/*                                                  */
/****************************************************/

class cPDV
{
    public :
       cPDV
       (
            const std::string &  aNameIm,
            const std::string &  aNameOrient,
            cAppliInfoChantier & anIAC
       ) :
           mName (aNameIm),
           mCam  (Cam_Gen_From_File(aNameOrient,"OrientationConique")),
           mAppli (anIAC),
           mActive (false)
       {
       }


       std::string          mName;
       ElCamera *         mCam;
       cAppliInfoChantier & mAppli;
       bool                 mActive;
};


class cAppliInfoChantier 
{
    public :
         cAppliInfoChantier(int argc,char ** argv);

	 void NoOp(){}

         void ShowVoisineImC();

    private :

        std::string  mDir;
        std::string  mKeySet;
        std::string  mKeyOrient;
        std::string mImCentrale;
        cInterfChantierNameManipulateur * mICNM;
        std::vector<cPDV *>               mPDVs;
        cPDV *                            mPdvC;

};



cAppliInfoChantier::cAppliInfoChantier(int argc,char ** argv) :
   mPdvC (0)
{
    ElInitArgMain
    (
           argc,argv,
           LArgMain() << EAM(mDir)
                      << EAM(mKeySet)
                      << EAM(mKeyOrient),
           LArgMain() << EAM(mImCentrale,"IC",true)
    );

    cTplValGesInit<std::string> aCompl;
    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDir,aCompl);
    
    const std::vector<std::string> * aSet = mICNM->Get(mKeySet);

    std::cout << "NB VUES " << aSet->size() << "\n";
    for (int aK=0 ; aK< int(aSet->size())  ; aK++)
    {
         const std::string & aNameIm = (*aSet)[aK];
         std::string aNameOri =  mDir + mICNM->Assoc1To1(mKeyOrient,aNameIm,true);
         mPDVs.push_back( new cPDV(aNameIm,aNameOri,*this));
 
         if ((aK%20)==0) 
         {
              std::cout << "RESTE " << (int(aSet->size()) - aK) << "\n";;
         }

         if (aNameIm== mImCentrale)
            mPdvC = mPDVs.back();
    }


    if (mPdvC)
    {
       for (int aKP=0 ; aKP<int(mPDVs.size()) ; aKP++)
       {
           cPDV * aCK = mPDVs[aKP];
           double aRatio = mPdvC->mCam->RatioInterSol(*(aCK->mCam));

           if (aRatio!=0)
           {
              std::cout << "For "<< aCK->mName << " Rec= " << aRatio << "\n";
              aCK->mActive=true;
           }
            
       }
    }
    
}


};


/*
 bin/InfoChantier /media/SQP/Vol_MPD/Quimper/Panchro45/  Key-Set-AllPanchro-OK-For-Init Key-Assoc-Im2OrInit

 
*/


/****************************************************/
/*                                                  */
/*            main                                  */
/*                                                  */
/****************************************************/

using namespace NS_InfoChantier;


int main(int argc,char ** argv)
{
    cAppliInfoChantier aAP(argc,argv);

    aAP.NoOp();
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
