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
#include <algorithm>

using namespace NS_ParamChantierPhotogram;

class cSom;
class cBSI;

class cSom
{
     public :
       cSom(const cBSI & aSB,const std::string & aName,const std::string & aNIns,const std::string & aNAero) ;

       const cBSI *        mBSI;
       std::string         mName;
       ElCamera            * mCamIns;
       ElCamera            * mCamAero;
       ElRotation3D          mRotI2A;
};


class cBSI
{
    public :
        friend class cSom;
        cBSI(int argc,char ** argv);
        cInterfChantierNameManipulateur * ICNM() const {return mICNM;}
        void  DoAll();

    private :
        std::string mDir;
        std::string mPat;
        std::string mKIns;
        std::string mKAero;
  
        double mDelta;
        cInterfChantierNameManipulateur * mICNM;
        std::list<std::string>  mLFile;
        std::vector<cSom >    mVC;
        int                    mNbSom;
        
};


cSom::cSom
(
        const cBSI & aBSI,
        const std::string & aName,
        const std::string & aNIns,
        const std::string & aNAreo
) :
    mBSI    (&aBSI),
    mName   (aName),
    mCamIns (Cam_Gen_From_File(aNIns,"OrientationConique",mBSI->ICNM())),
    mCamAero (Cam_Gen_From_File(aNAreo,"OrientationConique",mBSI->ICNM())),
    mRotI2A     (Pt3dr(0,0,0),0,0,0)
{
    // std::cout << aName << aNIns  << aNAreo << "\n";

   ElRotation3D aRInsC2M =   mCamIns->Orient().inv();
   ElRotation3D aRAreoC2M =  mCamAero->Orient().inv();


    mRotI2A  = aRAreoC2M.inv() * aRInsC2M;

   std::cout << aName << "  : " << mRotI2A.teta01() << " "  << mRotI2A.teta02() << " " << mRotI2A.teta12() << "\n";
   //ElRotation3D aBore
}




cBSI::cBSI(int argc,char ** argv) :
     mNbSom (-1)
{
    int mVitAff = 10;
      
    ElInitArgMain
    (
	argc,argv,
	LArgMain()  << EAM(mDir)
                    << EAM(mPat)
                    << EAM(mKIns)
                    << EAM(mKAero),
        LArgMain()  << EAM(mDelta,"Delta",true)
        
    );

    cTplValGesInit<std::string>  aTplFCND;
    mICNM = cInterfChantierNameManipulateur::StdAlloc(argc,argv,mDir,aTplFCND);
    mKIns = mICNM->StdKeyOrient(mKIns);
    mKAero = mICNM->StdKeyOrient(mKAero);

    mLFile =  mICNM->StdGetListOfFile(mPat);
    mNbSom =  mLFile.size();

    int aCpt = 0;
    for 
    (
         std::list<std::string>::const_iterator itS=mLFile.begin();
         itS!=mLFile.end();
         itS++
    )
    {
         std::string aNIns  = mDir + mICNM->Assoc1To1(mKIns ,*itS,true);
         std::string aNAero = mDir + mICNM->Assoc1To1(mKAero,*itS,true);

         mVC.push_back(cSom(*this,*itS,aNIns,aNAero));
         if ((aCpt %mVitAff) == (mVitAff-1)) 
         {
            std::cout << "Load  : remain " << (mNbSom-aCpt) << " to do\n";
         }
         aCpt++;
    }
}

void cBSI::DoAll()
{
}




int main(int argc,char ** argv)
{
    cBSI aBSI(argc,argv);
    aBSI.DoAll();
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
