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
#include "im_tpl/image.h"

bool BugMM = false;
using namespace NS_SuperposeImage;

/*
  Extrait un modele radial d'une image de points homologues
*/

namespace NS_SimulePDV
{
class cOnePDV;
class cSimulePDV;

class cOnePDV
{
     public :
        cOnePDV
        (
             const cSimulePDV &,
             Pt2di  aNb
        );

     private :

        
        const cSimulePDV & mSim;
        Pt2di mIIndex;
        Pt2dr mRIndex;
        Pt2dr mC2Loc;
        Pt3dr mV3;  // Vecteur CentrePDV -> 
        Pt3dr mC3;

        Pt3dr mK;  // I,J,K trieder lie a la camera
        Pt3dr mI;  // 
        Pt3dr mJ;  // 
};


class cSimulePDV
{
     public :
        cSimulePDV(int argc,char ** argv);

        cInterfChantierNameManipulateur * mICNM;
        std::string mFullName;
        std::string mDir;
        std::string mNameCalib;
        CamStenope * mCam;
        double       mFoc;
        Pt2dr        mSzIm;

      // ecart pour n'avoir aucun recouvrement
        Pt2dr        mEcartNorm;
      // ecart pour avoir le recouvrement specifie
        Pt2dr        mEcartBH;
   
        Pt2di       mNb;
        double      mRecouvrt;
        double      mConv;
        double      mProf;
        Pt3dr       mCentreChantier;
        Pt3dr       mOffsV;
        

        std::vector<cOnePDV *>  mPDVs;
};

cOnePDV::cOnePDV
(
    const cSimulePDV & aSim,
    Pt2di  aIIndex
) :
   mSim       (aSim),
   mIIndex    (aIIndex),
   mRIndex    (Pt2dr(mIIndex) - (Pt2dr(aSim.mNb)-Pt2dr(1,1))/2.0),
   mC2Loc     (mRIndex.mcbyc(aSim.mEcartBH)),
   mV3        (
                      Pt3dr(mC2Loc.x,mC2Loc.y,0)
                   +  aSim.mOffsV
              ),
   mC3        ( mV3 +  aSim.mCentreChantier),
   mK         (  vunit
                 (
                     Pt3dr(0,0,-1)* (1-mSim.mConv)
                     -vunit(mV3) * mSim.mConv
                 )
              ),
   mI         (Pt3dr(0,1,0) ^ mK),
   mJ         (mK ^ Pt3dr(0,0,1))
{
    std::cout << mC2Loc << mC3 << "\n";
    std::cout << " IJK " << mI  << mJ   << mK << "\n";

    ElMatrix<double> aM0 =  MatFromCol(mI,mJ,mK);
std::cout << "BBBBbbbbb\n";
    ElMatrix<double> aMat =  NearestRotation(aM0);
std::cout << "AAAAAAAAAAAa\n";
    Pt3dr aI2,aJ2,aK2;
    aMat.GetCol(0,aI2);
    aMat.GetCol(1,aJ2);
    aMat.GetCol(2,aK2);
    std::cout << " IJK2 " << aI2  << aJ2   << aK2 << "\n";
}



cSimulePDV::cSimulePDV(int argc,char ** argv)
{
    mConv = 0.0;
    mProf = 10.0;
    mCentreChantier = Pt3dr(0,0,0);
    ElInitArgMain
    (
         argc,argv,
	 LArgMain()  << EAM(mFullName)
                     << EAM(mNb)
                     << EAM(mRecouvrt),
          LArgMain()  << EAM(mConv,"Conv",true)
	             << EAM(mProf,"Prof",true)
	             << EAM(mCentreChantier,"CC",true)
    );

    mOffsV = Pt3dr(0,0,mProf);

    mCam = Std_Cal_From_File(mFullName);
    mFoc = mCam->Focale();
    mSzIm = mCam->Sz();

    mEcartNorm = mSzIm * (mProf/mFoc);
    mEcartBH   = mEcartNorm * (1-mRecouvrt);


    std::cout << "FOCALE " <<  mFoc  << " EcN " << mEcartNorm << "\n";

    SplitDirAndFile(mFullName,mDir,mNameCalib);
    cTplValGesInit<std::string> aName;

    mICNM = cInterfChantierNameManipulateur::StdAlloc
            (
                argc, argv,
                mDir,aName
            );

    Pt2di aP;
    for (aP.x = 0 ; aP.x <mNb.x ; aP.x++)
    {
       for (aP.y = 0 ; aP.y <mNb.y ; aP.y++)
       {
            mPDVs.push_back(new  cOnePDV(*this,aP));
       }
    }
}



};

using namespace NS_SimulePDV;

int main(int argc,char ** argv)
{
    cSimulePDV aSim(argc,argv);

    return 0;
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
