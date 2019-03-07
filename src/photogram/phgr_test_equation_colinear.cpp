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


class cGeneratorEqColLin
{
    public :
       cGeneratorEqColLin();
   private:
       cSetEqFormelles mSet;
       AllocateurDInconnues &  mAlloc;
       Pt3dr                   mPTer;
       Pt3d<Fonc_Num>          mFPTer;
       Pt3dr                   mCPerspectif;
       Pt3d<Fonc_Num>          mFCPerspectif;
       Pt3dr                   mPAngles;
       Pt3d<Fonc_Num>          mPFAngles;
       ElMatrix<Fonc_Num>      mRot;

       Pt2dr                   mPP;
       Pt2d<Fonc_Num>          mFPP;
       double                  mFoc;
       Fonc_Num                mFFoc;
       cP2d_Etat_PhgrF         mFPObsIm;
       double                  mDR1;
       Fonc_Num                mFDR1;
       double                  mDR2;
       Fonc_Num                mFDR2;
};

cGeneratorEqColLin::cGeneratorEqColLin() :
    mSet          (),
    mAlloc        (mSet.Alloc()),
    mPTer         (0,0,10),
    mFPTer        (mAlloc.NewPt3("Pter",mPTer)),
    mCPerspectif  (0,0,0),
    mFCPerspectif (mAlloc.NewPt3("CPersp",mCPerspectif)),
    mPAngles      (0,0,0),
    mPFAngles     (mAlloc.NewPt3("Angles",mPAngles)),
    mRot          (ElMatrix<Fonc_Num>::Rotation(mPFAngles.x,mPFAngles.y,mPFAngles.z)),
    mPP           (2000,2000),
    mFPP          (mAlloc.NewPt2("PP",mPP)),
    mFoc          (10000),
    mFFoc         (mAlloc.NewF("Foc","Foc",&mFoc)),
    mFPObsIm      ("ObsIm"),
    mDR1          (0.0),
    mFDR1         (mAlloc.NewF("DR1","DR1",&mDR1)),
    mDR2          (0.0),
    mFDR2         (mAlloc.NewF("DR2","DR2",&mDR2))
{
    std::cout << "cGeneratorEqColLin : \n";
    Fonc_Num aForm0 = (mFPTer.x + mFPTer.y) / tan(mFPTer.z);
    Fonc_Num aForm0Deriv = aForm0.deriv(2);
    aForm0.show(std::cout); std::cout << "\n";
    aForm0Deriv.show(std::cout); std::cout << "\n";

    mFCPerspectif.x.show(std::cout); std::cout << "\n";
    mRot(0,0).show(std::cout);  std::cout << "\n";
    mRot(1,1).show(std::cout);  std::cout << "\n";
    // std::cout << "M00 " << mRot(0,0).val() << "\n";

    Pt3d<Fonc_Num> aFPCam =  mRot * (mFPTer-mFCPerspectif);
    Pt2d<Fonc_Num> aFDir(aFPCam.x/aFPCam.z,aFPCam.y/aFPCam.z);

    Fonc_Num  aRho2 = ElSquare(aFDir.x) +  ElSquare(aFDir.y);
    Fonc_Num  aRho4 = aRho2 * aRho2;
    Fonc_Num  aFactCorrec = (1+ mFDR1 * aRho2 + mFDR2 * aRho4);
    aFDir =  Pt2d<Fonc_Num>(aFDir.x*aFactCorrec,aFDir.y*aFactCorrec);



    Pt2d<Fonc_Num>  aFProjIm (aFDir.x * mFFoc + mFPP.x, aFDir.y * mFFoc + mFPP.y);
    Pt2d<Fonc_Num> aFResidu = mFPObsIm.PtF() - aFProjIm;

    aFResidu.x.show(std::cout) ;  std::cout << "\n";

    cIncIntervale aInt("Glob",0,14);
    cIncListInterv aLInterv;
    aLInterv. AddInterv(aInt);

    std::vector<Fonc_Num> aVResidu;
    aVResidu.push_back(aFResidu.x);
    aVResidu.push_back(aFResidu.y);

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,
        "TestCeresColinearity",
        // aFResidu.x,
        aVResidu,
        aLInterv
    );
    
}

/*
void TestcGeneratorEqColLin()
{
    cGeneratorEqColLin aGen;
}
*/

// #include "../../CodeGenere/photogram/TestCeresColinearity.h"
#include "../../CodeGenere/photogram/TestCeresColinearity.cpp"


class cTestTestCeresColinearity  : public TestCeresColinearity
{
    public :
       cTestTestCeresColinearity()
       {
            for (int aK=0 ; aK<14 ; aK++)
            {
                mCompCoord[aK] = aK;
            }
       }

       void Test(int aNb)
       {
           {
                ElTimer aChrono;
                for (int aK=0 ; aK<aNb ; aK++)
                {
                     ComputeVal();
                }
                std::cout << "Time Val " << aChrono.uval() << "\n";
           }
           {
                ElTimer aChrono;
                for (int aK=0 ; aK<aNb ; aK++)
                {
                     ComputeValDeriv();
                }
                std::cout << "Time Derive " << aChrono.uval() << "\n";
           }
       }
};

void TestcGeneratorEqColLin()
{
    cTestTestCeresColinearity aTTCC;
    aTTCC.Test(347000*25);
     
/*
    TestCeresColinearity aTC;
    for (int aK=0 ; aK<14 ; aK++)
    {
        aTC.mCompCoord[aK] = aK;
    }
*/
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
