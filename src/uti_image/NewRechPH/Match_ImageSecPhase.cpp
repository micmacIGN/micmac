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


/*
    Acceleration :
      - Presel sur point les plus stables
      - calcul de distance de stabilite ? => Uniquement si pas Invar Ech !!!!
      - Apres pre-sel, a simil (ou autre) :
             * selection des point dans  regions homologues
             * indexation

    Plus de points :
        SIFT Criteres ?
*/

#include "NewRechPH.h"
#include "Match_Image.h"



/*************************************************/
/*                                               */
/*           cBiaisedRandGenerator               */
/*                                               */
/*************************************************/


cBiaisedRandGenerator::cBiaisedRandGenerator(const std::vector<double> & aVec) :
   mNb  (aVec.size())
{
    mCumul.push_back(0.0);
    for (const auto & aVal : aVec)
    {
       ELISE_ASSERT(aVal>=0,"cBiaisedRandGenerator val<0");
       mCumul.push_back(mCumul.back()+aVal);
    }
    double aVMax = mCumul.back();

    ELISE_ASSERT(aVMax!=0,"cBiaisedRandGenerator no val >0");

    for (auto & aVal : mCumul)
    {
       aVal /= aVMax;
       std::cout << "VvVvVvvv " << aVal << "\n";
    }
}

int cBiaisedRandGenerator::Generate()
{
    return Generate(NRrandom3());
}

int cBiaisedRandGenerator::Generate(double aVal)
{
    if (aVal<=0) return 0;
    
    std::vector<double>::iterator aLB = lower_bound(mCumul.begin(),mCumul.end(),aVal);
    // std::vector<double>::iterator aLB = upper_bound(mCumul.begin(),mCumul.end(),aVal);
    ELISE_ASSERT (aLB != mCumul.end(),"cBiaisedRandGenerator::Generate");
    return aLB - mCumul.begin()-1;
}

void TestcBiaisedRandGenerator()
{
   std::vector<double>  aV;
   aV.push_back(1.0);
   aV.push_back(2.0);
   aV.push_back(4.0);
   aV.push_back(1.0);
   cBiaisedRandGenerator aCBR(aV);

   std::cout << "Grrr " << aCBR.Generate(-0.01) << "\n";
   std::cout << "Grrr " << aCBR.Generate(0.0) << "\n";
   std::cout << "Grrr " << aCBR.Generate(0.001) << "\n";
   std::cout << "Grrr " << aCBR.Generate(0.2) << "\n";
   std::cout << "Grrr " << aCBR.Generate(0.99) << "\n";
   std::cout << "Grrr " << aCBR.Generate(1.0) << "\n";

   getchar();
   

   cFHistoInt aFH;
   for (int aK=0 ; aK<1000000 ; aK++)
   {
       int  aG = aCBR.Generate();
// std::cout << "Gggggg " << aG << "\n";
       aFH.Add(aG);
   }
   for (int aK=0 ; aK<int(aV.size()) ; aK++)
   {
       std::cout << "BiasedRanGen " << aFH.Perc(aK) / aV[aK] << "\n";
   }


/*
   std::cout << "TTT " << aCBR.Test(-0.001) << "\n";
   std::cout << "TTT " << aCBR.Test(0) << "\n";
   std::cout << "TTT " << aCBR.Test(0.1) << "\n";
   std::cout << "TTT " << aCBR.Test(1.0) << "\n";
   std::cout << "TTT " << aCBR.Test(2.0) << "\n";
   std::cout << "TTT " << aCBR.Test(3.0) << "\n";
   std::cout << "TTT " << aCBR.Test(7.0) << "\n";
   std::cout << "TTT " << aCBR.Test(8.0) << "\n";
   std::cout << "TTT " << aCBR.Test(8.1) << "\n";
   std::cout << "TTT " << aCBR.Test(1999998.1) << "\n";

   // double * aV = lower_bound(
    // emplate <class ForwardIterator, class T>
  // ForwardIterator lower_bound (ForwardIterator first, ForwardIterator last, const T& val);
*/
}




/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/


//  aTBuf.oset(Pt2di(aKRho,aKTeta),aVal);


double cAppli_FitsMatch1Im::DistHistoGrad(cCompileOPC & aMast,int aShift,cCompileOPC & aSec)
{
    Pt2di aSzInit = aMast.mSzIm;
    Pt2di aSzG (aSzInit.x-1,aSzInit.y);

    TIm2D<INT1,INT>  aImM (aMast.mOPC.ImLogPol());
    TIm2D<INT1,INT>  aImS ( aSec.mOPC.ImLogPol());

    double aSomEcPds = 0;
    double aSomPds = 0;

/*
    Im2D_REAL4 aMGx(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMGx(aMGx);
    Im2D_REAL4 aMGy(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMGy(aMGy);
    double aSumGM = 0;
    Im2D_REAL4 aMN(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTMN(aMN);

    Im2D_REAL4 aSGx(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSGx(aSGx);
    Im2D_REAL4 aSGy(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSGy(aSGy);
    double aSumGS = 0;
    Im2D_REAL4 aSN(aSzG.x,aSzG.y);
    TIm2D<REAL4,REAL8> aTSN(aSN);
*/
    
    double aSomEcRad =0;
    double aSomEcGrad =0;
    int aNb=0;
    for (int aKRho=0 ; aKRho<aSzG.x; aKRho++)
    {
        for (int aKTeta=0 ; aKTeta<aSzG.y; aKTeta++)
        {
             Pt2di aPM (aKRho  ,  aKTeta);
             Pt2di aPMr(aPM.x+1,  aPM.y);
             Pt2di aPMt(aPM.x  , (aPM.y+1)%aSzG.y);


             Pt2di aPS  (aKRho  ,  (aKTeta+aShift + aSzG.y) %aSzG.y);
             Pt2di aPSr (aPS.x+1,aPS.y);
             Pt2di aPSt (aPS.x  ,(aPS.y+1)%aSzG.y);

             Pt2dr aGradM (aImM.get(aPM)-aImM.get(aPMr),aImM.get(aPM)-aImM.get(aPMt));
             double aNormM = euclid(aGradM);

             Pt2dr aGradS (aImS.get(aPS)-aImS.get(aPSr),aImS.get(aPS)-aImS.get(aPSt));
             double aNormS = euclid(aGradS);
             double aPds = pow(aNormM*aNormS,ExposantPdsDistGrad()/2.0); //  2.0 prend en compte la racine carre
             // double aPds = sqrt(aNormM*aNormS);
             aSomEcRad += ElAbs(aImM.get(aPM)-aImS.get(aPS));
             aSomEcGrad += dist4(aGradM-aGradS);
             aNb++;
             if (aPds > 0)
             {
                aGradM = aGradM / aNormM;
                aGradS = aGradS / aNormS;
                Pt2dr aPEcart = aGradM / aGradS;
                aPEcart = aPEcart - Pt2dr(1,0);
                aSomEcPds +=  aPds * euclid(aPEcart);
                aSomPds += aPds;
             }
        }
    }
/*
Pt2dr aPM = aMast.mOPC.Pt();
Pt2dr aPS = aSec.mOPC.Pt();
std::cout << "PppPpp " << aPM << " " << aPS << (aPM+aPS) / 2.0 << aPM-aPS 
          << " ECART RAD " << aSomEcRad/aNb 
          << " ECART Grad " << aSomEcGrad/aNb 
          << "\n";
std::cout << "mShitfBestmShitfBest " << aMast.mShitfBest 
          << " " << (aSomEcPds / aSomPds) 
          <<  aSzInit 
          << "\n";
*/

    return aSomEcPds / aSomPds;
}



Pt2dr cPrediCoord::Predic(const Pt2dr & aP) const
{
   Pt2dr aPInd = aP / mFacRed; 
   // Pt2dr aPDif = aPS-(*aMap)(aPM);
   return (*mMap)(aP) + Pt2dr(mTImX.getprojR(aPInd),mTImY.getprojR(aPInd));
}



cPrediCoord::cPrediCoord(Pt2di aSzGlob,int  aNbPix) :
   mSzGlob (aSzGlob),
   mFacRed (dist8(mSzGlob)/double(aNbPix)),
   mMap    (0),
   mSzRed  (round_up(Pt2dr(mSzGlob)/ mFacRed)),
   mImX    (mSzRed.x,mSzRed.y,0.0),
   mTImX   (mImX),
   mImY    (mSzRed.x,mSzRed.y,0.0),
   mTImY   (mImY),
   mImPds  (mSzRed.x,mSzRed.y,0.0),
   mTImPds (mImPds),
   mImInc  (mSzRed.x,mSzRed.y,0.0),
   mTImInc (mImInc)
{
}
 

void cPrediCoord::Init(double aMulDist,cElMap2D * aMap,const std::vector<cCdtCplHom> aVC) 
{
    mMap = aMap;
    for (const auto & aCpl : aVC)
    {
         Pt2dr aPM = aCpl.PM() ;
         Pt2dr aPS = aCpl.PS() ;
         Pt2dr aPDif = aPS-(*aMap)(aPM);
         double aPds = 1.0;
         Pt2dr aPInd = aPM / mFacRed;
         mTImPds.incr(aPInd,aPds);
         mTImX.incr(aPInd,aPds*aPDif.x);
         mTImY.incr(aPInd,aPds*aPDif.y);
    }

    double aSurfMoy = (mSzRed.x * mSzRed.y) / ElMax(1,int(aVC.size()));
    double aDistMoy = sqrt(aSurfMoy);
    aDistMoy *= aMulDist;
    double aFact = FactExpFromSigma2(ElSquare(aDistMoy));
    FilterExp(mImPds,aFact);
    FilterExp(mImX,aFact);
    FilterExp(mImY,aFact);

    ELISE_COPY
    (
       mImX.all_pts(),
       Virgule(mImX.in(),mImY.in())/Max(1e-10,mImPds.in()),
       Virgule(mImX.out(),mImY.out())
    );
    

    // Tiff_Im::CreateFromIm(mImX,"ImX.tif");
    // Tiff_Im::CreateFromIm(mImY,"ImY.tif");
    // double 
    // double FactExpFromSigma2(double aS2)

/*
    for (const auto & aCpl : aVC)
    {
         Pt2dr aPM = aCpl.PM() ;
         Pt2dr aPS = aCpl.PS() ;
         std::cout << "Dif " << Predic(aPM) - aPS << "\n";
    }
*/

    Im2D_REAL8 aImNorm(mSzRed.x,mSzRed.y);
    double aSomNorm = 0.0;
    double aNbRed = mSzRed.x * mSzRed.y;
    ELISE_COPY(mImX.all_pts(),sqrt(Square(mImX.in())+Square(mImY.in())),aImNorm.out() | sigma(aSomNorm));
    aSomNorm /= aNbRed;
 
    std::cout << "NORM=" << aSomNorm << "\n";

    int aSzMaxMin = euclid(mSzRed) / 10.0;
    Im2D_REAL8 aImVarX(mSzRed.x,mSzRed.y);
    Im2D_REAL8 aImVarY(mSzRed.x,mSzRed.y);

    ELISE_COPY(mImX.all_pts(),rect_max(mImX.in(-1e10),aSzMaxMin)-rect_min(mImX.in(1e10),aSzMaxMin),aImVarX.out());
    ELISE_COPY(mImY.all_pts(),rect_max(mImY.in(-1e10),aSzMaxMin)-rect_min(mImY.in(1e10),aSzMaxMin),aImVarY.out());

    // Tiff_Im::CreateFromIm(aImVarX,"IVarX.tif");
    // Tiff_Im::CreateFromIm(aImVarY,"IVarY.tif");
   

    ELISE_COPY
    (
         mImInc.all_pts(),
         aImNorm.in()/5.0 + aSomNorm/12.0 + Max(aImVarX.in(),aImVarY.in()) + euclid(mSzGlob) /100.0,
         mImInc.out()
    );

    // Tiff_Im::CreateFromIm(mImInc,"ImInc.tif");
}

void  OneTestcPrediCoord(double aDist)
{
   std::vector<cCdtCplHom>  aVC;
   ElSimilitude aSim;
   cPrediCoord aPC(Pt2di(2000,2000),1000);
   aPC.Init(aDist,&aSim,aVC);
}

void OneTestcPrediCoord()
{
   OneTestcPrediCoord(2.0);
   OneTestcPrediCoord(5.0);
   OneTestcPrediCoord(10.0);
   OneTestcPrediCoord(20.0);
   OneTestcPrediCoord(50.0);
   OneTestcPrediCoord(100.0);
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
aooter-MicMac-eLiSe-25/06/2007*/
