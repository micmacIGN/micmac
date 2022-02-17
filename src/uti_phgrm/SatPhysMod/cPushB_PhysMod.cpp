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


#include "SatPhysMod.h"

static double TheCsteGravi = 6.67384e-11;
static double TheMasseTerre = 5.972e24;
static double TheRayTerre = 6.371e6;

/******************************************************/
/*                                                    */
/*           cPushB_PhysMod                           */
/*                                                    */
/******************************************************/

const double cPushB_PhysMod::TheEpsilonRefine = 1e-7;

cPushB_PhysMod::cPushB_PhysMod(const Pt2di & aSz,eModeRefinePB aModeRefine,const Pt2di & aSzGeoL) :
   mSz            ( ((aSz + Pt2di(1,1))/2) * 2),  // On rend pair les nombres
   mSzGeoL        (aSzGeoL),
   mModeRefine    (aModeRefine),
   mSwapXY        (false)
{
}

double cPushB_PhysMod::CoherAR(const Pt2dr & aPIm) const
{
   ElSeg3D aSeg = Im2GeoC(aPIm);
   return euclid(aPIm,GeoC2Im(aSeg.P0())) + euclid(aPIm,GeoC2Im(aSeg.P1()));
}

void cPushB_PhysMod::CoherARGlob(int aNbPts,double & aMoyCoh,double & aMaxCoh) const
{
    aMoyCoh = 0.0;
    aMaxCoh = 0.0;

    for (int anX=0 ; anX<=aNbPts ; anX++)
    {
        for (int anY=0 ; anY<=aNbPts ; anY++)
        {
            Pt2dr aPIm( (double(anX)*mSz.x)/aNbPts , (double(anY)*mSz.y)/aNbPts);
            double aCoh = CoherAR(aPIm);
            aMoyCoh += aCoh;
            ElSetMax(aMaxCoh,aCoh);
        }
    }
    aMoyCoh /= ElSquare(1+2*aNbPts);
}


Pt3dr  cPushB_PhysMod::RoughPtIm2GeoC_Init(const Pt2dr & aPIm) const
{
   return  Im2GeoC_Init(aPIm).Mil();
}


Pt3dr  cPushB_PhysMod::Rough_GroundSpeed(double anY) const
{
    double XMil = mSz.x / 2.0;
    return  RoughPtIm2GeoC_Init(Pt2dr(XMil,anY+0.5)) - RoughPtIm2GeoC_Init(Pt2dr(XMil,anY-0.5));
}
/*
*/

Pt2di cPushB_PhysMod::Sz() const
{
   return mSz;
}

Pt2dr   cPushB_PhysMod::GeoC2Im(const Pt3dr & aP)   const  // Invert
{
   return (mModeRefine==eMRP_Invert) ?
          GeoCToIm_Refined(aP)       :
          GeoC2Im_Init(aP)           ;
}

ElSeg3D cPushB_PhysMod::Im2GeoC(const Pt2dr & aP)   const
{
   return (mModeRefine==eMRP_Direct) ?
          Im2GeoC_Refined(aP)        :
          Im2GeoC_Init(aP)           ;
}



Pt2dr    cPushB_PhysMod::GeoCToIm_Refined(const Pt3dr & aPTer)   const 
{
    Pt2dr aPIm = GeoC2Im_Init(aPTer);
    ElSeg3D aSeg0 = Im2GeoC_Init(aPIm);
    Pt3dr aProj = aSeg0.ProjOrtho(aPTer);

    Pt3dr aGradX = Im2GeoC_Init(aPIm+Pt2dr(1,0)).ProjOrtho(aPTer) -aProj;
    Pt3dr aGradY = Im2GeoC_Init(aPIm+Pt2dr(0,1)).ProjOrtho(aPTer) -aProj;
    Pt3dr aGradZ = aSeg0.TgNormee();
    ElMatrix<double> aGrad = MatFromCol(aGradX,aGradY,aGradZ);
    aGrad = gaussj(aGrad);

    int aNbIter =0 ;
    while ((euclid(aProj-aPTer) > TheEpsilonRefine) && (aNbIter<5))
    {
        aNbIter ++;
        Pt3dr aSol = aGrad * (aPTer-aProj);
        aPIm = aPIm + Pt2dr(aSol.x,aSol.y);

        aProj = Im2GeoC_Init(aPIm).ProjOrtho(aPTer);
    }

    return aPIm;
}

Pt3dr cPushB_PhysMod::Im2GeoC_Refined(const Pt2dr & aP0Im,Pt3dr aPTer,const Pt3dr &aU,const Pt3dr & aV) const
{
    Pt2dr  aPIm = GeoC2Im_Init(aPTer);
    Pt2dr  aGradX = GeoC2Im_Init(aPTer+aU) - aPIm;
    Pt2dr  aGradY = GeoC2Im_Init(aPTer+aV) - aPIm;

    ElMatrix<double> aGrad = MatFromCol(aGradX,aGradY);
    aGrad = gaussj(aGrad);

    int aNbIter =0 ;
    while ((euclid(aPIm-aP0Im) > TheEpsilonRefine) && (aNbIter<5))
    {
        aNbIter ++;
        Pt2dr aSol = aGrad * (aP0Im-aPIm);
        aPTer = aPTer + aU*aSol.x + aV*aSol.y;

        aPIm = GeoC2Im_Init(aPTer);
    }
    return aPTer;
}
/*
*/

ElSeg3D  cPushB_PhysMod::Im2GeoC_Refined(const Pt2dr & aPIm)   const 
{
    ElSeg3D aSeg = Im2GeoC_Init(aPIm);
    Pt3dr aW = aSeg.TgNormee();
    Pt3dr aU,aV;
    MakeRONWith1Vect(aW,aU,aV);

    return ElSeg3D(Im2GeoC_Refined(aPIm,aSeg.P0(),aU,aV),Im2GeoC_Refined(aPIm,aSeg.P1(),aU,aV));
}

void cPushB_PhysMod::PostInitLinesPB()
{
    mMoyRay = 0;
    mMoyAlt = 0;
    for (int aK=0 ; aK<= mSzGeoL.y ; aK++)
    {
         double anY = (mSwapXY ? mSz.x : mSz.y)  * (aK/double(mSzGeoL.y)); 
         cPushB_GeomLine * aLPB = new cPushB_GeomLine(this,mSzGeoL.x,anY);
         mLinesPB.push_back(aLPB);
         mMoyRay +=  euclid(aLPB->Center());
         mMoyAlt +=  euclid(aLPB->Center()) - TheRayTerre;
         
    }

    mMoyRay /= mLinesPB.size();
    mMoyAlt /= mLinesPB.size();

    mPeriod = sqrt((4*ElSquare(PI) * pow(mMoyRay,3)) / (TheCsteGravi*TheMasseTerre));

    double aDist = euclid(mLinesPB.front()->CUnRot()-mLinesPB.back()->CUnRot());
    double aTeta = 2 * asin((aDist/2.0) / mMoyRay);
    mDureeAcq = mPeriod * (aTeta/ (2*PI));

    // Tentative de correction de la rotation de la terre pour retrouber une trajectoire plane
    for (int aK=0 ; aK< int(mLinesPB.size()) ; aK++)
    {
        Pt3dr aC = mLinesPB[aK]->Center();
        aC = cSysCoord::WGS84Degre()->FromGeoC(aC);
        aC.x +=  360.0 * (double(aK)/mLinesPB.size() ) * (mDureeAcq/(24*3600.0));
        aC =  cSysCoord::WGS84Degre()->ToGeoC(aC);
        mLinesPB[aK]->CUnRot() = aC;
    }

    //   Stat sur les indicateurs
    mMoyRes = 0;
    mMaxRes = 0;
    mMoyPlan = 0;
    mMaxPlan = 0;
    for (int aK=0 ; aK<int(mLinesPB.size())  ; aK++)
    {
         cPushB_GeomLine * aLPB = mLinesPB[aK];

         mMoyRes += aLPB->MoyResiduCenter();
         ElSetMax(mMaxRes,aLPB->MaxResiduCenter());
         mMoyPlan += aLPB->MoyDistPlan();
         ElSetMax(mMaxPlan,aLPB->MaxDistPlan());
    }


    // Calcul de la calibration 
    mCalib = mLinesPB[0]->Calib();
    int aNbC = (int)mCalib.size();
    for (int aKL=1 ; aKL< int(mLinesPB.size()) ; aKL++)
    {
        const std::vector<double>  aCalK =  mLinesPB[aKL]->Calib();
        ELISE_ASSERT(aNbC==int(aCalK.size()),"Incohe size of Calib in cPushB_PhysMod::PostInitLinesPB");
        for (int aKC=0 ;  aKC<aNbC ; aKC++)
        {
             mCalib[aKC] += aCalK[aKC];
        }
    }
    // double a
    mMoyCalib = 0.0;
    mMaxCalib = 0.0;
    for (int aKC=0 ;  aKC<aNbC ; aKC++)
    {
        mCalib[aKC] /= mLinesPB.size();
        for (int aKL=0 ; aKL< int(mLinesPB.size()) ; aKL++)
        {
            double anEr = ElAbs(mCalib[aKC]-mLinesPB[aKL]->Calib()[aKC]);
            ElSetMax(mMaxCalib,anEr);
            mMoyCalib += anEr;
        }
    }
    mMoyCalib /= aNbC * mLinesPB.size();
}

void cPushB_PhysMod::PostInit()
{
    PostInitLinesPB();
}

bool  cPushB_PhysMod::SwapXY() const
{
   return mSwapXY;
}

double TetaOfAxeRot(const ElMatrix<REAL> & aMat, Pt3dr & aP1);


void cPushB_PhysMod::ShowLinesPB(bool Det)
{
    cElPlan3D aPlanC (Pt3dr(0,0,0),mLinesPB.front()->Center(),mLinesPB.back()->Center());
    ElRotation3D aRE2P = aPlanC.CoordPlan2Euclid().inv();
    cElPlan3D aPlanUr (Pt3dr(0,0,0),mLinesPB.front()->CUnRot(),mLinesPB.back()->CUnRot());
    ElRotation3D aRE2PUr = aPlanUr.CoordPlan2Euclid().inv();


    
    if (Det)
    {
        // Pourquoi les redisu ZPU restent forts, explications :
        //
        //   * force de Coriolis ??? => a priori non car le UnRot remet dans un ref galileen ?
        //   * centre de masse terre != origine du repere GeoC
        //   * influence Lune/Soleil ; assimilabe à un déplacement du centre de masse ?? 
        //   * correlation entre position et attitude + legere approx des RPC 

        

        for (int aK=0 ; aK<int(mLinesPB.size())  ; aK++)
        {
             cPushB_GeomLine * aLPB = mLinesPB[aK];
             Pt3dr aPP = aRE2P.ImAff(aLPB->Center());
             Pt3dr aPPUr = aRE2PUr.ImAff(aLPB->CUnRot());

             std::cout << "Residu= " << aLPB->MoyResiduCenter() 
                       << " " << aLPB->MaxResiduCenter() 
                       << " DPlan " << aLPB->MoyDistPlan() 
                       << " " << aLPB->MaxDistPlan() 
                       << " ZP=" << aPP.z 
                       << " ZPU=" << aPPUr.z ;
             if (aK>0)
             {
                std::cout << " DRay=" << euclid(aLPB->Center()) -  euclid(mLinesPB[aK-1]->Center()) ;
                ElMatrix<double>  aMat = mLinesPB[aK-1]->MatC1ToC2(*(mLinesPB[aK]));
                Pt3dr anAxe =  AxeRot (aMat);
				if (anAxe.z<0)
					anAxe = -anAxe;
                double aTeta = TetaOfAxeRot(aMat,anAxe);
                std::cout << " Axe " << anAxe   << " AcAx " << euclid(anAxe-aMat*anAxe) << " Teta " << (aTeta ) * 1000 ;

             }
             std::cout  << " " << aK << "\n";
        }

    }

    mMoyRes /= mLinesPB.size();
    mMoyPlan /= mLinesPB.size();
    double aCMoy,aCMax;
    CoherARGlob(20,aCMoy,aCMax);


    std::cout << "\n";
    std::cout << "========= xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx =================== \n";
    std::cout << " RESIDIU Centre ; MOY=" << mMoyRes << " MAX=" << mMaxRes << "\n";
    std::cout << " RESIDIU Planar ; MOY=" << mMoyPlan*mMoyAlt << " MAX=" << mMaxPlan*mMoyAlt << " meter\n";
    std::cout << " RESIDIU Calib ; MOY=" << mMoyCalib*mMoyAlt << " MAX=" << mMaxCalib*mMoyAlt << " meter\n";
    std::cout << " ORBIT, Ray-MOY " << mMoyRay/1000 << " km;  Alt-MOY " << mMoyAlt/1000 << " km;" 
              << " PERIOD=" << mPeriod /60.0 << " Min,  Duree " << mDureeAcq << " Sec\n";
    std::cout << " SENSOR , SZ " << mSz  << " (Pixel) Coher, Moy=" <<  aCMoy << " Max=" << aCMax << "\n";
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
