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


#include "TiepTri.h"


/************************************************/
/*                                              */
/*          cImSecTieTri                        */
/*                                              */
/************************************************/

cImSecTieTri::cImSecTieTri(cAppliTieTri & anAppli ,const std::string& aNameIm,int aNum) :
   cImTieTri   (anAppli,aNameIm,aNum),
   mImReech    (1,1),
   mTImReech   (mImReech),
   mImLabelPC  (1,1),
   mTImLabelPC (mImLabelPC),
   mMasqReech  (1,1),
   mTMasqReech (mMasqReech),
   mAffMas2Sec (ElAffin2D::Id()),
   mAffSec2Mas (ElAffin2D::Id()),
   mHomMas2Sec (cElHomographie::Id()),
   mHomSec2Mas (cElHomographie::Id()),
   mMaster     (anAppli.Master())
{
}


bool cImSecTieTri::LoadTri(const cXml_Triangle3DForTieP &  aTri)
{
   if (! mAppli.NumImageIsSelect(mNum)) 
      return false;

   if (! cImTieTri::LoadTri(aTri)) 
   {
        return false;
   }

   // Reechantillonage des images

   // Estimer par affine
   mAffMas2Sec = ElAffin2D::FromTri2Tri
                 (
                      mMaster->mP1Loc,mMaster->mP2Loc,mMaster->mP3Loc,
                      mP1Loc,mP2Loc,mP3Loc
                 );

   mAffSec2Mas = mAffMas2Sec.inv();
   if (mAppli.mUseHomo)
   {
   // Estimer par Homographie
   if (PtTri3DHomoGrp().size() >= 4)
   {
           ElPackHomologue aPackHomoGr;
           for (uint aK = 0; aK<4; aK++)
           {
               ElCplePtsHomologues aCpl(mMaster->mCamS->Ter2Capteur(PtTri3DHomoGrp()[aK]) - Pt2dr(mMaster->Decal()) , mCamS->Ter2Capteur(PtTri3DHomoGrp()[aK]) - Pt2dr(mDecal));
               aPackHomoGr.Cple_Add(aCpl);
           }

           if (aPackHomoGr.size() >= 4)
           {
               mHomMas2Sec = cElHomographie(aPackHomoGr, true);
               mHomSec2Mas = mHomMas2Sec.Inverse();
           }
           else
           {
               cout<<"Homo Fail NB = "<<aPackHomoGr.size()<<endl;
               return false;
           }
       }
   }
    // ---------

   mSzReech = mMaster->mSzIm;

   mImReech.Resize(mSzReech);
   mTImReech =  TIm2D<tElTiepTri,tElTiepTri>(mImReech);

   mCutACD.ResetIm(mTImReech);

   mImLabelPC.Resize(mSzReech);
   mImLabelPC.raz();
   mTImLabelPC = TIm2D<U_INT1,INT>(mImLabelPC);

   mMasqReech = Im2D_Bits<1>(mSzReech.x,mSzReech.y,0);
   mTMasqReech =  TIm2DBits<1> (mMasqReech);



   Pt2di aPSec;
   Pt2dr aSom(0.0,0.0);
   Pt2dr aMaxErr(0.0, 0.0);
   for (aPSec.x=0 ; aPSec.x<mSzReech.x ; aPSec.x++)
   {
       for (aPSec.y=0 ; aPSec.y<mSzReech.y ; aPSec.y++)
       {
           Pt2dr aPMast = mAffMas2Sec(Pt2dr(aPSec));
           double aVal;
           if (mAppli.mUseHomo)
           {
                Pt2dr aPMastHom = mHomMas2Sec(Pt2dr(aPSec));
                Pt2dr aErr(abs(aPMast.x - aPMastHom.x), abs(aPMast.y - aPMastHom.y));
                if (aErr.x > aMaxErr.x)
                {
                    aMaxErr.x = aErr.x;
                }
                if (aErr.y > aMaxErr.y)
                {
                    aMaxErr.y = aErr.y;
                }
                aSom += aErr;
                aVal = mTImInit.getr(aPMastHom,-1);
           }
           else
           {
                aVal = mTImInit.getr(aPMast,-1);
           }
           mTImReech.oset(aPSec,aVal);
           mTMasqReech.oset(aPSec,mTMasqIm.get(round_ni(aPMast),0));
       }
   }

   if (mAppli.mUseHomo)
   {
       double aSurf = (mP1Glob-mP2Glob) ^ (mP1Glob-mP3Glob);
       Pt2dr aDiffMoy = aSom/(mSzReech.x*mSzReech.y);
       //cout<<"Diff Aff-Hom "<<aDiffMoy<<" - SURF "<<aSurf<<endl;

       if ( (mAppli.mSurfDiffAffHomo.x < -aSurf) && (-aSurf < mAppli.mSurfDiffAffHomo.y) )
       {
            mAppli.MoyDifAffHomo() += aDiffMoy;
            mAppli.CountDiff()++;
            if (aMaxErr.x > mAppli.MaxDifAffHomo().x)
            {
                mAppli.MaxDifAffHomo().x = aMaxErr.x;
            }
            if (aMaxErr.y > mAppli.MaxDifAffHomo().y)
            {
                mAppli.MaxDifAffHomo().y = aMaxErr.y;
            }
            mAppli.HistoErrAffHomoX()[round(aMaxErr.x/0.01)]++;
            mAppli.HistoErrAffHomoY()[round(aMaxErr.y/0.01)]++;
            if(mAppli.mErrLog.is_open())
                {mAppli.mErrLog<<aMaxErr.x<<endl;}
       }
   }

   if (mW)
   {
/*

      ELISE_COPY
      (
          mImReech.all_pts(),
          Max(0,Min(255,Virgule(mImReech.in(),mMaster->mImInit.in(0),mMaster->mImInit.in(0)))),
          mW->orgb()
      );
      mW->draw_circle_loc(aPtHom1LocOnMas,2.0, mW->pdisc()(P8COL::blue));
      mW->draw_circle_loc(aPtHom2LocOnMas,2.0, mW->pdisc()(P8COL::blue));
      mW->draw_circle_loc(aPtHom3LocOnMas,2.0, mW->pdisc()(P8COL::blue));
      mW->draw_circle_loc(aPtHom4LocOnMas,2.0, mW->pdisc()(P8COL::blue));

      //mW->clear();
*/

//  ===== Affichier org img 2nd =======
       /*
      ELISE_COPY
      (
          mImInit.all_pts(),
          Max(0,Min(255,255-mImInit.in())),
          mW->ogray()
      );
      ELISE_COPY(select(mImInit.all_pts(),mMasqTri.in()),Min(255,Max(0,mImInit.in())),mW->ogray());

      for (uint aK=0; aK<PtTri3DHomoGrp().size(); aK++)
      {
          Pt2dr aPtHom1LocOn2nd = mCamGen->Ter2Capteur(PtTri3DHomoGrp()[aK]) - Pt2dr(mDecal);
          mW->draw_circle_loc(aPtHom1LocOn2nd,1.0, mW->pdisc()(P8COL::red));
      }
      */

 /*  ===== Affichier rech img 2nd =======*/
       /*
      ELISE_COPY
      (
          mImReech.all_pts(),
          Max(0,Min(255,255-mImReech.in())),
          mW->ogray()
      );
      ELISE_COPY(select(mImReech.all_pts(),mMaster->mMasqTri.in()),Min(255,Max(0,mImReech.in())),mW->ogray());

      // mW->clik_in();
   }
   */
    }
   MakeInterestPoint(0,&mTImLabelPC,mMaster->mTMasqTri,mTImReech);

   //MakeInterestPointFAST(0,&mTImLabelPC,mMaster->mTMasqTri,mTImReech);

   return true;




}

/*
 glob  -0.0861334 [-9,7] [-2,2]
--loc-- -0.259069 [-8,5] [-1,0]
  glob  0.346995 [4,-8] [2,-1]
--loc-- 0.267576 [3,-8] [1,-1]
  glob  0.840767 [4,7] [2,-2]
--loc-- 0.781191 [3,8] [1,-1]
==================== [333,157] 2

*/

void  cImSecTieTri::DecomposeVecHom(const Pt2dr & aPSH1Local,const Pt2dr & aPSH2Local,Pt2dr & aDir,Pt2dr & aDecompos)
{
       Pt2dr aPSH1 = aPSH1Local  + Pt2dr(mDecal);
       Pt2dr aPSH2 = aPSH2Local  + Pt2dr(mDecal);

       bool Ok;
       Pt3dr aPTer =  mAppli.CurPlan().Inter(mCamGen->Capteur2RayTer(aPSH1) , &Ok);

       double aProf = mMaster->mCamGen->ProfondeurDeChamps(aPTer);
       Pt2dr aPM1 = mMaster->mCamGen->Ter2Capteur(aPTer);
       Pt3dr aPTerMod =  mMaster->mCamGen->ImEtProf2Terrain(aPM1,aProf*(1+1e-3));
       Pt2dr aPSH1Mod =  mCamGen->Ter2Capteur(aPTerMod);

       aDir = vunit(aPSH1Mod -aPSH1);
       aDecompos = (aPSH2-aPSH1) / aDir;

       

       std::cout << "GGGGGg  " << aDecompos << "\n";
       // Pt2dr aDirProf =  vunit(aPSH1Mod- aPSH1);
}





cResulRechCorrel cImSecTieTri::RechHomPtsInteretEntier(bool Interact,const cIntTieTriInterest & aPI)
{
    double aD= mAppli.DistRechHom();
    Pt2di aP0 = aPI.mPt;
    eTypeTieTri aLab = aPI.mType;

    const std::vector<Pt2di> &   aVH = mAppli.VoisHom();

    /*
         Pour tous les voisins au sens de VH, si ils ont la meme etiquette que aPI :

             1-  on fait une recherche d'optimisation locale, avec un pixel sur deux et en correlation 
                 entiere.
             2- si on est superieur au seuil TT_SEUIL_CORREL_1PIXSUR2, on reoptimise avec tous les pixels;

          Comme il y a en general plusieurs voisins ayant le meme label, on ne selectionn dans aCRCMax celui qui
        donne le meilleur resultat.
             
    */

    cResulRechCorrel aCRCMax;
    double aCorMax = -2; // Uniquement pour affichage
    for (int aKH=0 ; aKH<int(aVH.size()) ; aKH++)
    {
        Pt2di aPV = aP0+aVH[aKH];
        if ((mTImLabelPC.get(aPV,-1)==aLab) && InMasqReech(aPV))
        {
           if (Interact)
           {
               mW->draw_circle_loc(Pt2dr(aPV),2.0,ColOfType(aLab));
           }
               // cResulRechCorrel aCRC = TT_RechMaxCorrelBasique(mMaster->mTImInit,aP0,mTImReech,aPV,3,2,aSzRech);

           int aSzRech = TT_DemiFenetreCorrel;
           cResulRechCorrel aCRCLoc = TT_RechMaxCorrelLocale(mMaster->mTImInit,aP0,mTImReech,aPV,TT_DemiFenetreCorrel/2,2,aSzRech); // Correlation 1SUR2
           

           aCorMax = ElMax(aCRCLoc.mCorrel,aCorMax);



           if (
                      (aCRCLoc.mCorrel > mAppli.mTT_SEUIL_CORREL_1PIXSUR2)
                   && InMasqReech(aCRCLoc.mPt) 
                   && (euclid(Pt2di(aCRCLoc.mPt) - aPV) < mAppli.mTT_SEUIl_DIST_Extrema_Entier)
              )
           {
               //  Toujours correlation entiere, mais tt les pixels
               aCRCLoc = TT_RechMaxCorrelLocale(mMaster->mTImInit,aP0,mTImReech,Pt2di(aCRCLoc.mPt),TT_DemiFenetreCorrel,1,aSzRech);   
                   
               if (euclid(Pt2di(aCRCLoc.mPt) - aPV) < mAppli.mTT_SEUIl_DIST_Extrema_Entier)
                  aCRCMax.Merge(aCRCLoc);
           }
        }
    }

    if (Interact)
    {
        mW->draw_circle_loc(Pt2dr(aP0),1.0,mW->pdisc()(P8COL::green));    //point interet master image pendant matching
        mW->draw_circle_loc(Pt2dr(aP0),aD,mW->pdisc()(P8COL::yellow));    //
    }

    if (! aCRCMax.IsInit())
    {
       if (Interact)
       {
            std::cout  << "- NO POINT for Correl Int , Correl=" <<  aCorMax << "\n";
       }
       return cResulRechCorrel(Pt2dr(aCRCMax.mPt),TT_DefCorrel);
    }
   
    return aCRCMax;
}

cResulRechCorrel cImSecTieTri::RechHomPtsInteretBilin(bool Interact,const cResulMultiImRechCorrel &aRMIC,int aKIm)
{
    Pt2dr aP0 = Pt2dr(aRMIC.PtMast());
    cResulRechCorrel aCRC0 = aRMIC.VRRC()[aKIm];

    double aStep = 0.01;
    if (! aCRC0.IsInit())
       return aCRC0;
    
    int aSzWE = mAppli.mSzWEnd;  // command param SzWEnd, default = 6
    // Correlation sub-pixel, interpol bilin basique (step=1, step RCorell=0.1)
    cResulRechCorrel aRes =TT_RechMaxCorrelMultiScaleBilin (mMaster->mTImInit,Pt2dr(aP0),mTImReech,Pt2dr(aCRC0.mPt),aSzWE,aStep); 

    //ER variable that is unused; commented-out to avoid warning
    //double aRecCarre=0;
    if (0) // ( mAppli.mNumInterpolDense < 0)
    {
       Pt2dr aP0This = Pt2dr(Pt2di(aRes.mPt));
       cResulRechCorrel aResRecip = TT_RechMaxCorrelMultiScaleBilin(mTImReech,aP0This,mMaster->mTImInit,Pt2dr(aP0),aSzWE,aStep);

       Pt2dr aDec1 = aRes.mPt - Pt2dr(aP0);
       Pt2dr aDec2 = Pt2dr(aP0This) - aResRecip.mPt ;

       // std::cout << "RECIPROCITE " << aDec1 - aDec2 << "\n";
       // aRecCarre= euclid(aDec1-aDec2);
       // aRes.mCorrel -= euclid(aDec1-aDec2);
       // aRes.mCorrel = aRes.mCorrel- 10*euclid(aDec1-aDec2);
       aRes.mPt = Pt2dr(aP0) + (aDec1+aDec2) / 2.0; // D1 0.045417  D2 0.037713  D2-90  0.144614
       // aRes.mPt = Pt2dr(aP0) + aDec1; // D1 :  0.041964  D2 : 0.039659  D2-90 : 0.150437
       // aRes.mPt = Pt2dr(aP0) + aDec2; //   D1 : 0.06037  D2 : 0.0498  D2-90 : 0.1575
       aRes.mPt = Pt2dr(aP0) + (aDec1*3.0+aDec2) / 4.0; // D1 : 0.041397  D2 : 0.036412   D2-90 0.145165

    }

    if (0)      // using LSQ matching
    {
        ElAffin2D anAffOpt =  ElAffin2D::trans(aRes.mPt - mAffMas2Sec(Pt2dr(aP0)) ) * mAffMas2Sec;
        cLSQAffineMatch aMatchM2S(Pt2dr(aP0),mMaster->mImInit,mImInit,anAffOpt);
        for (int aK=0 ; aK<8 ; aK++)
        {
	    double aStepLSQPxlEntier = 1; 
	    bool aAffineGeom = false;
            bool aAffineRadiom = false; 
            /*bool aOk = */ aMatchM2S.OneIter(mAppli.Interpol(), aSzWE, aStepLSQPxlEntier, aAffineGeom, aAffineRadiom);
        }
        anAffOpt = aMatchM2S.Af1To2();
        Pt2dr aNewP2 =  anAffOpt(Pt2dr(aP0));

/*
        cLSQAffineMatch aMatchM2SRec(aNewP2,mImInit,mMaster->mImInit,anAffOpt.inv());
        aOk = aMatchM2SRec.OneIter(6,1,false);
        ElAffin2D anAffOptRec = aMatchM2SRec.Af1To2();
        Pt2dr aP0Bis = anAffOptRec(aNewP2);


        Pt2dr aDec1 = aNewP2 - Pt2dr(aP0);
        Pt2dr aDec2 = aNewP2 - aP0Bis;

        aRes.mCorrel =  1- euclid(Pt2dr(aP0)-aP0Bis);
        aRes.mPt = Pt2dr(aP0) + (aDec1+aDec2) / 2.0; // 
*/
        aRes.mPt = aNewP2;
    }

    return aRes;
}

/*
cResulRechCorrel cImSecTieTri::RechHomPtsInteretEntierAndRefine(bool Interact,const cIntTieTriInterest & aPI)
{
    cResulRechCorrel aRes = RechHomPtsInteretEntier(Interact,aPI);

    if (aRes.IsInit())
    {
        aRes = RechHomPtsInteretBilin(Interact,Pt2dr(aPI.mPt),aRes);
    }

    return aRes;
}
*/

// On passe des coordonnees master au coordonnees secondaire

cResulRechCorrel cImSecTieTri::RechHomPtsDense(bool Interact,const cResulMultiImRechCorrel &aRMIC,int aKIm)
{

    Pt2dr aP0 = Pt2dr(aRMIC.PtMast());
    cResulRechCorrel aPIn = aRMIC.VRRC()[aKIm];

    if ( mAppli.mNumInterpolDense < 0) // command's param : IntDM : -1=NONE, 0=BiL, 1=BiC, 2=SinC; default = -1
    {
       if (Interact) std::cout << "AAAAAAaaaaaaaaaaaaaaaa\n";
       cResulRechCorrel aRes2  = aPIn;
       aRes2.mPt = mAffMas2Sec(aRes2.mPt);
       return aRes2;
    }
    
    // mDoRaffImInit = command's param: DRInit:  Do refinement on initial images (true), instead of resampled (false), default = false
    ElAffin2D  aAffPred  = mAppli.mDoRaffImInit ? mAffMas2Sec : ElAffin2D::Id();
    tTImTiepTri aImSec =   mAppli.mDoRaffImInit ? mTImInit    : mTImReech ;

    int aSzWE = mAppli.mSzWEnd;
    double aPrecInit = (mAppli.mNivLSQM >=0) ? 1/4.0 : 1/8.0;      // LSQC : "Test LSQ,-1 None (Def), Flag 1=>Affine Geom, Flag 2=>Affin Radiom"
    double aPrecCible = (Interact) ?  1e-3 : ((mAppli.mNivLSQM >=0) ? 1/16.0 : 1/128.0);
    int aNbByPix = (mAppli.mNivLSQM >=0) ? 1 : mAppli.mNbByPix;    // NbByPix : "Number of point inside one pixel - default = 1"
    /* Set param LSQ :
         * aAffPred : matrix transformation affine correspondant with case affine on geometry original or reech
         * aImSec : Img 2nd correspondant with case affine on geometry original or reech
         * aSzWE = 6 (default) - size windows
         * 1/4.0 -> 1/16.0 if LSQC = Flag 1 (Aff Geom) or 2 (Aff Radio)
         * 1/8.0 -> 1/128.0 if LSQC = Flag -1
    */
    cResulRechCorrel aRes2 =  TT_MaxLocCorrelDS1R
                                      (
                                           mAppli.Interpol(),
                                           &aAffPred,
                                           mMaster->mTImInit,
                                           Pt2dr(aP0),
                                           aImSec,
                                           aAffPred(aPIn.mPt),
                                           aSzWE,  // SzW
                                           aNbByPix,
                                           aPrecInit,
                                           aPrecCible
                                       );
    if (!  mAppli.mDoRaffImInit)
    {
       // If we do refine on Im rech geometry (DRInit = false) => re-calcul matched point on 2nd image's global coordinate
       aRes2.mPt = mAffMas2Sec(aRes2.mPt);
    }

    if (Interact)
    {
       std::cout << "AFFINE " << aPIn.mCorrel << " => " << aRes2.mCorrel << " ; " << aPIn.mPt << " " << mAffSec2Mas(aRes2.mPt) << "\n"; 

       std::cout << "HHHH " << USE_SCOR_CORREL << " " << (mAppli.mNumInterpolDense==0) << "\n";
    }

    // After dense homologue search with TT_MaxLocCorrelDS1R, using result to adjust affine transformation
    ElAffin2D anAffOpt =  mAffMas2Sec.CorrectWithMatch(Pt2dr(aP0),aRes2.mPt);

    if (mAppli.mNivLSQM >=0)
    {
        // ElAffin2D anAffOpt =  ElAffin2D::trans(aRes2.mPt - mAffMas2Sec(Pt2dr(aP0)) ) * mAffMas2Sec;
        Pt2dr aP0Init = Pt2dr(aP0);
        if (mAppli.mRandomize)
        {
           Pt2dr aNoise = Pt2dr(NRrandC(),NRrandC()) * 0.25 * mAppli.mRandomize;
           std::cout << "SIMUL PERTURB = " << aNoise << "\n";
           aP0Init = aP0Init +  aNoise;                                            // Add noise of translation to pt master. Adjust by param's command Randomize
           anAffOpt =  mAffMas2Sec.CorrectWithMatch(aP0Init,aRes2.mPt);            // Re-optimized affine transformation
        }
        cLSQAffineMatch aMatchM2S(Pt2dr(aP0),mMaster->mImInit,mImInit,anAffOpt);   // (pt Master, ImMaster, Im2nd, Affine Master To 2nd)
        bool aOk= true;
        bool AffGeom   = ((mAppli.mNivLSQM  & 1) !=0);
        bool AffRadiom = ((mAppli.mNivLSQM  & 2) !=0);
        
        bool GoOn = true;
        for (int aK=0 ; GoOn ; aK++)
        {
            Pt2dr aLastSol =  aMatchM2S.Af1To2()(Pt2dr(aP0));
            aOk = aMatchM2S.OneIter
                  (
                      mAppli.Interpol(),
                      aSzWE,
                      1.0/mAppli.mNbByPix,
                      AffGeom,
                      AffRadiom
                  );                                        // Update affine transformation
            Pt2dr aCurSol = aMatchM2S.Af1To2()(Pt2dr(aP0)); // Calcul solution current with updated affine transformation
            double aDVar = euclid(aCurSol-aLastSol);
            if (aOk  && Interact)
            {
                if (aK==0)
                    std::cout << "#############################################\n";
                std::cout << "DVar=" << aDVar  << " D2Lim=" << euclid(aRes2.mPt-aLastSol) << "\n";
            }
            aLastSol = aCurSol;                             // Update solution
            if (aK>=7)                                      // Stop LSQ if over 7 iteration
               GoOn = false;
            if (aDVar<1e-2)                                 // Stop LSQ if solution is converge
               GoOn = false;
        }
        anAffOpt = aMatchM2S.Af1To2();                      // Update final affine transformation solution by LSQ
        aRes2.mPt = aMatchM2S.Af1To2()(Pt2dr(aP0));         // Update final 2nd point matched solution by LSQ

/*
        if (1)
        {
             ElAffin2D anAff1To2 = aMatchM2S.Af1To2();
             ElAffin2D anAffRec = anAff1To2.inv();
             Pt2dr aPC2 = aMatchM2S.Af1To2()(Pt2dr(aP0));
             cLSQAffineMatch aMatchM2SRec(aPC2,mImInit,mMaster->mImInit,anAffRec);
             Box2dr aBox1(Pt2dr(0,0),Pt2dr(1,1));
             Box2dr aImBox1 = aBox1.BoxImage(anAff1To2);
                 
        }
*/
    
    }

    if (0) //  (!  mAppli.mDoRaffImInit)
    {
       // Pt2dr aP0This = Pt2dr(Pt2di(aRes2.mPt));
       Pt2dr aP0This = Pt2dr(aRes2.mPt);
       ElAffin2D  aAffPredInv = aAffPred.inv().CorrectWithMatch(aP0This,Pt2dr(aP0));
       cResulRechCorrel aResRecip =  TT_MaxLocCorrelDS1R
                                      (
                                           mAppli.Interpol(),
                                           &aAffPredInv,
                                           aImSec,
                                           Pt2dr(aP0This),
                                           mMaster->mTImInit,
                                           aAffPredInv(aP0This),
                                           aSzWE* mAppli.mNbByPix,  // SzW
                                           1.0/ mAppli.mNbByPix,
                                           0.25,   // Step0
                                           1.0/ 128.0
                                       );


       Pt2dr aDec1 = aRes2.mPt - Pt2dr(aP0);
       Pt2dr aDec2 = Pt2dr(aP0This) - aResRecip.mPt ;
       aRes2.mPt = Pt2dr(aP0) + (aDec1*3.0+aDec2) / 4.0; // D1 : 0.041397  D2 : 0.036412   D2-90 0.145165
       aRes2.mPt = Pt2dr(aP0) + aDec1; // D1 : 0.041397  D2 : 0.036412   D2-90 0.145165

    }



    return aRes2;
}


                                       

/*

    Pt2dr aDir,aNewDec;
    DecomposeVecHom(mAffMas2Sec(Pt2dr(aP0)),mAffMas2Sec(aRes.mPt),aDir,aNewDec);

    if (aNivInter >=1)
    {
        Pt2dr aDepl = aRes.mPt - Pt2dr(aP0);

        // mW->draw_seg(Pt2dr(aP0),Pt2dr(aP0) + aDepl * 3  ,mW->pdisc()(P8COL::red));

        mW->draw_seg(Pt2dr(aP0),Pt2dr(aP0) + Pt2dr(0,aNewDec.y*5)  ,mW->pdisc()(P8COL::red));
        mW->draw_seg(Pt2dr(aP0),Pt2dr(aP0) + Pt2dr(aNewDec.x*5,0)  ,mW->pdisc()(P8COL::green));

        mW->draw_circle_loc(Pt2dr(aP0),0.5,mW->pdisc()(P8COL::yellow));
    }
*/



    // std::cout << "MulScale  = " << aRes.mPt -Pt2dr(aP0)  << " " << aRes.mCorrel << "\n\n";
        // std::cout << "==================== " << aP0 << " "  << (int) aLab << "\n";

bool  cImSecTieTri::IsMaster() const { return false; }

tTImTiepTri & cImSecTieTri::ImRedr() {return mTImReech;}



ElPackHomologue & cImSecTieTri::PackH()  
{
   return mPackH;
}

bool cImSecTieTri::InMasqReech(const Pt2dr & aP) const
{
   return mTMasqReech.get(round_ni(aP),0);
}
bool cImSecTieTri::InMasqReech(const Pt2di & aP) const
{
   return mTMasqReech.get(aP,0);
}

Pt2dr cImSecTieTri::Mas2Sec(const Pt2dr & aP) const
{
   return mAffMas2Sec(aP);
}

Pt2dr cImSecTieTri::Mas2Sec_Hom(const Pt2dr & aP) const
{
   return mHomMas2Sec(aP);
}



// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 
// Res[50.000000]=0.032987


// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 UseABCorrel=1
// Res[50.000000]=0.034954


// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 LSQC=0
// Res[50.000000]=0.035157


// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 LSQC=1
// Res[50.000000]=0.032076


// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 LSQC=2
// Res[50.000000]=0.034815

// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 LSQC=3
// Res[50.000000]=0.032437


// mm3d TestLib TiepTri XML_TiepTri/BIN_010-0117_14576019050_image_024_001_01313.thm.tif.xml Ori-TOrg/  NumSelIm=[15] IntDM=2 DRInit=1 LSQC=1 NbByPix=3

// Res[50.000000]=   0.031831 


// mm3d VisuRedHom BIN_010-0117_14576019050_image_024_001_01313.thm.tif  BIN_010-0116_14576019035_image_024_001_01312.thm.tif  Ori-TOrg/  SH=_TiepTri

// Ss Rec => 0.035157
// Av Rec => 0.034966



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
