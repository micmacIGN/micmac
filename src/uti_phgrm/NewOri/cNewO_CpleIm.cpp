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

#include "NewOri.h"

static double aSzW = 1200;

extern ElRotation3D RansacMatriceEssentielle
             (
                    bool                    Quick,
                    const ElPackHomologue & aPackFull,
                    const ElPackHomologue & aPack500,
                    const ElPackHomologue & aPack150,
                    const ElPackHomologue & aPack30,
                    double aFoc
              );


/***********************************************************************/
/*                                                                     */
/*           END LINEAR                                                */
/*                                                                     */
/***********************************************************************/

void InitVPairComp(std::vector<cNOCompPair> & aV,const ElPackHomologue & aPackH)
{
    aV.clear();
    for (ElPackHomologue::const_iterator itP=aPackH.begin() ; itP!=aPackH.end() ; itP++)
    {
       aV.push_back(cNOCompPair(itP->P1(),itP->P2(),itP->Pds()));
    }
}

//   Rot C2 =>C1; donc Rot( P(0,0,0)) donne le vecteur de Base dans C1
//   aRot  : M2C pour cam2
//   U1
//
//  Formule exacte et programmation simple et claire pour bench
//

#define FONC_EXACT_COST ProjCostMEP

double cNewO_OrInit2Im::ExactCost(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const
{
   return FONC_EXACT_COST(aRot,aP1,aP2,aTetaMax);
}
double cNewO_OrInit2Im::ExactCost(const ElRotation3D & aRot,double aTetaMax) const
{
    return FONC_EXACT_COST(mPackPStd,aRot,aTetaMax);
}
double  cNewO_OrInit2Im::PixExactCost(const ElRotation3D & aRot,double aTetaMax) const
{
   return ExactCost(aRot,aTetaMax) * FocMoy();
}





Pt2dr cNewO_OrInit2Im::ToW(const Pt2dr & aP) const
{
     return (aP-mP0W) *mScaleW;
}


void cNewO_OrInit2Im::ShowPack(const ElPackHomologue & aPack,int aCoul,double aRay)
{
    if (! mW) return;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
        mW->draw_circle_abs(ToW(itP->P1()),aRay,mW->pdisc()(aCoul));
}

void  cNewO_OrInit2Im::ClikIn()
{
   if (mW) mW->clik_in();
}

double cNewO_OrInit2Im::FocMoy() const
{
    double aF = 1/mI1->CS()->Focale() + 1/mI2->CS()->Focale();
    return 2 / aF;
}

ElRotation3D * TestOriPlanePatch
     (
         double aFoc,
         const ElPackHomologue & aPack,
         const ElPackHomologue & aPack150,
         const ElPackHomologue & aPack30,
         Video_Win * aW,
         Pt2dr       aP0W,
         double      mScaleW
     );

ElRotation3D TestcRanscMinimMatEss
             (
                  bool  aQuick,
                  const ElPackHomologue & aPack,
                  const ElPackHomologue & aPackRed,
                  const ElPackHomologue & aPack150,
                  const ElPackHomologue & aPack30,
                  double aFoc
             ) ;


double cNewO_OrInit2Im::RecouvrtHom(const cElHomographie & aHom)
{
    int aNb = NbRecHom;

    CamStenope * aCS1 = mI1->CS();
    CamStenope * aCS2 = mI2->CS();
    Pt2dr aSz1 = Pt2dr(aCS1->Sz());

    int aNbIn=0;
    int aNbOut=0;

    for (int aKx=0 ; aKx<=aNb ; aKx++)
    {
        for (int aKy=0 ; aKy<=aNb ; aKy++)
        {
            Pt2dr aP = aSz1.mcbyc(Pt2dr(aKx,aKy)/aNb);

            aP = ProjStenope(aCS1->F2toDirRayonL3(aP));
            aP = aHom.Direct(aP);
            bool Ok = aCS2->PIsVisibleInImage(PZ1(aP));
            if (Ok)
               aNbIn++;
            else
               aNbOut++;
        }
    }

    return aNbIn / double(aNbIn+aNbOut);
}


cNewO_OrInit2Im::cNewO_OrInit2Im
(
      bool          aGenereOri,  // False en frontal a Ratafia etc ....
      bool          aQuick,
      cNewO_OneIm * aI1,
      cNewO_OneIm * aI2,
      tMergeLPackH *      aMergeTieP,
      ElRotation3D *      aTestedSol,
      ElRotation3D *      aInOri,
      bool                aShow,
      bool                aHPP,
      bool                aSelAllIm,
      const cCommonMartiniAppli & aCMA
)  :
   mPdsSingle   (aCMA.mAcceptUnSym ? 0.1 : 0),
   mQuick       (aQuick),
   mI1          (aI1),
   mI2          (aI2),
   mMergePH     (aMergeTieP),
   mTestC2toC1  (aTestedSol),
   // Plus utilise
   mPackPDist   (ToStdPack(mMergePH,true,mPdsSingle)),
   //  Representation sous la classe habituelle ToStdPack
   mPackPStd    (ToStdPack(mMergePH,false,mPdsSingle)),
   mPInfI1      (1e5,1e5),
   mPSupI1      (-1e5,-1e5),
   //  On va cree un sous echantillonage des points, en cherchant a conserver
   //  la distribution initiale
   // mPackStdRed => contiendra au max 500 points choisis "intelligemmennt" parmi
   // 1500 pris completement au hasard
   mPackStdRed  (PackReduit(mPackPStd,(aGenereOri?1500 : 150) ,(aGenereOri?500 : 50) )),
   mPack150     (PackReduit(mPackStdRed, (aGenereOri ? 100 : 10) )),
   mPack30      (PackReduit(mPack150, (aGenereOri ? 30 : 3))),
   mSysLin5     (5),
   mSysLin2     (2),
   mSysLin3     (3),
   mLinDetIBI   (cInterfBundle2Image::LinearDet(mPackStdRed,FocMoy())),
   mBundleIBI   (cInterfBundle2Image::Bundle(mPackStdRed,FocMoy(),true)),
   mBundleIBI150   (cInterfBundle2Image::Bundle(mPack150,FocMoy(),true)),
   mRedPvIBI    (cInterfBundle2Image::LineariseAngle(mPackStdRed,FocMoy(),true)),
   mFullPvIBI   (cInterfBundle2Image::LineariseAngle(mPackPStd,FocMoy(),true)),
   mShow        (aShow),
   mBestSol     (ElRotation3D::Id),
   mCostBestSol (1e9),
   mBestSolIsInit (false),
   mSegAmbig      (Pt3dr(0,0,0),Pt3dr(1,1,1)),
   mW             (0),
   mSelAllIm      (aSelAllIm)
{
    bool DoOriByHom = (aCMA.ModeNO() == eModeNO_OnlyHomogr);
    bool DoOri3D    = (aCMA.ModeNO() != eModeNO_OnlyHomogr);


    Pt2dr aInf1(1e60,1e60);
    Pt2dr aInf2(1e60,1e60);
    Pt2dr aSup1(-1e60,-1e60);
    Pt2dr aSup2(-1e60,-1e60);
    // Sauvegarde des point hom flottants 
    {
 
        const tLMCplP & aLM = mMergePH->ListMerged();
        tVP2f aVP1;
        tVP2f aVP2;
        tVUI1 aVNb;
 
         for ( tLMCplP::const_iterator itC=aLM.begin() ; itC!=aLM.end() ; itC++)
         {
             const Pt2dr & aP1 = (*itC)->GetVal(0);
             aInf1 = Inf(aInf1,aP1);
             aSup1 = Sup(aSup1,aP1);
             const Pt2dr & aP2 = (*itC)->GetVal(1);
             aInf2 = Inf(aInf2,aP2);
             aSup2 = Sup(aSup2,aP2);

             aVP1.push_back(Pt2df(aP1.x,aP1.y));
             aVP2.push_back(Pt2df(aP2.x,aP2.y));
             aVNb.push_back((*itC)->NbArc());
         }
         mI1->NM().Dir3PDeuxImage(mI1,mI2,true);
         std::string aNameH = mI1->NM().NameHomFloat(mI1,mI2);
         mI1->NM().WriteCouple(aNameH,aVP1,aVP2,aVNb);

         if (! aGenereOri) 
         {
            return;
         }
    }

   // Prepare une partie de l'export xml
   mXml.Im1()   = mI1->Name();
   mXml.Im2()   = mI2->Name();
   mXml.Box1() = Box2dr(aInf1,aSup1);
   mXml.Box2() = Box2dr(aInf2,aSup2);
   mXml.Calib() =  mI1->NM().OriCal();
   mXml.NbPts() = mPackPStd.size();
   mXml.Foc1()  = mI1->CS()->Focale();
   mXml.Foc2()  = mI2->CS()->Focale();
   mXml.FocMoy() = FocMoy();

   mXml.Geom().SetNoInit();

   // std::cout << "SIIZZZ " << mXml.NbPts() << "\n";

   if (mShow)
   {
      std::cout << "NbPts " << mPackPStd.size() << " RED " << mPackStdRed.size() << "\n";
   }
   if (mXml.NbPts()<(mSelAllIm ? NbMinPts2Im_AllSel : NbMinPts2Im) )
   {
        return;
   }
   cXml_O2IComputed aXCmp;
   RazEllips(aXCmp.Elips());
   cXml_O2ITiming & aTiming = aXCmp.Timing();
        

   for (ElPackHomologue::const_iterator itP=mPackPStd.begin() ; itP!=mPackPStd.end() ; itP++)
   {
         Pt2dr aP1 = itP->P1();
         mPInfI1.SetInf(aP1);
         mPSupI1.SetSup(aP1);
   }

   if (mShow)
   {
         double aRab = 0.02;
         Pt2dr aSz = mPSupI1 - mPInfI1;
         mP0W = mPInfI1 - aSz * aRab;
         Pt2dr aP1W = mPSupI1 + aSz * aRab;
         aSz = aP1W-mP0W;

         mScaleW  = aSzW /ElMax(aSz.x,aSz.y) ;
         Pt2di aSzWI = round_ni(aSz*mScaleW);
         mW = Video_Win::PtrWStd(aSzWI);

         Tiff_Im aTif= Tiff_Im::StdConvGen(mI1->Name(),1,false);
         Pt2di aSzI = aTif.sz();
         Im2D_U_INT1 anIm(aSzI.x,aSzI.y);
         TIm2D<U_INT1,INT> aTIm(anIm);
         ELISE_COPY(anIm.all_pts(),aTif.in(),anIm.out());

         Im2D_U_INT1 aImW(aSzWI.x,aSzWI.y);
         TIm2D<U_INT1,INT> aTImW(aImW);

         Pt2di aP;
         CamStenope * aCS1 =  aI1->CS();
         for (aP.x=0 ; aP.x<aSzWI.x ; aP.x++)
         {
             for (aP.y=0 ; aP.y<aSzWI.y ; aP.y++)
             {
                   // aTImW.oset(aP,128);
                   Pt2dr aPPhom = (Pt2dr(aP)/mScaleW + mP0W);
                   Pt2dr aPIm = aCS1->R3toF2(PZ1(aPPhom));
                   aTImW.oset(aP,aTIm.getprojR(aPIm));
             }
         }
         ELISE_COPY(aImW.all_pts(),aImW.in(),mW->ogray());
         // TestNewSel(mPackPStd);
//          ShowPack(mPackPStd,P8COL::red,2.0);
   }



   ShowPack(mPackPStd,P8COL::red,2.0);
   ShowPack(mPackStdRed,P8COL::blue,6.0);
   // ClikIn();

   InitVPairComp(mStCPairs,mPackPStd);
   InitVPairComp(mRedCPairs,mPackStdRed);


    if (mTestC2toC1)
    {
        std::cout << " Cost sol ext : " << PixExactCost(*mTestC2toC1,0.1) << "\n";

    }
    double aDist ;
    bool   Ok;
    // cElHomographie aHom = cElHomographie::RobustInit(&aDist,mPackPStd,Ok,100,80,500);
    cElHomographie aHom = cElHomographie::RobustInit
                                   (
                                       aDist,
                                       (double *)0,
                                       mQuick?mPack150:mPackStdRed,
                                       Ok,
                                       DoOriByHom ? 200 : (mQuick?20 :80),  // Nb in Ransac
                                       80,             // %
                                       DoOriByHom? 2000 :500             // Nb Total suivi par random
                                    );
    // if (ShowDetailHom) std::cout << "THom0= " << aChrono.uval() << "\n";
    aXCmp.HomWithR().Hom() = aHom.ToXml();
    aXCmp.HomWithR().ResiduHom() = aDist ;
    double aRecHom = RecouvrtHom(aHom);
    aXCmp.RecHom() = aRecHom;


    /*******************************************************/
    /*      TEST DES DIFFERENTES INITIALISATIONS           */
    /*******************************************************/

   //  AmelioreSolLinear, appellee plusieurs fois, fait :
   //     * Optimisation de la solution par bundle "rapide" qui linearise directement les equation
   //     * memorisation si meilleure que la derniere solution enregistree

   // = T00 ============== Test Patch Plan,
   // on recherche dans l'image des zone plane
   double aTimeAdj=0;
   cInterfBundle2Image * aBundle = mQuick ? mRedPvIBI  :  mFullPvIBI;
   double anErr=-1;

   if (! aInOri)
   {
         {
            ElTimer aChrono;
            if ((! mQuick) && DoOri3D)
            {
               ElRotation3D  * aRP = TestOriPlanePatch(FocMoy(),mPackStdRed,mPack150,mPack30,mW,mP0W,mScaleW);
               if (aRP)
                  AmelioreSolLinear(*aRP,"Patch Plan");
            }
            aTiming.TimePatchP() = aChrono.uval();
  
         }
  
         // = T0 ============== Nouveau test par Ransac minimal a 8 points  + ME
         // Initialisation par matrice essentielle, version a 8 point .
         // Hypothes bcp d'oulier, peu de bruit
          {
             ElTimer aChrono;
             if (DoOri3D)
             {
                ElRotation3D aMRR = TestcRanscMinimMatEss(mQuick,mPackPStd,mPackStdRed,mPack150,mPack30,FocMoy());
                AmelioreSolLinear(aMRR,"Mini RE");
             }
             aTiming.TimeRanMin() = aChrono.uval();
  
             if (false &&  mShow)
             {
                  std::cout << "TIME RanscMinim " << aTiming.TimeRanMin() << "\n";
                  getchar();
             }
          }
         // = T1 ============== Nouveau test par Ransac + ME
  
          // Deuxieme essai  de la matrice essentielle , en faisant des tirage
          // a plus de point, hypothese peu d'oulier, bcp de bruit
          {
             ElTimer aChrono;
             if (DoOri3D)
             {
                ElRotation3D aRR =RansacMatriceEssentielle(mQuick,mPackPStd,mPackStdRed,mPack150,mPack30,FocMoy());
                AmelioreSolLinear(aRR,"Ran Ess");
             }
             aTiming.TimeRansacStd() = aChrono.uval();
          }
  
          // = T2 ==============   Test par Matrices essentielles  "classique"
          //  Matrice  essentielle sur tout les points, estimation L1 puis  L2
          for (int aL2 = 0 ; aL2 < 2 ; aL2++)
          {
              ElTimer aChrono;
              if (DoOri3D)
              {
                  ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackStdRed.MepRelPhysStd(1.0,false)) ;
                  // ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackPStd.MepRelPhysStd(1.0,false)) ;
                  aR = aR.inv();
                  AmelioreSolLinear(aR,(aL2 ? "L2 Ess": "L1 Ess" ));
              }
              if (aL2)
                 aTiming.TimeL2MatEss() = aChrono.uval();
              else
                 aTiming.TimeL1MatEss() = aChrono.uval();
          }
  
          //  = T3 ============  Test par  homographie plane "classique" (i.e. globale)
  
          // Test par homographoe globale (mais avec estimation robuste)
          {
             bool ShowDetailHom = mShow && false;
             ElTimer aChrono;
  
             cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));
             if (ShowDetailHom) std::cout << "THom2= " << aChrono.uval() << "\n";
  
             const std::list<cElemMepRelCoplan>  & aLSolPl = aRMC.LElem();
  
             cXml_MepHom aXMH;
             for (std::list<cElemMepRelCoplan>::const_iterator itS = aLSolPl.begin() ; itS != aLSolPl.end() ; itS++)
             {
                 ElRotation3D aR = itS->Rot();
                 aR = aR.inv();
                 if ( itS->PhysOk())
                 {
                     aXMH.Ori().push_back(El2Xml(aR));
                     AmelioreSolLinear(aR," Plane ");
                 }
             }
             if (DoOriByHom)
             {
                
                 aXCmp.HomWithR().ForMepHom().SetVal(aXMH);
             }

             if (ShowDetailHom) std::cout << "THom3= " << aChrono.uval() << "\n";
             aTiming.TimeHomStd() = aChrono.uval();
          }
  
          // == T4 ===========  Test Rotation pure
  
          // Test rotation pure, plus ou moins une branche momentanement morte
          // Adapatee au cas des rotations pures, idee :
          //   * 1- calculer la rotation pure 
          //   * 2- s'en servir pour trouver la rotation globale
          // 
  
          {
             ElTimer aChrono;
             if (DoOri3D)
             {
                 cResMepCoc aRCoc= MEPCoCentrik(mQuick,mPackStdRed,FocMoy(),mTestC2toC1,false);
                 if (! mQuick)
                    AmelioreSolLinear(aRCoc.mSolRot,"Cocent");
  
                 if (mShow)
                 {
                    std::cout << "Ecart RPUre " << aRCoc.mCostRPure * FocMoy() << " VraiR " << aRCoc.mCostVraiRot *FocMoy();
                    if (mTestC2toC1)
                       std::cout << " DREf (pix) " << aRCoc.mMat.L2(mTestC2toC1->Mat())  * FocMoy();
                    std::cout << "\n";
                 }
                 aXCmp.RPure().Ori() = ExportMatr(aRCoc.mMat);
                 aXCmp.RPure().ResiduRP() = aRCoc.mCostRPure * FocMoy();
             }
             aTiming.TimeRPure() = aChrono.uval();
          }



          if (! mBestSolIsInit)
          {
                return;
          }
  
          // Jusqu'a present les solutions ont ete optimisee avec un bundle approximatif
          // (directement lineaire). La on va faire du "vrai" bundle
  
          // Affinage solution plus precis 
          anErr = aBundle->ErrInitRobuste(mBestSol,0.75);
          ElTimer aChrono;
          anErr = aBundle->ResiduEq(mBestSol,anErr);
          
          for (int aK=0 ; aK< (DoOri3D ? (mQuick ? 6 : 10) : 2) ; aK++)
          {
                 // std::cout << "ERRCur " <<  anErr*FocMoy() << "\n";
                 // cInterfBundle2Image * anIBI = (aK<5) ? mRedPvIBI  : mFullPvIBI;
                 ElRotation3D aSol = aBundle->OneIterEq(mBestSol,anErr);
                 mBestSol = aSol;
          }
          // finalisation sauvegarde resultats
          aTimeAdj = aChrono.uval();
   }
   else
   {
        mBestSol = *aInOri;
        anErr = aBundle->ErrInitRobuste(mBestSol,0.75);
   }

    cXml_Elips2D anElips2D;
    RazEllips(anElips2D);
    for (ElPackHomologue::const_iterator itP=mPackPStd.begin() ; itP!=mPackPStd.end() ; itP++)
    {
        AddEllips(anElips2D,itP->P1(),itP->Pds());
        //std::cout << "ewlinaaaa " << itP->P1() << "\n";
    }
    NormEllips(anElips2D);
    aXCmp.Elips2().SetVal(anElips2D);

    double anErr90 =  mFullPvIBI->ErrInitRobuste(mBestSol,0.90);
    mIA =  MedianNuage(mPackStdRed,mBestSol);
    aXCmp.OrientAff().Ori() = ExportMatr(mBestSol.Mat());
    aXCmp.OrientAff().Centre() = mBestSol.tr();
    aXCmp.OrientAff().ResiduOr() = anErr * FocMoy();
    aXCmp.OrientAff().ResiduHighPerc() = anErr90 * FocMoy();
    aXCmp.OrientAff().PMed1() = mIA;

    Pt3dr aC =  mBestSol.tr();
    aXCmp.BSurH() = euclid(Pt2dr(aC.x,aC.y)) / ElAbs(mIA.z);


    if (mShow)
        std::cout << "EERRR FINALE " << anErr*FocMoy()  << " Er90 " <<  anErr90* FocMoy() << " B/H " << aXCmp.BSurH()  << aTimeAdj << "\n";




    if (mShow)
    {
        if (mBestSolIsInit)
        {
           std::cout << "Cost " << ExactCost(mBestSol,0.1)  *FocMoy() << " Centre " << mIA << "\n";
           if (mTestC2toC1)
           {
               std::cout << "Ref, Cost " << ExactCost(*mTestC2toC1,0.1) * FocMoy() << " dist/Ref " << DistRot(*mTestC2toC1,mBestSol) <<  "\n";
           }
        }
        else
           std::cout << "NO BEST SOL\n";
    }
    // CalcAmbig();

    mXml.Geom().SetVal(aXCmp);

}

const cXml_Ori2Im &  cNewO_OrInit2Im::XmlRes() const
{
   return mXml;
}


void cNewO_OrInit2Im::DoExpMM(cNewO_OneIm * aI,const ElRotation3D & aRot,const Pt3dr & aPMed)
{
    CamStenope * aCS = aI->CS()->Dupl();
    aCS->SetOrientation(aRot);

    cOrientationConique aOC = aCS->ExportCalibGlob(aI->CS()->Sz(),aPMed.z,euclid(aPMed),0,true,"ExportMartini");
    MakeFileXML(aOC,mI1->NM().ICNM()->NameOriStenope("MartiniRel",aI->Name()));
}


void   cNewO_OrInit2Im::DoExpMM()
{
    Pt3dr aPMed = mXml.Geom().Val().OrientAff().PMed1();

    DoExpMM(mI1,ElRotation3D::Id,aPMed);
    ElRotation3D aR2(mBestSol.tr(),mBestSol.Mat(),true);
    DoExpMM(mI2,aR2.inv(),aPMed);
}


/*****************************************************************/
/*                                                               */
/*                   cNO_AppliOneCple                            */
/*                                                               */
/*****************************************************************/

class cNO_AppliOneCple : public cCommonMartiniAppli
{
    public :
          cNO_AppliOneCple(int argc,char **argv);
          void Show();
          cNewO_OrInit2Im * CpleIm();
          std::string NameXmlOri2Im(bool Bin) const;
          ElRotation3D  * OrientationRelFromExisting(std::string &);
          bool   ExpMM() const;
    private :

         cNO_AppliOneCple(const cNO_AppliOneCple &); // N.I.

         bool                 mGenOri;
         std::string          mNameIm1;
         std::string          mNameIm2;
         cNewO_NameManager *  mNM;
         cNewO_OneIm *        mIm1;
         cNewO_OneIm *        mIm2;
         std::vector<cNewO_OneIm *>  mVI;
         std::string          mNameOriTest;
         bool                 mHPP;
         // Structure pour creer la representation explicite sous forme de points multiples
         tMergeLPackH         mMergeStr;
         ElRotation3D *       mTestSol;
         ElRotation3D *       mRotInOri;
         bool                 mIsTTK;  // Test Tomasi Kanade
         bool                 mSelAllCple;  // Test Tomasi Kanade
         bool                 mExpMM;       // Do an export in MM mode
};

std::string cNO_AppliOneCple::NameXmlOri2Im(bool Bin) const
{
    // return mNM->NameXmlOri2Im(mNameIm1,mNameIm2,Bin);
    return mNM->NameXmlOri2Im(mIm1,mIm2,Bin);
}





cNO_AppliOneCple::cNO_AppliOneCple(int argc,char **argv)  :
   mGenOri   (true),
   mHPP      (true),
   mMergeStr (2,false),
   mTestSol  (0),
   mRotInOri (0),
   mExpMM    (false)
{

   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mNameIm1,"Name First Image", eSAM_IsExistFile)
                   << EAMC(mNameIm2,"Name Second Image", eSAM_IsExistFile),
        LArgMain() << EAM(mGenOri,"GenOri",true,"Generate Ori, Def=true, false for quick process to RedTieP")
                   << EAM(mHPP,"HPP",true,"Homograhic Planar Patch")
                   << EAM(mExpMM,"ExpMM",true,"Export in MicMac Std format the relative orientation")
                   << ArgCMA()
   );


   mIsTTK  = (ModeNO()==eModeNO_TTK);
   mSelAllCple = mIsTTK;

   if (MMVisualMode) return;

   // Class cNewO_NameManager classe qui permet d'acceder a tous les nom de fichier
   // crees dans Martini
   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,DirOfFile(mNameIm1),mNameOriCalib,"dat");


   // Structure d'image specialisee martini
   mIm1 = new cNewO_OneIm(*mNM,mNameIm1);
   mIm2 = new cNewO_OneIm(*mNM,mNameIm2);

   mVI.push_back(mIm1);
   mVI.push_back(mIm2);

   // NOMerge_AddAllCams : Mets dans mMergeStr les points, cree les liens multiples,
   // et transforme les points en "points photgorammetrique" (t.q (x  y 1) est une direction
   // de rayon
   //
   //  Par la suite tout le code martini, lira ces points directement

   NOMerge_AddAllCams(mMergeStr,mVI);
   mMergeStr.DoExport();

   if (EAMIsInit(&mNameOriTest))
      mTestSol = OrientationRelFromExisting(mNameOriTest);

   
   if (EAMIsInit(&mInOri))
      mRotInOri = OrientationRelFromExisting(mInOri);

/*
   if (EAMIsInit(&mBlinis))
   {
   }
*/
/*
   if (EAMIsInit(&mNameOriTest))
   {
      StdCorrecNameOrient(mNameOriTest,mNM->Dir());
      CamStenope * aCam1 = mNM->CamOriOfName(mNameIm1,mNameOriTest);
      CamStenope * aCam2 = mNM->CamOriOfName(mNameIm2,mNameOriTest);
      // aCam2->Orient() : M =>C2  ;  aCam1->Orient().inv() :  C1=>M
      // Donc la aRot = C1=>C2
      ElRotation3D aRot = (aCam2->Orient() *aCam1->Orient().inv());
      //   Maintenat Rot C2 =>C1; donc Rot( P(0,0,0)) donne le vecteur de Base
      aRot = aRot.inv();
      mTestSol = new ElRotation3D(aRot);
   }
*/
}

ElRotation3D *  cNO_AppliOneCple::OrientationRelFromExisting(std::string & aNameOri)
{
   StdCorrecNameOrient(aNameOri,mNM->Dir());
   CamStenope * aCam1 = mNM->CamOriOfNameSVP(mNameIm1,aNameOri);
   CamStenope * aCam2 = mNM->CamOriOfNameSVP(mNameIm2,aNameOri);
   if ((aCam1==0) || (aCam2==0))
      return 0;
   // aCam2->Orient() : M =>C2  ;  aCam1->Orient().inv() :  C1=>M
   // Donc la aRot = C1=>C2
   ElRotation3D aRot = (aCam2->Orient() *aCam1->Orient().inv());
   //   Maintenat Rot C2 =>C1; donc Rot( P(0,0,0)) donne le vecteur de Base
   aRot = aRot.inv();
   aRot.tr() = vunit(aRot.tr());

   return new ElRotation3D(aRot);
}

bool   cNO_AppliOneCple::ExpMM() const 
{
   return mExpMM;
}


cNewO_OrInit2Im * cNO_AppliOneCple::CpleIm()
{
   return new cNewO_OrInit2Im(mGenOri,mQuick,mIm1,mIm2,&mMergeStr,mTestSol,mRotInOri,mShow,mHPP,mSelAllCple,*this);
}

void cNO_AppliOneCple::Show()
{
}

extern void  Bench_NewOri();
Pt3dr PRand() {return Pt3dr(NRrandC(),NRrandC(),NRrandC());}

void BenchNewFoncRot()
{
    for (int aK=0 ; aK< 0 ; aK++)
    {
        Pt3dr A=PRand();
        Pt3dr B=PRand();
        Pt3dr C=PRand();

         std::cout << "Test Mxte " << scal(A,B^C) << " " << scal(C,A^B) << "\n";
          std::cout << "Test Mxte "  << ((B^ C) -( MatProVect(B) * C)) << "\n";
    }
    for (int aK=0 ; aK< 0 ; aK++)
    {
        Pt3dr A=PRand();
        Pt3dr B=PRand();
        Pt3dr aC =PRand();

        ElSeg3D aSeg(A,B);
        Pt3dr aPC = aSeg.ProjOrtho(aC);

        double aTeta = NRrandC();
        ElRotation3D aR = AffinRotationArroundAxe(aSeg,aTeta);
        Pt3dr aIC = aR.ImAff(aC);
        Pt3dr  aV0 = vunit(aC-aPC);
        Pt3dr  aV1 = vunit(aIC-aPC);

        std::cout << "INV " << euclid(A-aR.ImAff(A)) << " " << euclid(B-aR.ImAff(B)) << "\n";
        std::cout << "Orth " << scal(aSeg.Tgt(),aV1)  << "Teta " << cos(aTeta) - scal(aV0,aV1)<< "\n";
    }
}



int TestNewOriImage_main(int argc,char ** argv)
{
   // std::cout << "WARNING LEVENBERG BADLY ASSERT ALL VAL to FIX  \n";

   BenchNewFoncRot();
   // Bench_NewOri();

   // Classe qui prepare les donnees
   cNO_AppliOneCple anAppli(argc,argv);
   anAppli.Show();
  
   // Classe qui fait le calcul 
   cNewO_OrInit2Im * aCple = anAppli.CpleIm();
   const cXml_Ori2Im &  aXml = aCple->XmlRes() ;

   MakeFileXML(aXml,anAppli.NameXmlOri2Im(true));
   MakeFileXML(aXml,anAppli.NameXmlOri2Im(false));

   if (anAppli.ExpMM())
      aCple->DoExpMM();

   return EXIT_SUCCESS;
}


   //==============================================

/* 
    La commande TestAllNewOriImage_main se rappelle elle meme, si il y a une pattern de N
  images, N process vont etre lances avec l'option NameIm1=Image...
*/

int TestAllNewOriImage_main(int argc,char ** argv)
{
   std::string aPat;
   bool aGenOri=true;
   cCommonMartiniAppli aCMA;
   std::string aNameIm1;
   std::string aPatGlob;
   bool aExpTxt=0;


   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aPat,"Pattern"),
        LArgMain() << EAM(aGenOri,"GenOri",true,"Set false to accelarate pre process for RedTieP)")
                   << aCMA.ArgCMA()
                   << EAM(aNameIm1,"NameIm1",true,"Name of Image1, internal purpose")
                   << EAM(aPatGlob,"PatGlob",true,"Name of Image1, internal purpose")
                   << EAM(aExpTxt,"ExpTxt",true,"input homol format is txt? def false, binary format")
   );

   bool aModeIm1 = EAMIsInit(&aNameIm1);
	
	std::string aInHomol="dat";
   if (aExpTxt) aInHomol="txt";
   if (aModeIm1) 
      aPat = aNameIm1;
   
   cElemAppliSetFile anEASF(aPat);
   const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();
   std::string aDir = anEASF.mDir;
   cNewO_NameManager * aNM =  new cNewO_NameManager(aCMA.mExtName,aCMA.mPrefHom,aCMA.mQuick,aDir,aCMA.mNameOriCalib,"dat");
   cInterfChantierNameManipulateur* anICNM = anEASF.mICNM;

   if (!aModeIm1)
   {
       // Branche ou va lance un process par image
       MakeXmlXifInfo(anEASF.mPat,anEASF.mICNM);
       // cExeParalByPaquets => parallelise avev message d'avancement
       cExeParalByPaquets aExePaq("NewOri of One Image",aVIm->size());

       // Force la creation des directories
       for (int aK=0 ; aK<int(aVIm->size())  ; aK++)
           aNM->Dir3POneImage((*aVIm)[aK],true);

       for (int aK=0 ; aK<int(aVIm->size())  ; aK++)
       {
           std::string aName = (*aVIm)[aK];
           aNM->NameXmlOri2Im(aName,aName,true);
           std::string aCom =  GlobArcArgv  + " NameIm1=" + aName + " PatGlob="+ QUOTE(anEASF.mPat) + " ExpTxt=" + ToString(aExpTxt);

          if (aCMA.mShow) 
             std::cout << "Com= " << aCom << "\n";

           aExePaq.AddCom(aCom);
       }
       // C'est le ~cExeParalByPaquets => qui va lance l'execution
   }
   else
   {
       // Mode ou on execute vraiment pour une image
       std::string aKeySub = "NKS-Set-HomolOfOneImage@"+ aCMA.mPrefHom + "@"+ aInHomol +"@" + aNameIm1;
       const cInterfChantierNameManipulateur::tSet *   aVH = anICNM->Get(aKeySub);
       std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ aCMA.mPrefHom  + "@"+ aInHomol ;

       cListOfName aLON;

       // cElRegex anAutom(aPatGlob,10);

       const cInterfChantierSetNC::tSet * aSetGlob = anICNM->Get(aPatGlob);


       aNM->Dir3POneImage(aNameIm1,true);

       //  On parcourt les points homologues
       for (int aKH = 0 ; aKH<int(aVH->size()) ; aKH++)
       {
           std::string aNameH = (*aVH)[aKH];
           std::pair<std::string,std::string> aPair = anICNM->Assoc2To1(aKeyH,aNameH,false);
           ELISE_ASSERT(aNameIm1==aPair.first,"Incoh in NO_AllOri2Im");

           // On recupere le nom de l'image 2
           std::string aNameIm2 = aPair.second;
           std::string aNameHomReciproque = anICNM->Assoc1To2(aKeyH,aNameIm2,aNameIm1,true);

// std::cout << "N2=" << aNameIm2 << " " << aSetName->SetBasicIsIn(aNameIm2) << "\n";

           if (    ((aNameIm1<aNameIm2) || (aCMA.mAcceptUnSym  && (!ELISE_fp::exist_file(aDir+aNameHomReciproque))) )  // Pour ne faire le calcul que dans un sens
                && (ELISE_fp::exist_file(aDir+aNameIm2))  //  Precaution si qqun a detruit
                // && (anAutom.Match(aNameIm2))   // Pour que l'image soit dans le pattern
                && (BoolFind(*aSetGlob,aNameIm2))   // Pour que l'image soit dans le pattern
              )
           {


               std::string aNameCalc1 = ElMin(aNameIm1,aNameIm2);
               std::string aNameCalc2 = ElMax(aNameIm1,aNameIm2);


               std::string aNamOri = aDir + aNM->NameXmlOri2Im(aNameCalc1,aNameCalc2,true);
               // Sans doute le  IMGP7029.JPG/OriRel-IMGP7030.JPG
               // Pour ne pas refaire le calcul si deja fait
               if (! ELISE_fp::exist_file(aNamOri))
               {
                    // std::string aNameH21 = aDir +  anEASF.mICNM->Assoc1To2(aKeyH,aNameIm2,aNameIm1,true);
                    // Lance la commande qui va vraiment faire le calcul 
                    // if (ELISE_fp::exist_file(aNameH21)   || aCMA.mAcceptUnSym)
                    {
                        std::string aCom =   MM3dBinFile("TestLib NO_Ori2Im") + " " + aNameCalc1 + " " + aNameCalc2 + " ";
                        aCom = aCom + " GenOri=" + ToString(aGenOri);
                        aCom = aCom + aCMA.ComParam();
                        if (aCMA.mShow)
                           std::cout << "COM2EXE=" <<  aCom << "\n";
                        
                        System(aCom);
                    }
               }
               if (ELISE_fp::exist_file(aNamOri))
                  aLON.Name().push_back(aNameIm2);
           }
       }
       MakeFileXML(aLON,aDir + aNM->NameListeImOrientedWith(aNameIm1,true));
       MakeFileXML(aLON,aDir + aNM->NameListeImOrientedWith(aNameIm1,false));
       exit(EXIT_SUCCESS);
   }

   cXml_O2ITiming aTiming;
   aTiming.TimeRPure()     = 0;
   aTiming.TimePatchP()    = 0;
   aTiming.TimeRanMin()    = 0;
   aTiming.TimeRansacStd() = 0;
   aTiming.TimeL2MatEss()  = 0;
   aTiming.TimeL1MatEss()  = 0;
   aTiming.TimeHomStd()    = 0;


   cSauvegardeNamedRel                aLCpleOri;
   cSauvegardeNamedRel                aLCpleConc;
   std::map<std::string,cListOfName>  aMapConc;
   std::map<std::string,cListOfName>  aMapConcRev;
   // std::map<std::string,cListOfName>  aMapOri;

   for (int aK1=0 ; aK1<int(aVIm->size()) ; aK1++)
   {
       const std::string & aName1 = (*aVIm)[aK1];

       
       cListOfName aLON = StdGetFromPCP(aDir+aNM->NameListeImOrientedWith(aName1,true),ListOfName);
       for (std::list<std::string>::const_iterator itN2=aLON.Name().begin(); itN2!=aLON.Name().end(); itN2++)
       {
           const std::string & aName2 = *(itN2);

           std::string aNameCalc1 = ElMin(aName1,aName2);
           std::string aNameCalc2 = ElMax(aName1,aName2);
           cCpleString aCplCalc(aNameCalc1,aNameCalc2);

           aMapConc[aNameCalc1].Name().push_back(aNameCalc2);
           aMapConcRev[aNameCalc2].Name().push_back(aNameCalc1);

           std::string aNamOri = aDir + aNM->NameXmlOri2Im(aNameCalc1,aNameCalc2,true);
           cXml_Ori2Im  aXmlOri = StdGetFromSI(aNamOri,Xml_Ori2Im);
           if (aXmlOri.Geom().IsInit())
           {
               const cXml_O2ITiming & aLocT = aXmlOri.Geom().Val().Timing();

               aTiming.TimeRPure()     += aLocT.TimeRPure();
               aTiming.TimePatchP()    += aLocT.TimePatchP();
               aTiming.TimeRanMin()    += aLocT.TimeRanMin();
               aTiming.TimeRansacStd() += aLocT.TimeRansacStd();
               aTiming.TimeL2MatEss()  += aLocT.TimeL2MatEss();
               aTiming.TimeL1MatEss()  += aLocT.TimeL1MatEss();
               aTiming.TimeHomStd()    += aLocT.TimeHomStd();

               cCpleString aCple;
               aLCpleOri.Cple().push_back(aCplCalc);
           }
           aLCpleConc.Cple().push_back(aCplCalc);
       }
   }


   for (int aK1=0 ; aK1<int(aVIm->size()) ; aK1++)
   {
       const std::string & aName1 = (*aVIm)[aK1];
       MakeFileXML(aMapConc[aName1],aDir+aNM->NameListeImOrientedWith(aName1,true));
       MakeFileXML(aMapConc[aName1],aDir+aNM->NameListeImOrientedWith(aName1,false));

       MakeFileXML(aMapConcRev[aName1],aDir+aNM->RecNameListeImOrientedWith(aName1,true));
       MakeFileXML(aMapConcRev[aName1],aDir+aNM->RecNameListeImOrientedWith(aName1,false));
   }

   MakeFileXML(aTiming,aDir +   aNM->NameTimingOri2Im());
   MakeFileXML(aLCpleOri,aDir +   aNM->NameListeCpleOriented(true));
   MakeFileXML(aLCpleOri,aDir +   aNM->NameListeCpleOriented(false));
   MakeFileXML(aLCpleConc,aDir +   aNM->NameListeCpleConnected(true));
   MakeFileXML(aLCpleConc,aDir +   aNM->NameListeCpleConnected(false));

   return EXIT_SUCCESS;
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
