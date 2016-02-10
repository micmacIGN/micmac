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
      bool          aQuick,
      cNewO_OneIm * aI1,
      cNewO_OneIm * aI2,
      tMergeLPackH *      aMergeTieP,
      ElRotation3D *      aTestedSol,
      bool                aShow,
      bool                aHPP
)  :
   mQuick       (aQuick),
   mI1          (aI1),
   mI2          (aI2),
   mMergePH     (aMergeTieP),
   mTestC2toC1  (aTestedSol),
   mPackPDist   (ToStdPack(mMergePH,true,0.1)),
   mPackPStd    (ToStdPack(mMergePH,false,0.1)),
   mPInfI1      (1e5,1e5),
   mPSupI1      (-1e5,-1e5),
   mPackStdRed  (PackReduit(mPackPStd,1500,500)),
   mPack150     (PackReduit(mPackStdRed,100)),
   mPack30      (PackReduit(mPack150,30)),
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
   mW             (0)
{
    // Sauvegarde des point hom flottants 
    {
        const tLMCplP & aLM = mMergePH->ListMerged();
        tVP2f aVP1;
        tVP2f aVP2;
        tVUI1 aVNb;
 
         for ( tLMCplP::const_iterator itC=aLM.begin() ; itC!=aLM.end() ; itC++)
         {
             const Pt2dr & aP1 = (*itC)->GetVal(0);
             const Pt2dr & aP2 = (*itC)->GetVal(1);
             aVP1.push_back(Pt2df(aP1.x,aP1.y));
             aVP2.push_back(Pt2df(aP2.x,aP2.y));
             aVNb.push_back((*itC)->NbArc());
         }
         mI1->NM().Dir3PDeuxImage(mI1,mI2,true);
         std::string aNameH = mI1->NM().NameHomFloat(mI1,mI2);
         mI1->NM().WriteCouple(aNameH,aVP1,aVP2,aVNb);
    }

   mXml.Im1()   = mI1->Name();
   mXml.Im2()   = mI2->Name();
   mXml.Calib() =  mI1->NM().OriCal();
   mXml.NbPts() = mPackPStd.size();
   mXml.Foc1()  = mI1->CS()->Focale();
   mXml.Foc2()  = mI2->CS()->Focale();
   mXml.FocMoy() = FocMoy();

   mXml.Geom().SetNoInit();

   // std::cout << "SIIZZZ " << mXml.NbPts() << "\n";

   if (mShow)
      std::cout << "NbPts " << mPackPStd.size() << " RED " << mPackStdRed.size() << "\n";
   if (mXml.NbPts()<NbMinPts2Im)
   {
        return;
   }
   cXml_O2IComputed aXCmp;
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


    /*******************************************************/
    /*      TEST DES DIFFERENTES INITIALISATIONS           */
    /*******************************************************/
   // = T00 ============== Test Patch Plan

   {
      ElTimer aChrono;
      if (! mQuick)
      {
         ElRotation3D  * aRP = TestOriPlanePatch(FocMoy(),mPackStdRed,mPack150,mPack30,mW,mP0W,mScaleW);
         if (aRP)
            AmelioreSolLinear(*aRP,"Patch Plan");
      }
      aTiming.TimePatchP() = aChrono.uval();

   }

   // = T0 ============== Nouveau test par Ransac minimal a 8 points  + ME
    {
       ElTimer aChrono;
       ElRotation3D aMRR = TestcRanscMinimMatEss(mQuick,mPackPStd,mPackStdRed,mPack150,mPack30,FocMoy());
       AmelioreSolLinear(aMRR,"Mini RE");
       aTiming.TimeRanMin() = aChrono.uval();

       if (false &&  mShow)
       {
            std::cout << "TIME RanscMinim " << aTiming.TimeRanMin() << "\n";
            getchar();
       }
    }
   // = T1 ============== Nouveau test par Ransac + ME
    {
       ElTimer aChrono;
       ElRotation3D aRR =RansacMatriceEssentielle(mQuick,mPackPStd,mPackStdRed,mPack150,mPack30,FocMoy());
/*
       if (true ||  mShow)
       {
            std::cout << "TIME RansacStd " << aChrono.uval() << "\n";
            exit(-1);
       }
*/
       AmelioreSolLinear(aRR,"Ran Ess");
       aTiming.TimeRansacStd() = aChrono.uval();
    }

  // = T2 ==============   Test par Matrices essentielles  "classique"
    for (int aL2 = 0 ; aL2 < 2 ; aL2++)
    {
        ElTimer aChrono;
        ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackStdRed.MepRelPhysStd(1.0,false)) ;
        // ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackPStd.MepRelPhysStd(1.0,false)) ;
        aR = aR.inv();
        AmelioreSolLinear(aR,(aL2 ? "L2 Ess": "L1 Ess" ));
        if (aL2)
           aTiming.TimeL2MatEss() = aChrono.uval();
        else
           aTiming.TimeL1MatEss() = aChrono.uval();
    }

  //  = T3 ============  Test par  homographie plane "classique" (i.e. globale)

    {
       bool ShowDetailHom = mShow && false;
       ElTimer aChrono;
       double aDist ;
       bool   Ok;
       // cElHomographie aHom = cElHomographie::RobustInit(&aDist,mPackPStd,Ok,100,80,500);
       cElHomographie aHom = cElHomographie::RobustInit
                             (
                                 aDist,
                                 (double *)0,
                                 mQuick?mPack150:mPackStdRed,
                                 Ok,
                                 mQuick?20 :80,
                                 80,
                                 500
                              );
       if (ShowDetailHom) std::cout << "THom0= " << aChrono.uval() << "\n";
       aXCmp.HomWithR().Hom() = aHom.ToXml();
       aXCmp.HomWithR().ResiduHom() = aDist * FocMoy();
       double aRecHom = RecouvrtHom(aHom);
       if (ShowDetailHom) std::cout << "THom1= " << aChrono.uval() << "\n";
          if (mShow)
       std::cout << "   #### Residu Homographie " << aDist *FocMoy()  << " Recvrt=" << aRecHom << "\n";

       cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));
       if (ShowDetailHom) std::cout << "THom2= " << aChrono.uval() << "\n";

       const std::list<cElemMepRelCoplan>  & aLSolPl = aRMC.LElem();

       for (std::list<cElemMepRelCoplan>::const_iterator itS = aLSolPl.begin() ; itS != aLSolPl.end() ; itS++)
       {
           ElRotation3D aR = itS->Rot();
           aR = aR.inv();
           if ( itS->PhysOk())
           {
               AmelioreSolLinear(aR," Plane ");
           }
       }
       if (ShowDetailHom) std::cout << "THom3= " << aChrono.uval() << "\n";
       aTiming.TimeHomStd() = aChrono.uval();
       aXCmp.RecHom() = aRecHom;
    }

    // == T4 ===========  Test Rotation pure

    // Test rotation pure

    {
       ElTimer aChrono;
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
       aTiming.TimeRPure() = aChrono.uval();
    }




    if (! mBestSolIsInit)
    {
        return;
    }


    // Affinage solution
    cInterfBundle2Image * aBundle = mQuick ? mRedPvIBI  :  mFullPvIBI;
    double anErr = aBundle->ErrInitRobuste(mBestSol,0.75);
    ElTimer aChrono;
    anErr = aBundle->ResiduEq(mBestSol,anErr);
    for (int aK=0 ; aK< (mQuick ? 6 : 10) ; aK++)
    {
         // std::cout << "ERRCur " <<  anErr*FocMoy() << "\n";
         // cInterfBundle2Image * anIBI = (aK<5) ? mRedPvIBI  : mFullPvIBI;
         ElRotation3D aSol = aBundle->OneIterEq(mBestSol,anErr);
         mBestSol = aSol;
    }
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
        std::cout << "EERRR FINALE " << anErr*FocMoy()  << " Er90 " <<  anErr90* FocMoy() << " B/H " << aXCmp.BSurH()  << aChrono.uval() << "\n";




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



/*****************************************************************/
/*                                                               */
/*                   cNO_AppliOneCple                            */
/*                                                               */
/*****************************************************************/

class cNO_AppliOneCple
{
    public :
          cNO_AppliOneCple(int argc,char **argv);
          void Show();
          cNewO_OrInit2Im * CpleIm();
          std::string NameXmlOri2Im(bool Bin) const;
    private :

         cNO_AppliOneCple(const cNO_AppliOneCple &); // N.I.

         bool                 mQuick;
         std::string          mPrefHom;
         std::string          mNameIm1;
         std::string          mNameIm2;
         std::string          mNameOriCalib;
         cNewO_NameManager *  mNM;
         cNewO_OneIm *        mIm1;
         cNewO_OneIm *        mIm2;
         std::vector<cNewO_OneIm *>  mVI;
         std::string          mNameOriTest;
         bool                 mShow;
         bool                 mHPP;
         tMergeLPackH         mMergeStr;
         ElRotation3D *       mTestSol;
};

std::string cNO_AppliOneCple::NameXmlOri2Im(bool Bin) const
{
    // return mNM->NameXmlOri2Im(mNameIm1,mNameIm2,Bin);
    return mNM->NameXmlOri2Im(mIm1,mIm2,Bin);
}




cNO_AppliOneCple::cNO_AppliOneCple(int argc,char **argv)  :
   mQuick   (false),
   mPrefHom (""),
   mShow    (false),
   mHPP     (true),
   mMergeStr (2,false),
   mTestSol (0)
{

   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(mNameIm1,"Name First Image", eSAM_IsExistFile)
                   <<  EAMC(mNameIm2,"Name Second Image", eSAM_IsExistFile),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration", eSAM_IsExistDirOri)
                   << EAM(mNameOriTest,"OriTest",true,"Orientation for test to a reference", eSAM_IsExistDirOri)
                   << EAM(mShow,"Show",true,"Show")
                   << EAM(mQuick,"Quick",true,"Quick option adapted for UAV or easy acquisition, def = true")
                   << EAM(mPrefHom,"PrefHom",true,"Prefix Homologous points, def=\"\"")
                   << EAM(mHPP,"HPP",true,"Homograhic Planar Patch")
   );

   if (MMVisualMode) return;

   mNM = new cNewO_NameManager(mPrefHom,mQuick,DirOfFile(mNameIm1),mNameOriCalib,"dat");


   mIm1 = new cNewO_OneIm(*mNM,mNameIm1);
   mIm2 = new cNewO_OneIm(*mNM,mNameIm2);

   mVI.push_back(mIm1);
   mVI.push_back(mIm2);

   NOMerge_AddAllCams(mMergeStr,mVI);
   mMergeStr.DoExport();

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
}

cNewO_OrInit2Im * cNO_AppliOneCple::CpleIm()
{
   return new cNewO_OrInit2Im(mQuick,mIm1,mIm2,&mMergeStr,mTestSol,mShow,mHPP);
}

void cNO_AppliOneCple::Show()
{
    // std::cout << "NbTiep " << .size() << "\n";
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
   cNO_AppliOneCple anAppli(argc,argv);
   anAppli.Show();
   cNewO_OrInit2Im * aCple = anAppli.CpleIm();
   const cXml_Ori2Im &  aXml = aCple->XmlRes() ;

   MakeFileXML(aXml,anAppli.NameXmlOri2Im(true));
   MakeFileXML(aXml,anAppli.NameXmlOri2Im(false));

   return EXIT_SUCCESS;
}


   //==============================================

int TestAllNewOriImage_main(int argc,char ** argv)
{
   std::string aPat,aNameOriCalib;
   bool aQuick=false;
   std::string aPrefHom;
   std::map<std::string,cListOfName > aMLCpleOk;


   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(aPat,"Pattern"),
        LArgMain() << EAM(aNameOriCalib,"OriCalib",true,"Orientation for calibration ")
                   << EAM(aQuick,"Quick",true,"Quick option, adapted to simple acquisition (def=false)")
                   << EAM(aPrefHom,"PrefHom",true,"Prefix Homologous")
   );

   cElemAppliSetFile anEASF(aPat);
   const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();
   std::string aDir = anEASF.mDir;

   cNewO_NameManager * aNM =  new cNewO_NameManager(aPrefHom,aQuick,aDir,aNameOriCalib,"dat");

   // Force la creation des directories
   for (int aK=0 ; aK<int(aVIm->size())  ; aK++)
   {
       std::string aName = (*aVIm)[aK];
       // aNM->NameXmlOri2Im((*aVIm)[aK],(*aVIm)[aK],true);
       aNM->NameXmlOri2Im(aName,aName,true);
       aMLCpleOk[aName] = cListOfName();
   }

/*
*/

   std::list<std::string> aLCom;
   std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ aPrefHom  + "@dat";
   int aCptCom = 1;
   int aNbVideCom = 200;
   ElTimer aChrono;
   for (int aK1=0 ; aK1<int(aVIm->size()) ; aK1++)
   {
       const std::string & aName1 = (*aVIm)[aK1];
       for (int aK2=0 ; aK2<int(aVIm->size()) ; aK2++)
       {
           const std::string & aName2 = (*aVIm)[aK2];
           if (aName1<aName2)
           {
               std::string aNamOri = aDir + aNM->NameXmlOri2Im(aName1,aName2,true);
               if (! ELISE_fp::exist_file(aNamOri))
               {
                    std::string aNameH12 = aDir +  anEASF.mICNM->Assoc1To2(aKeyH,aName1,aName2,true);
                    std::string aNameH21 = aDir +  anEASF.mICNM->Assoc1To2(aKeyH,aName2,aName1,true);
                    if (ELISE_fp::exist_file(aNameH12) && ELISE_fp::exist_file(aNameH21))
                    {
                        std::string aCom =   MM3dBinFile("TestLib NO_Ori2Im") + " " + aName1 + " " + aName2 + " ";
                        if (EAMIsInit(&aNameOriCalib))
                           aCom = aCom + " OriCalib=" + aNameOriCalib;
                        aCom = aCom + " Quick=" + ToString(aQuick);
                        aCom = aCom + " PrefHom=" + aPrefHom;

                        aLCom.push_back(aCom);
                    }
                    if ((aCptCom%aNbVideCom)==0)
                    {
                       cEl_GPAO::DoComInParal(aLCom);
                       aLCom.clear();
                       int aNbIm = (int)aVIm->size();
                       std::cout << "    Done  "  << aCptCom  << " on " <<  (aNbIm *(aNbIm-1)) /2  << " in T=" << aChrono.uval() << "\n";
                    }
                    aCptCom++;
               }
           }
       }
   }

   cEl_GPAO::DoComInParal(aLCom);

   cXml_O2ITiming aTiming;
   aTiming.TimeRPure()     = 0;
   aTiming.TimePatchP()    = 0;
   aTiming.TimeRanMin()    = 0;
   aTiming.TimeRansacStd() = 0;
   aTiming.TimeL2MatEss()  = 0;
   aTiming.TimeL1MatEss()  = 0;
   aTiming.TimeHomStd()    = 0;


   cSauvegardeNamedRel                aLCple;

   for (int aK1=0 ; aK1<int(aVIm->size()) ; aK1++)
   {
       const std::string & aName1 = (*aVIm)[aK1];
       for (int aK2=0 ; aK2<int(aVIm->size()) ; aK2++)
       {
           const std::string & aName2 = (*aVIm)[aK2];
           if (aName1<aName2)
           {
               std::string aNamOri = aDir + aNM->NameXmlOri2Im(aName1,aName2,true);
               if ( ELISE_fp::exist_file(aNamOri))
               {
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

                      aMLCpleOk[aName1].Name().push_back(aName2);
                      aMLCpleOk[aName2].Name().push_back(aName1);
                      cCpleString aCple;
                      aLCple.Cple().push_back(cCpleString(aName1,aName2));
                   }
               }
           }
       }
   }

   for
   (
       std::map<std::string,cListOfName >::const_iterator  itM=aMLCpleOk.begin();
       itM !=aMLCpleOk.end();
       itM++
   )
   {
          MakeFileXML(itM->second,aDir + aNM->NameListeImOrientedWith(itM->first,true));
          MakeFileXML(itM->second,aDir + aNM->NameListeImOrientedWith(itM->first,false));
   }

   MakeFileXML(aTiming,aDir +   aNM->NameTimingOri2Im());
   MakeFileXML(aLCple,aDir +   aNM->NameListeCpleOriented(true));
   MakeFileXML(aLCple,aDir +   aNM->NameListeCpleOriented(false));

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
