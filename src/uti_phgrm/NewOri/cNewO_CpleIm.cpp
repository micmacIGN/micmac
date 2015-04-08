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

double cNewO_CpleIm::ExactCost(const ElRotation3D & aRot,const Pt2dr & aP1,const Pt2dr & aP2,double aTetaMax) const
{
   return FONC_EXACT_COST(aRot,aP1,aP2,aTetaMax);
}
double cNewO_CpleIm::ExactCost(const ElRotation3D & aRot,double aTetaMax) const
{
    return FONC_EXACT_COST(mPackPStd,aRot,aTetaMax);
}
double  cNewO_CpleIm::PixExactCost(const ElRotation3D & aRot,double aTetaMax) const
{
   return ExactCost(aRot,aTetaMax) * FocMoy();
}





Pt2dr cNewO_CpleIm::ToW(const Pt2dr & aP) const
{
     return (aP-mP0W) *mScaleW;
}


void cNewO_CpleIm::ShowPack(const ElPackHomologue & aPack,int aCoul,double aRay)
{
    if (! mW) return;
    for (ElPackHomologue::const_iterator itP=aPack.begin() ; itP!=aPack.end() ; itP++)
        mW->draw_circle_abs(ToW(itP->P1()),aRay,mW->pdisc()(aCoul));
}

void  cNewO_CpleIm::ClikIn()
{
   if (mW) mW->clik_in();
}

double cNewO_CpleIm::FocMoy() const
{
    double aF = 1/mI1->CS()->Focale() + 1/mI2->CS()->Focale();
    return 2 / aF;
}

ElRotation3D TestOriPlanePatch
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
                  const ElPackHomologue & aPack,
                  const ElPackHomologue & aPackRed,
                  const ElPackHomologue & aPack150,
                  const ElPackHomologue & aPack30,
                  double aFoc
             ) ;



cNewO_CpleIm::cNewO_CpleIm
(
      cNewO_OneIm * aI1,
      cNewO_OneIm * aI2,
      tMergeLPackH *      aMergeTieP,
      ElRotation3D *      aTestedSol,
      bool                aShow,
      bool                aHPP
)  :
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
   mRedPvIBI    (cInterfBundle2Image::LineariseAngle(mPackStdRed,FocMoy(),true)),
   mFullPvIBI   (cInterfBundle2Image::LineariseAngle(mPackPStd,FocMoy(),true)),
   mShow        (aShow),
   mBestSol     (ElRotation3D::Id),
   mCostBestSol (1e9),
   mBestSolIsInit (false),
   mSegAmbig      (Pt3dr(0,0,0),Pt3dr(1,1,1)),
   mW             (0)
{

   if (mShow)
      std::cout << "NbPts " << mPackPStd.size() << " RED " << mPackStdRed.size() << "\n";

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
      ElRotation3D  aRP = TestOriPlanePatch(FocMoy(),mPackStdRed,mPack150,mPack30,mW,mP0W,mScaleW);
      AmelioreSolLinear(aRP,"Patch Plan");
   }

   // = T0 ============== Nouveau test par Ransac + ME
    {
       ElRotation3D aMRR = TestcRanscMinimMatEss(mPackPStd,mPackStdRed,mPack150,mPack30,FocMoy());
       AmelioreSolLinear(aMRR,"Mini RE");
    }
   // = T1 ============== Nouveau test par Ransac + ME
    {
       ElRotation3D aRR =RansacMatriceEssentielle(mPackPStd,mPackStdRed,FocMoy());
       AmelioreSolLinear(aRR,"Ran Ess");
    }

  // = T2 ==============   Test par Matrices essentielles  "classique" 
    for (int aL2 = 0 ; aL2 < 2 ; aL2++)
    {
        ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackStdRed.MepRelPhysStd(1.0,false)) ;
        // ElRotation3D aR =  (aL2 ? mPackPStd.MepRelPhysStd(1.0,true)  : mPackPStd.MepRelPhysStd(1.0,false)) ;
        aR = aR.inv();
        AmelioreSolLinear(aR,(aL2 ? "L2 Ess": "L1 Ess" ));
    }

  //  = T3 ============  Test par  homographie plane "classique" (i.e. globale) 
    double aDist ; 
    bool   Ok;
    cElHomographie aHom = cElHomographie::RobustInit(&aDist,mPackPStd,Ok,100,80,500);
    cResMepRelCoplan aRMC =  ElPackHomologue::MepRelCoplan(1.0,aHom,tPairPt(Pt2dr(0,0),Pt2dr(0,0)));

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

    // == T4 ===========  Test Rotation pure

    // Test rotation pure

    cResMepCoc aRCoc= MEPCoCentrik(mPackStdRed,FocMoy(),mTestC2toC1,false);
    AmelioreSolLinear(aRCoc.mSolRot,"Cocent");

    if (mShow)
    {
       std::cout << "Ecart RPUre " << aRCoc.mCostRPure * FocMoy() << " VraiR " << aRCoc.mCostVraiRot *FocMoy();
       if (mTestC2toC1)  
          std::cout << " DREf (pix) " << aRCoc.mMat.L2(mTestC2toC1->Mat())  * FocMoy();
       std::cout << "\n";
    }




    if (! mBestSolIsInit)
    {
        return;
    }


    // Affinage solution
    double anErr = mRedPvIBI->ErrInitRobuste(mBestSol,0.75);
    ElTimer aChrono;
    anErr = mRedPvIBI->ResiduEq(mBestSol,anErr);
    for (int aK=0 ; aK< 10 ; aK++)
    {
         // std::cout << "ERRCur " <<  anErr*FocMoy() << "\n";
         cInterfBundle2Image * anIBI = (aK<5) ? mRedPvIBI  : mFullPvIBI;
         ElRotation3D aSol = anIBI->OneIterEq(mBestSol,anErr);
         mBestSol = aSol;
    }

    if (mShow)
        std::cout << "EERRR FINALE " << anErr*FocMoy() << " " << aChrono.uval() << "\n";

    mIA =  MedianNuage(mPackStdRed,mBestSol);



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
    private :
         typedef cFixedMergeTieP<2,Pt2dr> tMerge;


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
};





cNO_AppliOneCple::cNO_AppliOneCple(int argc,char **argv)  :
   mShow (false),
   mHPP  (true)
{

   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(mNameIm1,"Name First Image")
                   <<  EAMC(mNameIm2,"Name Second Image"),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ")
                   << EAM(mNameOriTest,"OriTest",true,"Orientation for test to a reference")
                   << EAM(mShow,"Show",true,"Orientation for test to a reference")
                   << EAM(mHPP,"HPP",true,"Homograhic Planar Patch")
   );


   mNM = new cNewO_NameManager("./",mNameOriCalib,"dat");
   mIm1 = new cNewO_OneIm(*mNM,mNameIm1);
   mIm2 = new cNewO_OneIm(*mNM,mNameIm2);

   mVI.push_back(mIm1);
   mVI.push_back(mIm2);
   cFixedMergeStruct<2,Pt2dr> aMergeStr;
   NOMerge_AddAllCams(aMergeStr,mVI);

   aMergeStr.DoExport();

   ElRotation3D * aTestSol = 0;
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
      aTestSol = new ElRotation3D(aRot);
   }

//    cNewO_CombineCple aARI(aMergeStr,aTestSol);
   cNewO_CpleIm aCple(mIm1,mIm2,&aMergeStr,aTestSol,mShow,mHPP);
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
/*
*/

   return EXIT_SUCCESS;
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
