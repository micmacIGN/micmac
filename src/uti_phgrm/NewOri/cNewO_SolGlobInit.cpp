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
#include "SolInitNewOri.h"

double DefMedianPond(double aDefVal,int aDefKeMed,std::vector<Pt2df> &  aV,int * aKMed)
{
   if (aV.size() == 0)
   {
      if (aKMed) 
         *aKMed = aDefKeMed;
      return aDefVal;
   }
   return MedianPond(aV,aKMed);
}


double DistBase(Pt3dr  aB1,Pt3dr  aB2)
{
      if (scal(aB1,aB2) < 0) aB2 = - aB2;
      double aD1 = euclid(aB1);
      double aD2 = euclid(aB2);

      if (aD1 > aD2) 
         aB1 = aB1 * (aD2/aD1);
      else
         aB2 = aB2 * (aD1/aD2);

      return euclid(aB1-aB2);
}

double DistanceRot(const ElRotation3D & aR1,const ElRotation3D & aR2,double aBSurH)
{
      ElMatrix<double> aDif = aR1.Mat() - aR2.Mat();
      double aDistRot = sqrt(aDif.L2());
      double aDistTr =  DistBase(aR1.tr(),aR2.tr()) * aBSurH;

      return aDistTr + aDistRot;
}

// Calcul robuste d'un element moyen comme etant celui qui minimise la somme des distance
#if (0)

#endif

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrSom                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrSom::cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli) :
   mName          (aName),
   mAppli         (&anAppli),
   mIm            (new cNewO_OneIm(mAppli->NM(),mName)),
   mCurRot        (ElRotation3D::Id),
   mTestRot       (ElRotation3D::Id),
   mSomGainByTriplet  (0.0),
   mNbGainByTriplet   (0),
   mCalcGainByTriplet (0.0),
   mSomPdsReMoy       (0.0),
   mSomTrReMoy        (0,0,0),
   mSomMatReMoy       (3,3,0.0),
   mCamInOri          (0)
{
   ReInit();
   if (anAppli.HasInOri())
   {
       mCamInOri = anAppli.NM().CamOriOfNameSVP(aName,anAppli.InOri());

       // std::cout << "cNOSolIn_AttrSomcNOSolIn_AttrSom " << mName << " " << mCamInOri << "\n";
   }
}

void cNOSolIn_AttrSom::ReInit()
{
    mCurCostMin = 1e20;
}

void cNOSolIn_AttrSom::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}

void cNOSolIn_AttrSom::ResetGainByTriplet()
{
   mSomGainByTriplet = 0;
   mNbGainByTriplet = 0;
}

void  cNOSolIn_AttrSom::AddGainByTriplet(const double & aVal)
{
    mSomGainByTriplet += aVal;
    mNbGainByTriplet += 1;
    mCalcGainByTriplet = mSomGainByTriplet / sqrt((double)mNbGainByTriplet);
}


/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_Triplet                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_Triplet::cNOSolIn_Triplet(cAppli_NewSolGolInit* anAppli,tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit & aTrip) :
    mAppli (anAppli),
    mR2on1 (Xml2El(aTrip.Ori2On1())),
    mR3on1 (Xml2El(aTrip.Ori3On1())),
    mBOnH  (aTrip.BSurH()),
    mNb3   (aTrip.NbTriplet()),
    mPMed  (aTrip.PMed()),
    mNumCC (-1)
{
   mSoms[0] = aS1;
   mSoms[1] = aS2;
   mSoms[2] = aS3;
}

bool cNOSolIn_Triplet::TripletIsInOri()
{
    for (int aK=0 ; aK<3 ; aK++)
        if (!mSoms[aK]->attr().CamInOri())
           return false;
    return  true;
}

void cNOSolIn_Triplet::SetArc(int aK,tArcNSI * anArc)
{
   mArcs[aK] = anArc;
}

// Initialise une solution globale sur ce triplet
void cNOSolIn_Triplet::InitRot3Som()
{
     mSoms[0]->attr().CurRot() = ElRotation3D::Id;
     mSoms[1]->attr().CurRot() = mR2on1;
     mSoms[2]->attr().CurRot() = mR3on1;
}

// Verifie coherence Arc/Sommet 
void cNOSolIn_Triplet::CheckArcsSom()
{
   for (int aK=0 ; aK<3 ; aK++)
   {
      tSomNSI  * aS1 = mSoms[aK];
      tSomNSI  * aS2 = mSoms[(aK+1)%3];
      tSomNSI  * aS3 = mSoms[(aK+2)%3];

      tSomNSI *  aSA1 = & (mArcs[aK]->s1());
      tSomNSI *  aSA2 = & (mArcs[aK]->s2());

      ELISE_ASSERT((aS1!=aSA1) && (aS1!=aSA2),"cNOSolIn_Triplet::CheckArcsSom");

      ELISE_ASSERT( (aS2==aSA1) ,"cNOSolIn_Triplet::CheckArcsSom");
      ELISE_ASSERT( (aS3==aSA2) ,"cNOSolIn_Triplet::CheckArcsSom");
   }
}


// Les arcs ayant une rotation "moyenne" calculee sur tout les triplets les contenant
// on estime le triplet

void  cNOSolIn_Triplet::CalcCoherFromArcs(bool Test)
{
   double aSomD = 0.0;
   double aTabD[3];

   for (int aK=0 ; aK<3 ; aK++)
   {
       ElRotation3D aRArc2to1 = mArcs[aK]->attr().EstimC2toC1();

       ElRotation3D aRTri2to1 = RotationC2toC1(mArcs[aK],this);
       double aD = DistanceRot(aRArc2to1,aRTri2to1,mBOnH);

       aSomD += aD;
       aTabD[aK] = aD;

       if (Test) 
       {
            std::cout << "Tri " << aK << " D=" << aD  <<  " " <<  mArcs[aK]->s1().attr().Im()->Name()  <<  " " << mArcs[aK]->s2().attr().Im()->Name()<< "\n";
            std::cout << " AAA " << DistanceRot(aRArc2to1,aRTri2to1,0) << " " << DistanceRot(aRArc2to1,aRTri2to1.inv(),0) << "\n";
       }
     
   }

   mCostArcMed = ElMedian(aTabD[0],aTabD[1],aTabD[2]);


   mCostArc = aSomD / 3.0;
}


void cNOSolIn_Triplet::Show(const std::string &aMes) const 
{
    std::cout << aMes
              << " Trii " << mCostArc  << " MED " << mCostArcMed << " CREL " <<  ((mCostArc /3) / mAppli->CoherMed12())
              << " "  << mSoms[0]->attr().Im()->Name() 
              << " "  << mSoms[1]->attr().Im()->Name() 
              << " "  << mSoms[2]->attr().Im()->Name() 
              << "\n";
}

double cNOSolIn_Triplet::CoherTest() const
{
    std::vector<ElRotation3D> aVRLoc;
    std::vector<ElRotation3D> aVRAbs;
    for (int aK=0 ; aK<3 ; aK++)
    {
         aVRLoc.push_back(RotOfK(aK));
         aVRAbs.push_back(mSoms[aK]->attr().TestRot());
    }

    //
    cSolBasculeRig aSolRig = cSolBasculeRig::SolM2ToM1(aVRAbs,aVRLoc);
    double aRes=0;

    for (int aK=0 ; aK<3 ; aK++)
    {
          const ElRotation3D & aRAbs =  aVRAbs[aK];
          const ElRotation3D & aRLoc =  aVRLoc[aK];
          ElRotation3D   aRA2 = aSolRig.TransformOriC2M(aRLoc);

          double aD = DistanceRot(aRAbs,aRA2,mBOnH);
          aRes += aD;
    }
    aRes = aRes / 3.0;


    return aRes;
}

/***************************************************************************/
/*                                                                         */
/*                 cLinkTripl                                              */
/*                                                                         */
/***************************************************************************/

tSomNSI * cLinkTripl::S1() const {return  m3->KSom(mK1);}
tSomNSI * cLinkTripl::S2() const {return  m3->KSom(mK2);}
tSomNSI * cLinkTripl::S3() const {return  m3->KSom(mK3);}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrASym                                       */
/*                                                                         */
/***************************************************************************/


void  cNOSolIn_AttrASym::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}


cNOSolIn_AttrASym::cNOSolIn_AttrASym() :
    mEstimC2toC1 (ElRotation3D::Id)            
{
}


void cNOSolIn_AttrASym::PostInit(bool Show)
{
   std::vector<double> aVBSurH;
   for (int aKL=0 ; aKL<int(mLnk3.size()) ; aKL++)
   {
        cLinkTripl & aLnk = mLnk3[aKL];
        cNOSolIn_Triplet & aTri = *(aLnk.m3);

        ElRotation3D aR1 = aTri.RotOfSom(aLnk.S1());
        ElRotation3D aR2 = aTri.RotOfSom(aLnk.S2());

        double aD12 =  euclid(aR1.tr()-aR2.tr());  // Base distance
        double aD1M =  euclid(aR1.tr()-aTri.PMed());  // depth cam1
        double aD2M =  euclid(aR2.tr()-aTri.PMed());  // depth cam2

        double aBOnH = aD12/((aD1M+aD2M)/2.0);  // Basic formula
        aVBSurH.push_back(aBOnH);

        if (Show)
        {
           std::cout << "B/H " << aBOnH << "\n";
        }
   }
   mBOnH = MedianeSup(aVBSurH);
}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrArc                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrArc::cNOSolIn_AttrArc(cNOSolIn_AttrASym * anASym,bool OrASym) :
   mASym   (anASym),
   mOrASym (OrASym)
{
}

/***************************************************************************/
/*                                                                         */
/*                 cAppli_NewSolGolInit                                    */
/*                                                                         */
/***************************************************************************/

void cAppli_NewSolGolInit::SetCurNeigh3(tSomNSI * aSom)
{
     if (! aSom->flag_kth(mFlag3))
     {
        mVCur3.push_back(aSom);
        aSom->flag_set_kth_true(mFlag3);
     }
}

void cAppli_NewSolGolInit::SetCurNeigh2(tSomNSI * aSom)
{
     if (! aSom->flag_kth(mFlag2))
     {
        mVCur2.push_back(aSom);
        aSom->flag_set_kth_true(mFlag2);
     }
}

/*

    mTestTrip
    90 91 92     / 91 92 86

*/

void AssertArcOriented(tArcNSI * anArc)
{
      tSomNSI * aS1 = &anArc->s1();
      tSomNSI * aS2 = &anArc->s2();
      ELISE_ASSERT(aS1<aS2,"AssertArcOriented");
}



ElRotation3D RotationC2toC1(tArcNSI * anArc,cNOSolIn_Triplet * aTri)
{
   return aTri->RotOfSom(&anArc->s1()).inv() *  aTri->RotOfSom(&anArc->s2());
}



double DistCoherence1to2 (tArcNSI * anArc,cNOSolIn_Triplet * aTriA,cNOSolIn_Triplet * aTriB)
{
     AssertArcOriented(anArc);
     return DistanceRot
            (
                RotationC2toC1(anArc,aTriA),  
                RotationC2toC1(anArc,aTriB),
                MoyHarmonik(aTriA->BOnH(),aTriB->BOnH())
            );
}


double DistCoherenceAtoB(tArcNSI * anArc,cNOSolIn_Triplet * aTriA,cNOSolIn_Triplet * aTriB)
{
      AssertArcOriented(anArc);

      const ElRotation3D  &  aR1A =  aTriA->RotOfSom(&anArc->s1());   // Cam1 => MondA
      const ElRotation3D  &  aR2A =  aTriA->RotOfSom(&anArc->s2());   // Cam2 => MondA
      const ElRotation3D  &  aR1B =  aTriB->RotOfSom(&anArc->s1());   // Cam1 => MondB
      const ElRotation3D  &  aR2B =  aTriB->RotOfSom(&anArc->s2());   // Cam2 => MondB




      ElRotation3D  aR1AtoB =  aR1B * aR1A.inv() ;  //  (Cam1 => MondB) * (MondA = > Cam1)
      ElRotation3D  aR2AtoB =  aR2B * aR2A.inv() ;  //  (Cam1 => MondB) * (MondA = > Cam1)


      ElMatrix<double> aMatA2B = NearestRotation((aR1AtoB.Mat() + aR2AtoB.Mat()) * 0.5);

      double aD1 = (aMatA2B-aR1AtoB.Mat()).L2();
      double aD2 = (aMatA2B-aR2AtoB.Mat()).L2();


      Pt3dr aVA12 = aMatA2B * (aR2A.tr()- aR1A.tr());
      Pt3dr aVB12 = aR2B.tr()- aR1B.tr();

      double aDistRot = sqrt(aD1 + aD2) * (2.0/3);
      double aBOnH = MoyHarmonik(aTriA->BOnH(),aTriB->BOnH());
      double aDistTr =  DistBase(aVA12,aVB12) * aBOnH;



       // std::cout << "TTTTT " <<  euclid(aVA12 - aVB12) /  DistBase(aVA12,aVB12) << "\n";


      // std::cout << "Dist " << aDistRot*1000 << " " << aDistTr*1000 << " BH " << aBOnH << "; " << aTriA->BOnH() << " , " << aTriB->BOnH() << "\n";

      return aDistRot + aDistTr ;
}


bool cAppli_NewSolGolInit::TripletIsValide(cNOSolIn_Triplet * aTri)
{
    return  (aTri->CostArc() < mSeuilCostArc) ||  ((aTri->CostArcMed()*PenalMedMed) < mSeuilCostArc);
}



void cAppli_NewSolGolInit::TestInitRot(tArcNSI * anArc,const cLinkTripl & aLnk)

{
      // Mp = M' coordonnees monde du triplet
      // M = coordonnees mondes courrament construite
      // La tranfo M' -> peut etre construite de deux maniere
      ElRotation3D aR1Mp2M  = aLnk.S1()->attr().CurRot() * aLnk.m3->RotOfSom(aLnk.S1()).inv();
      ElRotation3D aR2Mp2M  = aLnk.S2()->attr().CurRot() * aLnk.m3->RotOfSom(aLnk.S2()).inv();


      // ElRotation3D aTest = aR1Mp2M * aR2Mp2M.inv();
      // ElMatrix<double> aMT = aTest.Mat() -  ElMatrix<double>(3,true);
      ElMatrix<double>  aMT = aR1Mp2M.Mat() - aR2Mp2M.Mat();
      std::cout << "DIST MAT " << sqrt(aMT.L2())/3  << " IsTrip2=" << (mTestTrip2==aLnk.m3)  << "S3 " << aLnk.S3()->attr().Im()->Name() << "\n";

      if (mTestTrip)
      {
          DistCoherenceAtoB(anArc,mTestTrip,aLnk.m3);
          // std::cout << "DCCCC " << DistCoherence(anArc,mTestTrip,aLnk.m3)  << "\n";
          ELISE_ASSERT(aLnk.S1()->attr().Im()->Name() == aLnk.m3->KSom(aLnk.mK1)->attr().Im()->Name(),"AAAAAaaaa");
          ELISE_ASSERT(aLnk.S2()->attr().Im()->Name() == aLnk.m3->KSom(aLnk.mK2)->attr().Im()->Name(),"AAAAAaaaa");

/*
          std::cout << "PERM " << (int) aLnk.mK1 << " " << (int) aLnk.mK2 << " " << (int) aLnk.mK3 << "\n";
          std::cout << mTestTrip->KSom(0)->attr().Im()->Name() << " "
                    << mTestTrip->KSom(1)->attr().Im()->Name() << " "
                    << mTestTrip->KSom(2)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S1()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK1)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S2()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK2)->attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S3()->attr().Im()->Name() << " " <<  aLnk.m3->KSom(aLnk.mK3)->attr().Im()->Name() << "\n";
          std::cout <<  anArc->s1().attr().Im()->Name() << " " <<  anArc->s2().attr().Im()->Name() << "\n";
          std::cout <<  aLnk.S3()->attr().Im()->Name()  << " " <<  aLnk.m3  << "\n";
          getchar();
*/
      }
     // :Pt3dr 
}

void cAppli_NewSolGolInit::SetNeighTriplet(cNOSolIn_Triplet * aTripl)
{
    // On ajoute le triplet lui meme
    for (int aK=0 ; aK< 3 ; aK++)
    {
        tSomNSI * aKS = aTripl->KSom(aK);
        SetCurNeigh3(aKS);
        SetCurNeigh2(aKS);
    }
    aTripl->InitRot3Som();



    //  On recheche les sommet voisin 
    for (int aKA=0 ; aKA< 3 ; aKA++)
    {
         tArcNSI *  anA = aTripl->KArc(aKA);
         if (aTripl==mTestTrip) std::cout << "================ ARC ===== " << anA->s1().attr().Im()->Name() << " " <<  anA->s2().attr().Im()->Name() << "\n";

         std::vector<cLinkTripl> &  aLK3 = anA->attr().ASym()->Lnk3() ;
         for (int aK3=0 ; aK3 <int(aLK3.size()) ; aK3++)
         {
             tSomNSI * aSom = aLK3[aK3].S3();
             if (! aSom->flag_kth(mFlag3))
             {
                 if (! aSom->flag_kth(mFlag2))
                 {
                     SetCurNeigh2(aSom);
                 }
                 TestInitRot(anA,aLK3[aK3]);
             }
         }
    }
}


void cAppli_NewSolGolInit::FinishNeighTriplet()
{
    for (int aK3=0 ; aK3<int(mVCur3.size()) ; aK3++)
    {
        mVCur3[aK3]->flag_set_kth_false(mFlag3);
    }
    for (int aK2=0 ; aK2<int(mVCur2.size()) ; aK2++)
    {
        mVCur2[aK2]->flag_set_kth_false(mFlag2);
        mVCur2[aK2]->attr().ReInit();
    }
}


void   cAppli_NewSolGolInit::CreateArc(tSomNSI * aS1,tSomNSI * aS2,cNOSolIn_Triplet * aTripl,int aK1,int aK2,int aK3)
{
     tArcNSI * anArc = mGr.arc_s1s2(*aS1,*aS2);
     if (anArc==0)
     {
         cNOSolIn_AttrASym * anAttrSym = new cNOSolIn_AttrASym;
         cNOSolIn_AttrArc anAttr12(anAttrSym,aS1<aS2);
         cNOSolIn_AttrArc anAttr21(anAttrSym,aS1>aS2);
         anArc = &(mGr.add_arc(*aS1,*aS2,anAttr12,anAttr21));
         mNbArc ++;
     }
     anArc->attr().ASym()->AddTriplet(aTripl,aK1,aK2,aK3);
     aTripl->SetArc(aK3,anArc);

     // return anArc;
}

void cAppli_NewSolGolInit::EstimRotsArcsInit()
{
    for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
    {
          tSomNSI * aS1 = &(*anItS);
          for (tItANSI anItA=aS1->begin(mSubAll) ; anItA.go_on(); anItA++)
          {
                if ((*anItA).attr().IsOrASym())
                {
                    InitRotOfArc(&(*anItA),false);
                }
          }
    }
}

void cAppli_NewSolGolInit::FilterTripletValide(std::vector<cLinkTripl > & aV)
{
    std::vector<cLinkTripl > aNewV ;
    for (int aKL=0 ; aKL<int(aV.size()) ; aKL++)
    {
        if (TripletIsValide(aV[aKL].m3))
        {
           aNewV.push_back(aV[aKL]);
        }
    }

    aV = aNewV;
}

void cAppli_NewSolGolInit::FilterTripletValide()
{
    std::vector<cNOSolIn_Triplet*> aNewV3;

    for (int aK=0 ; aK<int(mV3.size()) ; aK++)
       if (TripletIsValide(mV3[aK]))
          aNewV3.push_back(mV3[aK]);

   // std::cout << "FilterTripletValide " << mV3.size() << " => " << aNewV3.size() << "\n";

   mV3 = aNewV3;

   // We must do the same stuff with the link
   for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
   {
          tSomNSI * aS1 = &(*anItS);
          FilterTripletValide(aS1->attr().Lnk3());
          for (tItANSI anItA=aS1->begin(mSubAll) ; anItA.go_on(); anItA++)
          {
                if ((*anItA).attr().IsOrASym())
                {
                    FilterTripletValide((*anItA).attr().ASym()->Lnk3());
                }
          }
/*
*/
   }
}



void  cAppli_NewSolGolInit::EstimCoheTriplet()
{
    std::vector<Pt2df> aVCost3A;
    for (int aK=0  ; aK<int(mV3.size()) ; aK++)
    {
         mV3[aK]->CalcCoherFromArcs(false);  // Calc coherence
         // Push value to have a median weighted by numlbre of 3 points
         aVCost3A.push_back(Pt2df(mV3[aK]->CostArc(),mV3[aK]->Nb3() ));  
    }

    mMedTripletCostA = DefMedianPond(0,-1,aVCost3A,0);
    mSeuilCostArc = CstSeuilMedianArc + MulSeuilMedianArc * mMedTripletCostA;


    // double aCostNorMax = mSeuilCostArc / (mSeuilCostArc+mMedTripletCostA);

    cNO_CmpPtrTriplOnCost aCmp;
    std::sort(mV3.begin(),mV3.end(),aCmp);

    for (int aK=0  ; aK<int(mV3.size()) ; aK++)
    {
        cNOSolIn_Triplet & a3 = *(mV3[aK]);
        double aCost = a3.CostArc() ;
        aCost = aCost/ (aCost+mMedTripletCostA);
        // aCost = aCost / aCostNorMax;
        a3.GainArc() = ElMax(0.0,1-aCost);

        // double aR = 
    }


    if (0)
    {
       int aNb= 20;
       for (int aK=0 ; aK<= aNb ; aK++)
       {
            int aKS = (aK * (int)(mV3.size() - 1)) / aNb;
             mV3[aKS]->Show(ToString(aKS)) ;
       }
       std::cout << "COST ;  S3A= " << mSeuilCostArc  << " M3A " << mMedTripletCostA << "\n";
    }
}


/***********************************************************
    Evaluation statistique des coherence moyennes         
    Pour chaque triplet , et chacun de ces 3 arcs on  calcule une valeur,
on en prend la mediane. Avec les grand jeu de donnees on en prend que 100000 au max
***********************************************************/

void  cAppli_NewSolGolInit::EstimCoherenceMed()
{
    // Calcul du nombre de couples de triplets ayant des arcs commun
  
    int aNbTT = 0; // Number of pair of triplet
    for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
    {
          tSomNSI * aS1 = &(*anItS);
          for (tItANSI anItA=aS1->begin(mSubAll) ; anItA.go_on(); anItA++)
          {
                if ((*anItA).attr().IsOrASym())
                {
                    int aNbT = (int)(*anItA).attr().ASym()->Lnk3().size();
                    aNbTT += (aNbT*(aNbT-1)) / 2;  // number of subset with 2 elements
                    (*anItA).attr().ASym()->PostInit(false); // compute average B/H
                }
          }
    }
    // std::cout << "NBTTT " << aNbTT  <<  " => " << NbMaxATT << "\n";

    // cRandNParmiQ aSel(aNbTT,ElMin(aNbTT,NbMaxATT));
    cRandNParmiQ aSel(ElMin(aNbTT,NbMaxATT),aNbTT);
    // std::vector<float> aVC;
    std::vector<Pt2df> aVPAB;
    std::vector<Pt2df> aVP12;
    for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
    {
          tSomNSI * aS1 = &(*anItS);
          for (tItANSI anItA=aS1->begin(mSubAll) ; anItA.go_on(); anItA++)
          {
                tArcNSI & anArc = (*anItA);
                tSomNSI * aS2 = &(anArc.s2());
                if (aS1 < aS2)
                {
                    std::vector<cLinkTripl> & aVL = anArc.attr().ASym()->Lnk3();
                    for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
                    {
                        cNOSolIn_Triplet * aTri1 = aVL[aK1].m3;
                        for (int aK2=aK1+1 ; aK2<int(aVL.size()) ; aK2++)
                        {
                            if (aSel.GetNext())
                            {
                                cNOSolIn_Triplet * aTri2 = aVL[aK2].m3;
                                double  aDCAB = DistCoherenceAtoB(&anArc,aTri1,aTri2);

                                // aVC.push_back(aDC);
                                int aNb = ElMin(aTri1->Nb3(),aTri2->Nb3());
                                aVPAB.push_back(Pt2df(aDCAB,aNb));


                                double  aDC12 = DistCoherence1to2(&anArc,aTri1,aTri2);
                                aVP12.push_back(Pt2df(aDC12,aNb));
// tArcNSI * anArc,cNOSolIn_Triplet * aTriA,cNOSolIn_Triplet * aTriB)
                                 aNbTT--;
                            }
                        }
                    }
                }
          }
    }
    mCoherMedAB =  DefMedianPond(0,-1,aVPAB,0);
    int aKMed;
    mCoherMed12 =  DefMedianPond(0,-1,aVP12,&aKMed);


    if (0)
    {
       for (int aK=0 ; aK<100 ; aK++)
       {
            int aKH = (int)((aVP12.size() * aK) /100);
            std::cout << " Med " << aK << " = " << aVP12[aKH] << "\n";
       }
       std::cout << "MEDIAN=" << mCoherMed12  << " Prop=" << aKMed/double(aVP12.size()) << "\n";
    }
}

void  cAppli_NewSolGolInit::InitRotOfArc(tArcNSI * anArc,bool Test)
{
   ELISE_ASSERT(anArc->attr().IsOrASym(),"Arc orient in cAppli_NewSolGolInit::InitRotOfArc");
   std::vector<cLinkTripl> & aVL = anArc->attr().ASym()->Lnk3();

   // Impression de la matrice
   if (Test)
   {
       for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
       {
           std::cout << "IRA " << aVL[aK1].S3()->attr().Im()->Name() << " ";
           for (int aK2=0 ; aK2<int(aVL.size()) ; aK2++)
           {
                 double  aDC = DistCoherenceAtoB(anArc,aVL[aK1].m3,aVL[aK2].m3);
                 printf("%5d " ,round_ni(aDC*1000));
           }
           std::cout << "\n";
       }
   }

   // Fill the kernel structure, see  include/general/bitm.h
   // Comment in file
   // "Robust computation of average element as the one minimizing the sum distance" 

   mCompKG.SetN((int)aVL.size());  // Set the size of kernel

   // Fill the matrix of cost, by parsing all pair of triplet
   for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
   {
       cNOSolIn_Triplet * aTri1 = aVL[aK1].m3;
       for (int aK2=aK1+1 ; aK2<int(aVL.size()) ; aK2++)
       {
           cNOSolIn_Triplet * aTri2 = aVL[aK2].m3;
           double  aDC = DistCoherenceAtoB(anArc,aTri1,aTri2);
           // We transformat the cost in a normalized value between 0 and 1 using formula :
           //  aDC / (aDC+mCoherMed*FactAttCohMed);
           double  aDatt =   CoutAttenueTetaMax(aDC,mCoherMedAB*FactAttCohMed);  
           // Weight is proportional to number of triple points
           mCompKG.AddCost(aK1,aK2,aTri2->Nb3(),aTri1->Nb3(),aDatt);
       }
   }

   // Store vector of relative rotation to compute them once an only once
   std::vector<ElRotation3D> aVR;
   for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
   {
       aVR.push_back(RotationC2toC1(anArc,aVL[aK1].m3));
   }
   
   // Get a robust value by extracting the rotation mimizing
   // GetKernelGen do some complicated stuff, I will document later, but basically it
   // extract the summit with mimimal somm o weight (with some "robust" filterign ?)
   int aKK = mCompKG.GetKernelGen();  
   double aBSurH0 =  aVL[aKK].m3->BOnH();
   ElRotation3D aR0 = aVR[aKK];
   double aD0 = euclid(aR0.tr());
   double aSomD =0.0;

   //  Now we have a initial guess of the relative rotation by some median like value
   // we can try to imrpove it by some weighted average
   
   int aNbIter = 4;
   for (int aKIter = 0 ; aKIter<(aNbIter) ; aKIter++)
   {
        ElMatrix<double> aSomMat(3,3,0.0);
        double aSomPds = 0.0;
        Pt3dr   aSomTr (0,0,0);
        aSomD = 0;
        // Here begin the somm of weighted avareage
        for (int aK=0 ; aK<int(aVL.size()) ; aK++)
        {
             double aD =  DistanceRot(aR0,aVR[aK],aBSurH0);
             aSomD += aD;

             double aPds = 0;
             if (aD < 6 * mCoherMed12) // Some threshold on "big" outlayr
             {
                   aPds = 1 /(1 + ElSquare(aD/(2*mCoherMed12))); // "Classiq MicMac" formula
                   double aPdsPop = aPds * aVL[aK].m3->Nb3();  // Weight prop to triple points
                   aSomPds += aPdsPop;
                   aSomTr  = aSomTr  + (vunit(aVR[aK].tr()) * aD0) * aPdsPop;
                   aSomMat = aSomMat + aVR[aK].Mat() * aPdsPop;
             } 
             if (Test && (aKIter==(aNbIter-1)))
                 std::cout << " IMm=" << aVL[aK].S3()->attr().Im()->Name() << " Pds=" << aPds << " D=" << aD << "\n";
        }
        if (aSomPds>0)
           aR0 = ElRotation3D(aSomTr/aSomPds,aSomMat*(1/aSomPds),true);
   }


   if (Test)
   {
      aSomD /= aVL.size();
      std::cout <<  "KERNEL " <<   aVL[aKK].S3()->attr().Im()->Name() <<  " D=" << aSomD << " R/M=" << aSomD/ mCoherMed12  << " BH0=" <<aBSurH0 << "\n";
   }

   anArc->attr().ASym()->EstimC2toC1() = aR0;
}



static cNO_CmpSomByGainBy3 TheCmp3;

cAppli_NewSolGolInit::cAppli_NewSolGolInit(int argc, char ** argv) :
    mTest       (true),
    mSimul      (false),
    mWithOriTest(false),
    mIterLocEstimRot (true),
    mFlag3      (mGr.alloc_flag_som()),
    mFlag2      (mGr.alloc_flag_som()),
    mTestTrip   (0),
    mTestTrip2  (0),
    mTestS1     (0),
    mTestS2     (0),
    mTestArc    (0),
    mNbSom      (0),
    mNbArc      (0),
    mNbTrip     (0),
    mFlag3Alive (mAllocFlag3.flag_alloc()),
    mFlag3CC    (mAllocFlag3.flag_alloc()),
    mHeapSom    (TheCmp3),
    mLastPdsMedRemoy  (0.0),
    mActiveRemoy      (true),
    mNbIterLast       (20)
{
   std::string aNameT1;
   std::string aNameT2;
   std::string aNameT3;
   std::string aNameT4;
   bool        aModeBin = true;


   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mFullPat,"Pattern"),
        LArgMain() 
                   << EAM(mTest,"Test",true,"Test for tuning",eSAM_IsBool)
                   << EAM(aNameT1,"Test1",true,"Name of first test image",eSAM_IsBool)
                   << EAM(aNameT2,"Test2",true,"Name of second test image",eSAM_IsBool)
                   << EAM(aNameT3,"Test3",true,"Name of third test image",eSAM_IsBool)
                   << EAM(aNameT4,"Test4",true,"Name of fourth test image",eSAM_IsBool)
                   << EAM(mSimul,"Simul",true,"Simulation of perfect triplet (tuning)",eSAM_IsBool)
                   << EAM(mOriTest,"OriTest",true,"Test on known solution (tuning)",eSAM_IsBool)
                   << EAM(aModeBin,"Bin",true,"Binaries file, def = true",eSAM_IsBool)
                   << EAM(mIterLocEstimRot,"ILER",true,"Iter Estim Loc, Def=true, tuning purpose",eSAM_IsBool)
                   << EAM(mActiveRemoy,"AR",true,"Active Remoy, Def=true, tuning purpose",eSAM_IsBool)
                   << EAM(mNbIterLast,"NbIterLast",true,"Nb Iter in last step",eSAM_IsBool)
                   << ArgCMA()
   );

   cTplTriplet<std::string> aKTest1(aNameT1,aNameT2,aNameT3);
   cTplTriplet<std::string> aKTest2(aNameT1,aNameT2,aNameT4);

   mHasInOri = (EAMIsInit(&mInOri) && (mInOri!=""));

   mEASF.Init(mFullPat);
   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,mEASF.mDir,mNameOriCalib,"dat",OriOut());
   const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

   if (EAMIsInit(&mOriTest))
   {
       mWithOriTest = true;
       StdCorrecNameOrient(mOriTest,mNM->Dir());
   }

   tSomNSI * mTestS1=0;
   tSomNSI * mTestS2=0;
   // tSomNSI * aTestS3=0;

   for (int aKIm=0 ; aKIm <int(aVIm->size()) ; aKIm++)
   {
       const std::string & aName = (*aVIm)[aKIm];
       tSomNSI & aSom = mGr.new_som(cNOSolIn_AttrSom(aName,*this));
       mMapS[aName] = & aSom;
       mNbSom++;
       if (mWithOriTest)
       {
           CamStenope * aCam = mNM->CamOriOfName(aName,mOriTest);
           aSom.attr().TestRot() = aCam->Orient().inv();
       }
       else if (mSimul)
       {
           ElMatrix<double> aR =  ElMatrix<double>::Rotation(aKIm+0.5,aKIm*10,aKIm*100);
           Pt3dr aTr(cos(aKIm*0.7),sin(aKIm*2.0),sin(4.0+aKIm*10.7));
           if (aName==aKTest1.mV0) 
           {
                aR= ElMatrix<double>::Rotation(0,0,0);
                aTr = Pt3dr(0,0,0);
           }
           aSom.attr().TestRot() = ElRotation3D(aTr,aR,true);
       }
       if (aName==aNameT1) mTestS1 = &(aSom);
       if (aName==aNameT2) mTestS2 = &(aSom);
       // if (aName==aNameT3) aTestS3 = &(aSom);
   }


    cXml_TopoTriplet aXml3 =  StdGetFromSI(mNM->NameTopoTriplet(true),Xml_TopoTriplet);

    for
    (
         std::list<cXml_OneTriplet>::const_iterator it3=aXml3.Triplets().begin() ;
         it3 !=aXml3.Triplets().end() ;
         it3++
    )
    {
            tSomNSI * aS1 = mMapS[it3->Name1()];
            tSomNSI * aS2 = mMapS[it3->Name2()];
            tSomNSI * aS3 = mMapS[it3->Name3()];

            ELISE_ASSERT(it3->Name1()<it3->Name2(),"Incogeherence cAppli_NewSolGolInit\n");
            ELISE_ASSERT(it3->Name2()<it3->Name3(),"Incogeherence cAppli_NewSolGolInit\n");


            cTplTriplet<std::string> anIdTri(it3->Name1(),it3->Name2(),it3->Name3());

            if ((aKTest1==anIdTri) || (aKTest2==anIdTri))
            {
                // std::cout << "WWWWWW " << it3->Name1() << "==" << it3->Name2() << "==" << it3->Name3() << "\n";
            }



            if (aS1 && aS2 && aS3)
            {
                 mNbTrip++;

                 std::string  aN3 = mNM->NameOriOptimTriplet
                                    (
                                        // mQuick,
                                        aModeBin,  // ModeBin
                                        aS1->attr().Im(),
                                        aS2->attr().Im(),
                                        aS3->attr().Im()
                                    );
                 cXml_Ori3ImInit aXml3Ori = StdGetFromSI(aN3,Xml_Ori3ImInit);

                 if (mSimul)
                 {
                     ElRotation3D aR1 = aS1->attr().TestRot();
                     ElRotation3D aR2 = aS2->attr().TestRot();
                     ElRotation3D aR3 = aS3->attr().TestRot();
                     ElRotation3D aR2On1 =  aR1.inv() * aR2;
                     ElRotation3D aR3On1 =  aR1.inv() * aR3;
                     double aScale = 1.1 + cos((double)mNbTrip);
                     aR2On1.tr() =  aR2On1.tr() * aScale;
                     aR3On1.tr() =  aR3On1.tr() * aScale;
                     aXml3Ori.Ori2On1() =  El2Xml(aR2On1);
                     aXml3Ori.Ori3On1() =  El2Xml(aR3On1);
                 }


                 cNOSolIn_Triplet * aTriplet = new cNOSolIn_Triplet(this,aS1,aS2,aS3,aXml3Ori);
                 mV3.push_back(aTriplet);

                 ///  ADD-SOM-TRIPLET
                 aS1->attr().AddTriplet(aTriplet,1,2,0);
                 aS2->attr().AddTriplet(aTriplet,0,2,1);
                 aS3->attr().AddTriplet(aTriplet,0,1,2);

                 ///  ADD-EDGE-TRIPLET
                 CreateArc(aS1,aS2,aTriplet,0,1,2);
                 CreateArc(aS2,aS3,aTriplet,1,2,0);
                 CreateArc(aS3,aS1,aTriplet,2,0,1);

                 aTriplet->CheckArcsSom();

                 if (aKTest1==anIdTri)
                 {
                     mTestTrip = aTriplet;
                 }
                 if (aKTest2==anIdTri)
                 {
                     mTestTrip2 = aTriplet;
                 }
            }
    }
    std::cout << "LOADED GRAPH " << mChrono.uval() << "\n";



    EstimCoherenceMed();
    // std::cout << "COHERENCE MED, DAB  " << mCoherMedAB    << " D12 " << mCoherMed12   << "\n";




    if (mTestS1 && mTestS2)
    {
          mTestArc = mGr.arc_s1s2(*mTestS1,*mTestS2);
    }
    if ( 1 && mTestArc)
    {
          InitRotOfArc(mTestArc,true);
          mTestArc->attr().ASym()->PostInit(true);

    }
    EstimRotsArcsInit();
    // std::cout << "Rots init \n";


    EstimCoheTriplet();
    // std::cout << "Triplets Init \n";



    if (mWithOriTest || mSimul)
    {
         // getchar();
         int aNbSupCoh = 0;
         int aNbInv = 0;

         for (int aKV=0 ; aKV<int(mV3.size()) ; aKV++)
         {
             cNOSolIn_Triplet * aTri = mV3[aKV];
             double aC = aTri->CoherTest();
             if (! TripletIsValide(aTri)) aNbInv++;

             if (aC>0.1)
             {
                  aTri->Show("TC ");
                  std::cout << "COHER " << aC << " " << TripletIsValide(aTri) << "\n";
                  aNbSupCoh ++;
             }
             //  std::cout << "COHER " << aC << "\n";
         }

         std::cout << "COHERENCE MED, DAB  " << mCoherMedAB    
                   << " D12 " << mCoherMed12   
                   << " TRI " << mMedTripletCostA 
                   << " SEUIL " << mSeuilCostArc   << "\n";

         std::cout << "PropInc " << (aNbSupCoh)/double(mV3.size()) << " PropInv " << aNbInv/double(mV3.size()) << "\n";
         getchar();
    }



    FilterTripletValide();
    // std::cout << "Filer done \n";




    if (mTestTrip)
    {
        std::cout << "mTestTrip " << mTestTrip << "\n";

        mTestTrip->CalcCoherFromArcs(true);
/*
        std::cout <<  "GLOB NbS = " <<  mNbSom 
                 << " NbA " << mNbArc  << ",Da=" <<   (2.0 *mNbArc)  / (mNbSom*mNbSom) 

                 << " Nb3 " << mNbTrip  << ",D3=" << (3.0 *mNbTrip)  / (mNbArc*mNbSom)  << "\n";

        // cAppli_NewSolGolInit::SetNeighTriplet
        SetNeighTriplet(mTestTrip);
        std::cout << "NbIn Neih " <<  mVCur2.size() << "\n";
        for (int aK=0 ; aK< int(mVCur2.size()) ; aK++)
        {
            std::cout << "  Neigh " << mVCur2[aK]->attr().Im()->Name() ;
            if (  mVCur2[aK]->flag_kth(mFlag3)) std::cout << " *** ";
            std::cout << "\n";
        }
*/
    }

    NumeroteCC();

    std::cout << "Begin calc orient\n";
   
    CalculOrient();

    std::cout << " Done Calc , T= " << mChrono.uval() << "\n";
}


void cAppli_NewSolGolInit::Save()
{

    double aSomDistMin = 1e30;
    Pt3dr aDirKMin(0,0,-1);
    Pt3dr aCentre(0,0,0);
    double aNbCentre=0;
    for (tItSNSI anItS1=mGr.begin(mSubAll) ; anItS1.go_on(); anItS1++)
    {
        // C'est du Cam 2 Monde 
        const ElRotation3D & aR1 = (*anItS1).attr().CurRot();
        aCentre =  aCentre+ aR1.ImAff(Pt3dr(0,0,0));
        aNbCentre++;
        Pt3dr aDirK1 =  aR1.ImVect(Pt3dr(0,0,-1));

        double aSomDist = 0;
        double aNbDist = 0;

        for (tItSNSI anItS2=mGr.begin(mSubAll) ; anItS2.go_on(); anItS2++)
        {
            const ElRotation3D & aR2 = (*anItS2).attr().CurRot();
            Pt3dr aDirK2 =  aR2.ImVect(Pt3dr(0,0,-1));
            aSomDist += ElSquare(euclid(aDirK1-aDirK2));
            aNbDist++;
        }
        aSomDist /= aNbDist;
        if (aSomDist< aSomDistMin)
        {
           aSomDistMin = aSomDist;
           aDirKMin = aDirK1;
        }
        // std::cout << "Moy Dist " << aSomDist << " K=" << aDirK1 << " C=" <<  aR1.ImAff(Pt3dr(0,0,0)) << "\n";
    }
    aCentre = aCentre / aNbCentre;
    Pt3dr aDirIMin,aDirJMin;
    MakeRONWith1Vect(aDirKMin,aDirIMin,aDirJMin);

    ElMatrix<double> aMat = MatFromCol(aDirIMin,aDirJMin,aDirKMin);
    aMat = NearestRotation(aMat);
    // std::cout <<"DMIIIIIN " << aSomDistMin << " " << aDirKMin << " " << aCentre <<  " DET=" << aMat.Det() << "\n";
    ElRotation3D  aNew2Old(aCentre,aMat,true);
    if (mHasInOri)
        aNew2Old = ElRotation3D::Id;

    Pt3dr aNewC(0,0,0);

    for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
    {
        Pt3dr aPMed =  (*anItS).attr().SomPMedReM() ;
        cNewO_OneIm * anI = (*anItS).attr().Im();
        std::string aNameIm = anI->Name();
        CamStenope * aCS = anI->CS();
        ElRotation3D aROld2Cam = (*anItS).attr().CurRot().inv();

        aCS->SetOrientation(aROld2Cam);
        if (0)   // Verif avec iii
           std::cout << "hhhhhh " << aCS->R3toF2(aPMed) << "\n";


        ElRotation3D aRNew2Cam =  aROld2Cam * aNew2Old;
        aCS->SetOrientation(aRNew2Cam);

        aNewC  = aNewC + aCS->VraiOpticalCenter() ;
        // std::cout << "JJJJJJJJJJJJJJJJ " << aCS->DirK() << aCS->VraiOpticalCenter() << "\n";

        cOrientationConique anOC =  aCS->StdExportCalibGlob();
        anOC.Interne().SetNoInit();
        anOC.FileInterne().SetVal(mNM->ICNM()->StdNameCalib(mNM->OriOut(),aNameIm));


        aPMed = aNew2Old.ImRecAff(aPMed);
        if (0)
           std::cout << "iiiiii " << aCS->R3toF2(aPMed) << "\n";
        double aD = euclid(aPMed-(*anItS).attr().CurRot().tr());

        anOC.Externe().AltiSol().SetVal(aPMed.z);
        anOC.Externe().Profondeur().SetVal(aD);
        
        //std::string aNameOri = mNM->ICNM()->Assoc1To1("NKS-Assoc-Im2Orient@-"+mNM->OriOut(),aNameIm,true);
        std::string aNameOri = mNM->NameOriOut(aNameIm);
      
        MakeFileXML(anOC,aNameOri);
    }
    // std::cout << " NEWWWCCCC  " << aNewC  << "\n";
    std::cout << " Done Save , T= " << mChrono.uval() << "\n";
}




int CPP_NewSolGolInit_main(int argc, char ** argv)
{
    cAppli_NewSolGolInit anAppli(argc,argv);
    anAppli.Save();
    return EXIT_SUCCESS;

/*
if (0)  // Test swap
{
    for (int aCpt=1 ; true; aCpt++)
    {
        new cAppli_NewSolGolInit (argc,argv);

        std::cout <<  "cAppli_NewSolGolInit  " << aCpt << "\n";
        if ((aCpt%10)==0) getchar();
    }
}
*/


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
