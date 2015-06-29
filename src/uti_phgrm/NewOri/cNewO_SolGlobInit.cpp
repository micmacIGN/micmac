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
class cComputecKernelGraph
{
    public :
         cComputecKernelGraph();
         void SetN(int aN);
         void AddCost(int aK1,int aK2,double aPds1,double aPds2,double aDist);

         int GetKernel();

         // void DupIn(const cComputecKernelGraph & );
    private :
         Im2D_REAL8      mPdsArc;
         Im2D_REAL8      mCostArc;
         Im1D_REAL8      mPdsSom;
         Im1D_REAL8      mCostSom;

         double **       mDPdsArc;
         double **       mDCostArc;
         double *        mDPdsSom;
         double *        mDCostSom;
         int             mNb;
};


cComputecKernelGraph::cComputecKernelGraph() :
    mPdsArc  (1,1),
    mCostArc (1,1),
    mPdsSom  (1),
    mCostSom (1)
{
}

void cComputecKernelGraph::AddCost(int aK1,int aK2,double aPds1,double aPds2,double aDist)
{
     mDPdsArc[aK1][aK2]  += aPds1;
     mDPdsArc[aK2][aK1]  += aPds2;
     mDPdsSom[aK1]       += aPds1;
     mDPdsSom[aK2]       += aPds2;

     mDCostArc[aK1][aK2] += aDist*aPds1;
     mDCostArc[aK2][aK1] += aDist*aPds2;
     mDCostSom[aK1]      += aDist*aPds1;
     mDCostSom[aK2]      += aDist*aPds2;
}

int  cComputecKernelGraph::GetKernel()
{
    int aKBest=-1;
    for (int anIter=2 ; anIter < mNb ; anIter++)
    {
         int aKWorst=-1;
         double aBestCost = 1e9;
         double aWortCost = -1e9;
         for (int aK=0 ; aK<mNb ; aK++)
         {
             if (mDPdsSom[aK] >0)
             {
                double aCost =  mDCostSom[aK]  / mDPdsSom[aK];
                if (aCost>aWortCost)
                {
                     aWortCost = aCost;
                     aKWorst = aK;
                }
                if (aCost< aBestCost)
                {
                     aBestCost = aCost;
                     aKBest = aK;
                }
             }
         }
         ELISE_ASSERT((aKWorst>=0) && (aKBest>=0),"cComputecKernelGraph::GetKernel");
         for (int aK=0 ; aK<mNb ; aK++)
         {
              mDPdsSom[aK] -=  mDPdsArc [aK][aKWorst];
              mDCostSom[aK] -= mDCostArc[aK][aKWorst];
         }
         mDPdsSom[aKWorst] = -1;
         
   
    }
    return aKBest;
}


void cComputecKernelGraph::SetN(int aN)
{
    ELISE_ASSERT(aN>=3,"cComputecKernelGraph::SetN");
    mNb = aN;
    mCostArc.Resize(Pt2di(aN,aN));
    mPdsArc.Resize(Pt2di(aN,aN));
    mPdsSom.Resize(aN);
    mCostSom.Resize(aN);

    mDPdsArc = mPdsArc.data();
    mDCostArc = mCostArc.data();
    mDPdsSom = mPdsSom.data();
    mDCostSom = mCostSom.data();

    for (int anX=0 ; anX < aN ; anX++)
    {
       mDPdsSom[anX] = 0.0;
       mDCostSom[anX] = 0.0;
       for (int anY=0 ; anY < aN ; anY++)
       {
           mDCostArc[anY][anX] = 0.0;
           mDPdsArc[anY][anX] = 0.0;
       }
    }
}




class cLinkTripl;
class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cAppli_NewSolGolInit;
class cNOSolIn_Triplet;

typedef  ElSom<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tSomNSI;
typedef  ElArc<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tArcNSI;
typedef  ElSomIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItSNSI;
typedef  ElArcIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItANSI;
typedef  ElGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>      tGrNSI;
typedef  ElSubGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>   tSubGrNSI;


class cLinkTripl
{
     public :
         cLinkTripl(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3) :
            m3   (aTrip),
            mK1  (aK1),
            mK2  (aK2),
            mK3  (aK3)
         {
         }

         cNOSolIn_Triplet  *  m3;
         U_INT1               mK1;
         U_INT1               mK2;
         U_INT1               mK3;
         tSomNSI *            S1() const;
         tSomNSI *            S2() const;
         tSomNSI *            S3() const;
};



class cNOSolIn_AttrSom
{
     public :
         cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli);
         cNOSolIn_AttrSom() :
             mCurRot (ElRotation3D::Id),
             mTestRot (ElRotation3D::Id)
         {}

         void AddTriplet(cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
         // std::vector<cNOSolIn_Triplet *> & V3() {return mV3;}
         cNewO_OneIm * Im() {return mIm;}
         void ReInit();
         ElRotation3D & CurRot() {return mCurRot;}
         ElRotation3D & TestRot() {return mTestRot;}
     private :
         std::string                      mName;
         cAppli_NewSolGolInit *           mAppli;
         cNewO_OneIm *                    mIm;
         std::vector<cLinkTripl >         mLnk3;
         double                           mCurCostMin;
         ElRotation3D                     mCurRot;
         ElRotation3D                     mTestRot;
};


class cNOSolIn_AttrASym
{
     public :
         void AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3);
         std::vector<cLinkTripl> & Lnk3() {return mLnk3;}
     private :
         std::vector<cLinkTripl> mLnk3;
           
};
class cNOSolIn_AttrArc
{
     public :
           cNOSolIn_AttrArc(cNOSolIn_AttrASym *);
           cNOSolIn_AttrASym * ASym() {return mASym;}
     private :
           cNOSolIn_AttrASym * mASym;
};


class cNOSolIn_Triplet
{
      public :
          cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit &);
          void SetArc(int aK,tArcNSI *);
          tSomNSI * KSom(int aK) const {return mSoms[aK];}
          tArcNSI * KArc(int aK) const {return mArcs[aK];}

          void InitRot3Som();
          const ElRotation3D & RotOfSom(tSomNSI * aS) const
          {
                if (aS==mSoms[0]) return ElRotation3D::Id;
                if (aS==mSoms[1]) return mR2on1;
                if (aS==mSoms[2]) return mR3on1;
                ELISE_ASSERT(false," RotOfSom");
                return ElRotation3D::Id;
          }
          double BOnH() const {return mBOnH;}
          int  Nb3() const {return mNb3;}

          


      private :
          tSomNSI *     mSoms[3];
          tArcNSI *     mArcs[3];
          ElRotation3D  mR2on1;
          ElRotation3D  mR3on1;
          double        mBOnH;
          int           mNb3;
   // Gere les triplets qui vont etre desactives
          bool          mAlive;
};


class cAppli_NewSolGolInit
{
    public :
        cAppli_NewSolGolInit(int , char **);
        cNewO_NameManager & NM() {return *mNM;}

    private :
        void FinishNeighTriplet();
 
        void TestInitRot(tArcNSI * anArc,const cLinkTripl & aLnk);
        void TestOneTriplet(cNOSolIn_Triplet *);
        void SetNeighTriplet(cNOSolIn_Triplet *);

        void SetCurNeigh3(tSomNSI *);
        void SetCurNeigh2(tSomNSI *);

        void                 CreateArc(tSomNSI *,tSomNSI *,cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
        void EstimCoherenceMed();
        void    InitRotOfArc(tArcNSI * anArc,bool Test);



        std::string          mFullPat;
        std::string          mOriCalib;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;
        bool                 mQuick;
        bool                 mTest;
        bool                 mSimul;
 
        tGrNSI               mGr;
        tSubGrNSI            mSubAll;
        std::map<std::string,tSomNSI *> mMapS;
        cComputecKernelGraph             mCompKG;

// Variables temporaires pour charger un triplet 
        std::vector<tSomNSI *>  mVCur3;  // Tripelt courrant
        std::vector<tSomNSI *>  mVCur2;  // Adjcent au triplet courant
        int                     mFlag3;
        int                     mFlag2;
        cNOSolIn_Triplet *      mTestTrip;
        cNOSolIn_Triplet *      mTestTrip2;
        tSomNSI *               mTestS1;
        tSomNSI *               mTestS2;
        tArcNSI *               mTestArc;
        int                     mNbSom;
        int                     mNbArc;
        int                     mNbTrip;
        double                  mCoherMedAB;       
        double                  mCoherMed12;       
 
};

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrSom                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrSom::cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli) :
   mName        (aName),
   mAppli       (&anAppli),
   mIm          (new cNewO_OneIm(mAppli->NM(),mName)),
   mCurRot      (ElRotation3D::Id),
   mTestRot     (ElRotation3D::Id)
{
   ReInit();
}

void cNOSolIn_AttrSom::ReInit()
{
    mCurCostMin = 1e20;
}

void cNOSolIn_AttrSom::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}


/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_Triplet                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_Triplet::cNOSolIn_Triplet(tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit & aTrip) :
    mR2on1 (Xml2El(aTrip.Ori2On1())),
    mR3on1 (Xml2El(aTrip.Ori3On1())),
    mBOnH  (aTrip.BSurH()),
    mNb3   (aTrip.NbTriplet()),
    mAlive (true)
{
   mSoms[0] = aS1;
   mSoms[1] = aS2;
   mSoms[2] = aS3;
}

void cNOSolIn_Triplet::SetArc(int aK,tArcNSI * anArc)
{
   mArcs[aK] = anArc;
}

void cNOSolIn_Triplet::InitRot3Som()
{
     mSoms[0]->attr().CurRot() = ElRotation3D::Id;
     mSoms[1]->attr().CurRot() = mR2on1;
     mSoms[2]->attr().CurRot() = mR3on1;
}



/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrASym                                       */
/*                                                                         */
/***************************************************************************/


void  cNOSolIn_AttrASym::AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3)
{
    mLnk3.push_back(cLinkTripl(aTrip,aK1,aK2,aK3));
}

tSomNSI * cLinkTripl::S1() const {return  m3->KSom(mK1);}
tSomNSI * cLinkTripl::S2() const {return  m3->KSom(mK2);}
tSomNSI * cLinkTripl::S3() const {return  m3->KSom(mK3);}

/***************************************************************************/
/*                                                                         */
/*                 cNOSolIn_AttrArc                                        */
/*                                                                         */
/***************************************************************************/

cNOSolIn_AttrArc::cNOSolIn_AttrArc(cNOSolIn_AttrASym * anASym) :
   mASym (anASym)
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


ElRotation3D RotationC2toC1(tArcNSI * anArc,cNOSolIn_Triplet * aTri)
{
   return aTri->RotOfSom(&anArc->s1()).inv() *  aTri->RotOfSom(&anArc->s2());
}



double DistCoherence1to2 (tArcNSI * anArc,cNOSolIn_Triplet * aTriA,cNOSolIn_Triplet * aTriB)
{
     return DistanceRot
            (
                RotationC2toC1(anArc,aTriA),  
                RotationC2toC1(anArc,aTriB),
                MoyHarmonik(aTriA->BOnH(),aTriB->BOnH())
            );
}


double DistCoherenceAtoB(tArcNSI * anArc,cNOSolIn_Triplet * aTriA,cNOSolIn_Triplet * aTriB)
{
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
         cNOSolIn_AttrArc anAttr12(anAttrSym);
         cNOSolIn_AttrArc anAttr21(anAttrSym);
         anArc = &(mGr.add_arc(*aS1,*aS2,anAttr12,anAttr21));
         mNbArc ++;
     }
     anArc->attr().ASym()->AddTriplet(aTripl,aK1,aK2,aK3);
     aTripl->SetArc(aK3,anArc);

     // return anArc;
}

void  cAppli_NewSolGolInit::EstimCoherenceMed()
{
    // Calcul du nombre de couples de triplets ayant des arcs commun
  
    int aNbTT = 0;
    for (tItSNSI anItS=mGr.begin(mSubAll) ; anItS.go_on(); anItS++)
    {
          tSomNSI * aS1 = &(*anItS);
          for (tItANSI anItA=aS1->begin(mSubAll) ; anItA.go_on(); anItA++)
          {
                tSomNSI * aS2 = &((*anItA).s2());
                if (aS1 < aS2)
                {
                    int aNbT = (*anItA).attr().ASym()->Lnk3().size();
                     aNbTT += (aNbT*(aNbT-1)) / 2;
                }
          }
    }
    // std::cout << "NBTTT " << aNbTT << "\n";

    cRandNParmiQ aSel(aNbTT,ElMin(aNbTT,NbMaxATT));
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
    mCoherMedAB =  MedianPond(aVPAB);
    mCoherMed12 =  MedianPond(aVP12);
}

void  cAppli_NewSolGolInit::InitRotOfArc(tArcNSI * anArc,bool Test)
{
   std::vector<cLinkTripl> & aVL = anArc->attr().ASym()->Lnk3();

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

   mCompKG.SetN(aVL.size());
   for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
   {
       cNOSolIn_Triplet * aTri1 = aVL[aK1].m3;
       for (int aK2=aK1+1 ; aK2<int(aVL.size()) ; aK2++)
       {
           cNOSolIn_Triplet * aTri2 = aVL[aK2].m3;
           double  aDC = DistCoherenceAtoB(anArc,aTri1,aTri2);
           double  aDatt =   CoutAttenueTetaMax(aDC,mCoherMedAB*FactAttCohMed);  //aDC / (aDC+mCoherMed*FactAttCohMed);
           mCompKG.AddCost(aK1,aK2,aTri2->Nb3(),aTri1->Nb3(),aDatt);
       }
   }

   std::vector<ElRotation3D> aVR;
   for (int aK1=0 ; aK1<int(aVL.size()) ; aK1++)
   {
       aVR.push_back(RotationC2toC1(anArc,aVL[aK1].m3));
   }
   
   int aKK = mCompKG.GetKernel();
   double aBSurH0 =  aVL[aKK].m3->BOnH();
   ElRotation3D aR0 = aVR[aKK];
   double aD0 = euclid(aR0.tr());


   for (int aNBIter = 0 ; aNBIter<4 ; aNBIter++)
   {
        ElMatrix<double> aSomMat(3,3,0.0);
        double aSomPds = 0.0;
        Pt3dr   aSomTr (0,0,0);
        for (int aK=0 ; aK<int(aVL.size()) ; aK++)
        {
             double aD =  DistanceRot(aR0,aVR[aK],aBSurH0);
             if (aD < 6 * mCoherMed12)
             {
                   double aPds = 1 /(1 + ElSquare(aD*(2*mCoherMed12)));
             } 
        }
   }


   if (Test)
   {
      std::cout <<  "KERNEL " <<   aVL[aKK].S3()->attr().Im()->Name() << "\n";
   }



}




cAppli_NewSolGolInit::cAppli_NewSolGolInit(int argc, char ** argv) :
    mQuick      (true),
    mTest       (true),
    mSimul      (false),
    mFlag3      (mGr.alloc_flag_som()),
    mFlag2      (mGr.alloc_flag_som()),
    mTestTrip   (0),
    mTestTrip2  (0),
    mTestS1     (0),
    mTestS2     (0),
    mTestArc    (0),
    mNbSom      (0),
    mNbArc      (0),
    mNbTrip     (0)
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
        LArgMain() << EAM(mOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mQuick,"Quick",true,"Quick version",eSAM_IsBool)
                   << EAM(mTest,"Test",true,"Test for tuning",eSAM_IsBool)
                   << EAM(aNameT1,"Test1",true,"Name of first test image",eSAM_IsBool)
                   << EAM(aNameT2,"Test2",true,"Name of second test image",eSAM_IsBool)
                   << EAM(aNameT3,"Test3",true,"Name of third test image",eSAM_IsBool)
                   << EAM(aNameT4,"Test4",true,"Name of fourth test image",eSAM_IsBool)
                   << EAM(mSimul,"Simul",true,"Simulation of perfect triplet",eSAM_IsBool)
                   << EAM(aModeBin,"Bin",true,"Binaries file, def = true",eSAM_IsBool)
   );

   cTplTriplet<std::string> aKTest1(aNameT1,aNameT2,aNameT3);
   cTplTriplet<std::string> aKTest2(aNameT1,aNameT2,aNameT4);

   mEASF.Init(mFullPat);
   mNM = new cNewO_NameManager(mQuick,mEASF.mDir,mOriCalib,"dat");
   const cInterfChantierNameManipulateur::tSet * aVIm = mEASF.SetIm();

   tSomNSI * mTestS1=0;
   tSomNSI * mTestS2=0;
   // tSomNSI * aTestS3=0;

   for (int aKIm=0 ; aKIm <int(aVIm->size()) ; aKIm++)
   {
       const std::string & aName = (*aVIm)[aKIm];
       tSomNSI & aSom = mGr.new_som(cNOSolIn_AttrSom(aName,*this));
       mMapS[aName] = & aSom;
       mNbSom++;
       if (mSimul)
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
                     double aScale = 1.1 + cos(mNbTrip);
                     aR2On1.tr() =  aR2On1.tr() * aScale;
                     aR3On1.tr() =  aR3On1.tr() * aScale;
                     aXml3Ori.Ori2On1() =  El2Xml(aR2On1);
                     aXml3Ori.Ori3On1() =  El2Xml(aR3On1);
                 }


                 cNOSolIn_Triplet * aTriplet = new cNOSolIn_Triplet(aS1,aS2,aS3,aXml3Ori);
/*
                 aS1->attr().AddTriplet(aTriplet,1,2,0);
                 aS2->attr().AddTriplet(aTriplet,0,2,1);
                 aS3->attr().AddTriplet(aTriplet,0,1,2);
*/
                 CreateArc(aS1,aS2,aTriplet,0,1,2);
                 CreateArc(aS2,aS3,aTriplet,1,2,0);
                 CreateArc(aS3,aS1,aTriplet,2,0,1);

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


    EstimCoherenceMed();

    std::cout << "COHERENCE MED, DAB  " << mCoherMedAB  * 1000  << " D12 " << mCoherMed12  * 1000 << "\n";


    if (mTestS1 && mTestS2)
    {
          mTestArc = mGr.arc_s1s2(*mTestS1,*mTestS2);
    }


    if (mTestArc)
    {
          InitRotOfArc(mTestArc,true);
    }

    if (mTestTrip)
    {
        std::cout << "mTestTrip " << mTestTrip << "\n";

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
    }
   
}


int CPP_NewSolGolInit_main(int argc, char ** argv)
{
    cAppli_NewSolGolInit anAppli(argc,argv);
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
