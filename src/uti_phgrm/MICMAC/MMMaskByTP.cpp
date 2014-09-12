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
extern const std::string TheDIRMergTiepForEPI();
#include "../src/uti_phgrm/MICMAC/MICMAC.h"
#ifdef MAC
// Modif Greg pour avoir le nom de la machine dans les log
#include <sys/utsname.h>
#endif


// Test Mercurial
extern Im2D_Bits<1>  TestLabel(Im2D_INT2 aILabel,INT aLabelOut);




//====================================================

static Video_Win *  TheWTiePCor = 0;
static double       TheScaleW = 1.0;
static Pt2di TheMaxSzW(1000,800);

void ShowPoint(Pt2dr aP,int aCoul,int aModeCoul,int aRab=0)
{
#ifdef ELISE_X11
   if (TheWTiePCor)
   {
       Pt2dr aP0 = aP - Pt2dr(aRab,aRab);
       Pt2dr aP1 = aP + Pt2dr(aRab+1,aRab+1);
       if (aModeCoul==0)
          TheWTiePCor->fill_rect(aP0,aP1,TheWTiePCor->pdisc()(aCoul));
       else if (aModeCoul==1)
          TheWTiePCor->fill_rect(aP0,aP1,TheWTiePCor->pcirc()(aCoul));
       else if (aModeCoul==2)
          TheWTiePCor->fill_rect(aP0,aP1,TheWTiePCor->pgray()(aCoul));
   }
#endif 
}

void ShowPoint(const std::vector<Pt2di> & aV,int aCoul,int aModeCoul)
{
#ifdef ELISE_X11
   if (TheWTiePCor)
   {
        for (int aK=0 ; aK<int(aV.size()) ; aK++)
        {
            ShowPoint(Pt2dr(aV[aK]),aCoul,aModeCoul);
        }
   }
#endif 
}

void  ShowMasq(Im2D_Bits<1> aMasq)
{
#ifdef ELISE_X11
   if (TheWTiePCor)
   {
       ELISE_COPY
       (
           select(aMasq.all_pts(),aMasq.in()==0),
           P8COL::green,
           TheWTiePCor->odisc()
       );
   }
#endif 
}

void MMClearW()
{
#ifdef ELISE_X11
   if (TheWTiePCor)
   {
      ELISE_COPY(TheWTiePCor->all_pts(),P8COL::green,TheWTiePCor->odisc());
   }
#endif 
}

/********************************************************************/
/*                                                                  */
/*                  Gestion des cellules                            */
/*                                                                  */
/********************************************************************/



const int TheMbMaxCel = 15;  // Pas tres fier de ca ...

class cCelTiep
{
   public :
      static const INT2  TheNoZ = -32000;
      cCelTiep() :
         mHeapInd  (HEAP_NO_INDEX),
         mZ        (TheNoZ),
         mNbCel    (0)
      {
              SetCostCorel(4.0);
      }

      int NbCel() const {return mNbCel;}
      void InitCel()
      {
          mNbCel =0;
          SetCostCorel(4.0);
          mExploredIndex.clear();
          mZ =  TheNoZ;
      }

      bool ZIsExplored(int aZ)
      {
            return     (mNbCel>= TheMbMaxCel)
                   ||  mExploredIndex.find(aZ) != mExploredIndex.end();
      }
      void SetZExplored(int aZ)
      {
            mNbCel++;
            mExploredIndex.insert(aZ);
      }


     void SetZ(int aZ) {mZ = aZ;}
     void SetPlani(int anX,int anY)
     {
          mX= anX;
          mY= anY;
     }

     Pt3di Pt() const {return Pt3di(mX,mY,mZ);}
     bool  IsSet() {return  mZ!= TheNoZ;}
/*
*/
     static const int MulCor = 1000;

     int & HeapInd() {return mHeapInd;}
     int   ICostCorel() const {return mCostCor;}
     double CostCorel() const {return mCostCor/double(MulCor);}
     void  SetCostCorel(double aCostCor) { mCostCor = round_ni(aCostCor*MulCor);}
      
   private :

       cCelTiep(const cCelTiep&); // N.I. 

       U_INT2  mCostCor;
       int   mHeapInd;
       INT2  mX;
       INT2  mY;
       INT2  mZ;
       int           mNbCel;
       std::set<int> mExploredIndex;
};


typedef cCelTiep * cCelTiepPtr;


class cHeap_CTP_Index
{
    public :
     static void SetIndex(cCelTiepPtr & aCTP,int i) 
     {
        aCTP->HeapInd() = i;
     }
     static int  Index(const cCelTiepPtr & aCTP)
     {
             return aCTP->HeapInd();
     }

};

class cHeap_CTP_Cmp
{
    public :
         bool operator () (const cCelTiepPtr & aCTP1,const cCelTiepPtr & aCTP2)
         {
               return aCTP1->ICostCorel() < aCTP2->ICostCorel();
         }
};

cHeap_CTP_Cmp    TheCmpCTP;

class cMMTP
{
    public :
      cMMTP(const Box2di & aBox,cAppliMICMAC &);
      void  ConputeEnveloppe(const cComputeAndExportEnveloppe &);
/*
      bool IsInMasq3D(int anX,int anY,int aZ)
      {
            return mMasq3D->IsInMasq(IndexAndProfPixel2Euclid(Pt2dr(aXIm,aYIm),aZIm));
      }
*/
      bool Inside(const int & anX,const int & anY,const int & aZ)
      {
         return     (anX>=mP0Tiep.x) 
                 && (anY>=mP0Tiep.y) 
                 && (anX<mP1Tiep.x) 
                 && (anY<mP1Tiep.y) 
                 && (aZ> cCelTiep::TheNoZ)
                 && (aZ< 32000)
                 &&  (mTImMasquageInput.get(Pt2di(anX-mP0Tiep.x,anY-mP0Tiep.y)))
                 &&  ((mMasq3D==0) ||    mMasq3D->IsInMasq(mNuage3D->IndexAndProfPixel2Euclid(Pt2dr(anX,anY),aZ)))
                ;
      }
       
     void SetMasq3D(cMasqBin3D * aMasq3D,cElNuage3DMaille * aNuage3D)
     {
           mMasq3D = aMasq3D;
           mNuage3D = aNuage3D;
     }
 
      cCelTiep & Cel(int anX,int anY) { return mTabCTP[anY][anX]; }
      cCelTiep & Cel(const Pt2di & aP) { return Cel(aP.x,aP.y);}

      void MajOrAdd(cCelTiep & aCel)
      {
           mHeapCTP.MajOrAdd(&aCel);
      }
      bool PopCel(cCelTiepPtr & aCPtr)
      {
         return mHeapCTP.pop(aCPtr);
      }
      int NbInHeap() {return mHeapCTP.nb();}
      void DoMasqAndProfInit(const cMasqueAutoByTieP & aMATP);

       Im2D_Bits<1> ImMasquageInput() {return   mImMasquageInput;}
       Im2D_Bits<1> ImMasqFinal() {return mImMasqFinal;}
       Im2D_Bits<1> ImMasqInit() {return mImMasqInit;}
       Im2D_INT2    ImProf() {return mImProf;}
       bool InMasqFinal(const Pt2di & aP) const {return  (mTImMasqFinal.get(aP) == 1) ? true : false ;}

       Im2D_REAL4 ImOrtho(cGPU_LoadedImGeom *);

    private :
        cAppliMICMAC &    mAppli;
        Pt2di             mP0Tiep;
        Pt2di             mP1Tiep;
        Pt2di             mSzTiep;
        cCelTiep **       mTabCTP;
        ElHeap<cCelTiepPtr,cHeap_CTP_Cmp,cHeap_CTP_Index> mHeapCTP;

        Im2D_INT2       mImProf;
        TIm2D<INT2,INT> mTImProf;
        Im2D_Bits<1>    mImMasqInit;
        TIm2DBits<1>    mTImMasqInit;   
        Im2D_Bits<1>    mImMasqFinal;
        TIm2DBits<1>    mTImMasqFinal;   


        Im2D_Bits<1>    mImMasquageInput;
        TIm2DBits<1>    mTImMasquageInput;   
        cMasqBin3D *       mMasq3D;
        cElNuage3DMaille * mNuage3D;

};

Im2D_REAL4 cMMTP::ImOrtho(cGPU_LoadedImGeom * aGLI)
{
   Im2D_REAL4 aIRes(mSzTiep.x,mSzTiep.y,0.0);
   TIm2D<REAL4,REAL8> aTRes(aIRes);
   Pt2di aP;
   for (aP.x=mP0Tiep.x ; aP.x<mP1Tiep.x ; aP.x++)
   {
       for (aP.y=mP0Tiep.y ; aP.y<mP1Tiep.y ; aP.y++)
       {
           Pt2di aPIm = aP-mP0Tiep;
           if (mTImMasqFinal.get(aPIm))
           {
               aTRes.oset(aPIm,aGLI->GetValOfDisc(aP.x,aP.y,mTImProf.get(aPIm)));
           }
       }
   }

   return aIRes;
}

cMMTP::cMMTP(const Box2di & aBox,cAppliMICMAC & anAppli) : 
   mAppli            (anAppli),
   mP0Tiep           (aBox._p0),
   mP1Tiep           (aBox._p1),
   mSzTiep           (aBox.sz()),
   mHeapCTP          (TheCmpCTP),
   mImProf           (mSzTiep.x,mSzTiep.y,0),
   mTImProf          (mImProf),
   mImMasqInit       (mSzTiep.x,mSzTiep.y,1),
   mTImMasqInit      (mImMasqInit),
   mImMasqFinal      (mSzTiep.x,mSzTiep.y),
   mTImMasqFinal     (mImMasqFinal),
   mImMasquageInput  (mSzTiep.x,mSzTiep.y,1),
   mTImMasquageInput (mImMasquageInput),
   mMasq3D           (0),
   mNuage3D          (0)
{
   mTabCTP = (new  cCelTiepPtr [mSzTiep.y]) - mP0Tiep.y;
   for (int anY = mP0Tiep.y ; anY<mP1Tiep.y ; anY++)
   {
        mTabCTP[anY] = (new cCelTiep [mSzTiep.x]) - mP0Tiep.x;
        for (int anX = mP0Tiep.x ; anX<mP1Tiep.x ; anX++)
        {
             mTabCTP[anY][anX].SetPlani(anX,anY);
        }
   }

}

extern Im2D_REAL4 ProlongByCont (Im2D_Bits<1> & aMasqRes, Im2D_Bits<1> aIMasq, Im2D_INT2 aInput, INT aNbProl,double aDistAdd,double DMaxAdd);

double    pAramSizeProlResol1()        {return 50;}
double    pAramSizeProlResolCur()      {return 5;}
double pAramSsEchFiler()            {return 3.0;}
int    pAramSzFilter()              {return 7;}
double    pAramPropMaxMin()         {return 0.9;}
double    pAramProlDistAdd()        {return 0.25;}
double    pAramProlDistMaxAdd()     {return 5.0;}

int pAramZoomFinal() {return 8;}



void  cMMTP::ConputeEnveloppe(const cComputeAndExportEnveloppe &)
{
   std::string  aNameTarget = mAppli.WorkDir() + TheDIRMergTiepForEPI() + "-" +  mAppli.PDV1()->Name() + "/NuageImProf_LeChantier_Etape_1.xml";

  std::cout << aNameTarget << "\n"; getchar();


   int aZoomTer = mAppli.CurEtape()->DeZoomTer();
   double aPasPx =  mAppli.CurEtape()->GeomTer().PasPxRel0();
//=============== READ PARAMS  ====================
   double  aStepSsEch = pAramSsEchFiler();
   int     aSzFiltrer = pAramSzFilter();
   double  aProp = pAramPropMaxMin();

   int     aDistProl = round_up(  ElMax(pAramSizeProlResolCur(),pAramSizeProlResol1() /double (aZoomTer)) /aStepSsEch);
   double  aDistCum =  (pAramProlDistMaxAdd()  / (aPasPx*  aZoomTer));
   double aDistAdd =   (pAramProlDistAdd() * aStepSsEch )  / (aPasPx);



//===================================

   int aNbIn = 0;
   int aNbOut = 0;
   Pt2di aP;

   for (aP.y = mP0Tiep.y ; aP.y<mP1Tiep.y ; aP.y++)
   {
        for (aP.x = mP0Tiep.x ; aP.x<mP1Tiep.x ; aP.x++)
        {
            cCelTiep & aCel =  Cel(aP);
            Pt2di aPIm = aP - mP0Tiep;
            if (aCel.IsSet())
            {
                 int aZ = aCel.Pt().z;
                 aNbIn++;
                 if (! mMasq3D->IsInMasq(mNuage3D->IndexAndProfPixel2Euclid(Pt2dr(aPIm),aZ)))
                 {
                    aNbOut++;
                 }
                 mTImProf.oset(aPIm,aZ);
                 mTImMasqInit.oset(aPIm,1);
            }
            else
            {
                mTImProf.oset(aPIm,0);
                mTImMasqInit.oset(aPIm,0);
            }
        }
    }
    ElTimer aChrono;



    int     aSeuilNbV = 2 * (1+2*aSzFiltrer); // Au moins une bande de 2 pixel pour inferer qqch
    Pt2di aSzRed = round_up(Pt2dr(mSzTiep)/aStepSsEch);

    Im2D_Bits<1>    aMasqRed(aSzRed.x,aSzRed.y,0);
    TIm2DBits<1>    aTMR(aMasqRed);
    TIm2D<INT2,INT> aPMaxRed(aSzRed);
    TIm2D<INT2,INT> aPMinRed(aSzRed);

    Pt2di aPRed;
    for (aPRed.y = 0 ; aPRed.y<aSzRed.y ; aPRed.y++)
    {
        for (aPRed.x = 0 ; aPRed.x<aSzRed.x ; aPRed.x++)
        {
             Pt2di aPR1 = round_ni(Pt2dr(aPRed)*aStepSsEch);
             int anX0 = ElMax(0,aPR1.x-aSzFiltrer);
             int anX1 = ElMin(mSzTiep.x-1,aPR1.x+aSzFiltrer);
             int anY0 = ElMax(0,aPR1.y-aSzFiltrer);
             int anY1 = ElMin(mSzTiep.y-1,aPR1.y+aSzFiltrer);
             std::vector<int> aVVals;
             Pt2di aVoisR1;
             for (aVoisR1.x=anX0 ; aVoisR1.x<=anX1 ; aVoisR1.x++)
             {
                  for (aVoisR1.y=anY0 ; aVoisR1.y<=anY1 ; aVoisR1.y++)
                  {
                     if (mTImMasqInit.get(aVoisR1))
                        aVVals.push_back( mTImProf.get(aVoisR1));
                  }
             }
             if (int(aVVals.size()) >= aSeuilNbV)
             {
                  int aVMax = KthValProp(aVVals,aProp);
                  int aVMin = KthValProp(aVVals,1-aProp);
                  aPMaxRed.oset(aPRed,aVMax);
                  aPMinRed.oset(aPRed,aVMin);
                  aTMR.oset(aPRed,1);
                  ELISE_ASSERT(aVMin<=aVMax,"Mic>Max !!!! in BasicMMTiep");
             }
             else
             {
                  aPMaxRed.oset(aPRed,-32000);
                  aPMinRed.oset(aPRed, 32000);
             }
        }
    }
    Tiff_Im::Create8BFromFonc("TDifInit.tif",aSzRed,Max(0,Min(255,Iconv(aPMaxRed._the_im.in()-aPMinRed._the_im.in()))));

    Im2D_Bits<1> aNewM(1,1);
    Im2D_REAL4  aNewMax = ProlongByCont (aNewM,aMasqRed,aPMaxRed._the_im,aDistProl,aDistAdd,aDistCum);
    Im2D_REAL4  aNewMin = ProlongByCont (aNewM,aMasqRed,aPMinRed._the_im,aDistProl,-aDistAdd,aDistCum);

/*
    std::cout <<  ":ConputeEnveloppe   IN " << aNbIn << " ;; Out " << aNbOut << " TimeF " << aChrono.uval() << "\n";
    
    Tiff_Im::Create8BFromFonc("TMax.tif",aSzRed,mod(aPMaxRed._the_im.in(),256));
    Tiff_Im::Create8BFromFonc("TMin.tif",aSzRed,mod(aPMinRed._the_im.in(),256));

    Im2D_REAL4  aNewMax = ProlongByCont (aNewM,aMasqRed,aPMaxRed._the_im,20 );
*/
    // Tiff_Im::Create8BFromFonc("TDifProl.tif",aSzRed,mod(Iconv(aNewMin.in()-aNewMax.in()),256));
    Tiff_Im::Create8BFromFonc("TDifProl.tif",aSzRed,Max(0,Min(255,Iconv(aNewMax.in()-aNewMin.in()))));

    std::cout << " DCUM " << aDistCum << "\n";
    std::cout << "DONE MASQ\n"; getchar();
}
             



void cMMTP::DoMasqAndProfInit(const cMasqueAutoByTieP & aMATP)
{
   Pt2di aP;
   for (aP.y = mP0Tiep.y ; aP.y<mP1Tiep.y ; aP.y++)
   {
        for (aP.x = mP0Tiep.x ; aP.x<mP1Tiep.x ; aP.x++)
        {
            cCelTiep & aCel =  Cel(aP);
            Pt2di aPIm = aP - mP0Tiep;
            mTImMasqInit.oset(aPIm,aCel.IsSet());
            mTImProf.oset(aPIm,aCel.Pt().z);
        }
   }

   ELISE_COPY
   (
       mImMasqInit.all_pts(),
       close_32(mImMasqInit.in(0),18),
       mImMasqFinal.out()
   );

   for (aP.y = mP0Tiep.y ; aP.y<mP1Tiep.y ; aP.y++)
   {
        for (aP.x = mP0Tiep.x ; aP.x<mP1Tiep.x ; aP.x++)
        {
            Pt2di aPIm = aP - mP0Tiep;
            if ((mTImMasqFinal.get(aPIm) && (!mTImMasqInit.get(aPIm))))
            {
                cCelTiep & aCel =  Cel(aP);
                aCel.InitCel();
            }
        }
   }

   if (1)
      ShowMasq(mImMasqFinal);

   Liste_Pts_INT4 aL(2);
   ELISE_COPY
   (
       select
       (
             mImMasqInit.all_pts(),
             mImMasqInit.in() & (!erod_d4(mImMasqInit.in(0),1))
       ),
       P8COL::red,
       Output(aL) 
   );
   Im2D_INT4 anIP = aL.image();
   int aNbP  = anIP.tx();
   int * aTX =  anIP.data()[0];
   int * aTY =  anIP.data()[1];

   std::cout << "NNbbbb  " << aL.card()  << " " << anIP.tx() << "\n";

   for (int aKP=0 ; aKP<aNbP; aKP++)
   {
      int aXIm = aTX[aKP];
      int aYIm = aTY[aKP];
      int aZIm = mTImProf.get(Pt2di(aXIm,aYIm));

      mAppli.MakeDerivAllGLI(aXIm,aYIm,aZIm);
      cCelTiep & aCel =  Cel(aXIm,aYIm);
      MajOrAdd(aCel);
   }
   mAppli.OneIterFinaleMATP(aMATP,true);

   for (aP.y = mP0Tiep.y ; aP.y<mP1Tiep.y ; aP.y++)
   {
        for (aP.x = mP0Tiep.x ; aP.x<mP1Tiep.x ; aP.x++)
        {
            cCelTiep & aCel =  Cel(aP);
            Pt2di aPIm = aP - mP0Tiep;
            mTImProf.oset(aPIm,aCel.Pt().z);
        }
   }


}

class cResCorTP
{
    public :
      cResCorTP (double aCSom,double aCMax,double aCMed) :
           mCostSom (aCSom),
           mCostMax (aCMax),
           mCostMed (aCMed)
      {
      }
      double  CSom() const {return  mCostSom;}
      double  CMax() const {return  mCostMax;}
      double  CMed() const {return  mCostMed;}
    private :
       double mCostSom;
       double mCostMax;
       double mCostMed;
};

cResCorTP cAppliMICMAC::CorrelMasqTP(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ)
{

    int aNbScale =   NbScaleOfPt(anX,anY);
    double aPdsCum = mVScaIm[aNbScale-1][0]->CumSomPdsMS();
 // std::cout << "NbSssCalll " << aNbScale << "\n";

static int aCptCMT =0 ; aCptCMT++;

    std::vector<int> aVOk;
    bool             Ok0 = false;
    for (int aKI=0 ; aKI<mNbIm ; aKI++)
    {
         double aSomIm = 0; 
         double aSomI2 = 0; 
         bool AllOk = true;
         for (int aKS=0 ; aKS<aNbScale ; aKS++)
         {
               Pt2di aSzV0 = mVScaIm[aKS][0]->SzV0();
               cGPU_LoadedImGeom * aLI = mVScaIm[aKS][aKI];
               cStatOneImage * aStat = aLI->ValueVignettByDeriv(anX,anY,aZ,1, aSzV0);
               if (aStat)
               {
                   double aPds = aLI->PdsMS();
                   aSomIm += aStat->mS1 * aPds;
                   aSomI2 += aStat->mS2 * aPds;
               }
               else
               {
                  AllOk = false;
               }
         }

         if (AllOk)
         {
             aVOk.push_back(aKI);
             if (aKI==0)
                Ok0 = true;
             aSomIm /= aPdsCum;
             aSomI2 /= aPdsCum;
             double anEct = aSomI2-ElSquare(aSomIm);
             aSomI2 = sqrt(ElMax(1e0,anEct));
             for (int aKS=0 ; aKS<aNbScale ; aKS++)
             {
                 cGPU_LoadedImGeom * aLI = mVScaIm[aKS][aKI];
                 cStatOneImage * aStat = aLI->VignetteDone();
                 aStat->Normalise(aSomIm,aSomI2);
             }
         }
    }

//std::cout << "AAAAAAAAAAAAA " << Ok0 << " " << aVOk.size() << "\n";

    if ((! Ok0) || (aVOk.size() < 2))
       return cResCorTP(4,4,4);
//std::cout << "BBBbbbb\n";

    double aSomDistTot = 0;
    double aMaxDistTot = 0;
    double aMinDistTot = 4;
    int aNbCpleOk = aVOk.size() - 1;
    for (int aKK=1 ; aKK<int(aVOk.size()) ; aKK++)
    {
         int aK0 = 0;
         int aK1 = aVOk[aKK];
         double aDistLoc = 0;
         for (int aKS=0 ; aKS<aNbScale ; aKS++)
         {
             cGPU_LoadedImGeom * aLI0 = mVScaIm[aKS][aK0];
             cGPU_LoadedImGeom * aLI1 = mVScaIm[aKS][aK1];
             cStatOneImage * aStat0 = aLI0->VignetteDone();
             cStatOneImage * aStat1 = aLI1->VignetteDone();
 
             aDistLoc += aStat0->SquareDist(*aStat1) * aLI0->PdsMS() ;
         }
         aSomDistTot += aDistLoc;
         aMaxDistTot = ElMax(aMaxDistTot,aDistLoc);
         aMinDistTot = ElMin(aMinDistTot,aDistLoc);
    }

    double aDistMed = aSomDistTot - (aMaxDistTot+aMinDistTot);
    aSomDistTot /=  (aPdsCum * aNbCpleOk);
    aMaxDistTot /=  aPdsCum;

    if (aNbCpleOk>2)
       aDistMed /= aPdsCum*(aNbCpleOk-2);
    else
       aDistMed = (aSomDistTot+aMaxDistTot) /2.0;

    //  std::cout << "DIISTTT :: " << aMaxDistTot << " " << aSomDistTot << "\n";
    
    return cResCorTP(aSomDistTot,aMaxDistTot,aDistMed);
}


void cAppliMICMAC::CTPAddCell(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ,bool Final)
{
   static int aCptR[5] ={0,0,0,0,0};
   if (0)
   {
       for (int aK=0 ; aK<5 ; aK++ )
           std::cout <<  " K"<<aK << "=" << aCptR[aK];
       std::cout << "\n";
   }

    aCptR[0] ++;

bool Pb = false && ((anX==247) && (anY==259)) ;// || ((anX==374) && (anY==234));
if (Pb)
{
    std::cout << "Pb " << mMMTP->Inside(anX,anY,aZ) 
              << " Masq "<< mGLOBMasq3D->IsInMasq(mGLOBNuage->IndexAndProfPixel2Euclid(Pt2dr(anX,anY),aZ))
              << " Pt " << mGLOBNuage->IndexAndProfPixel2Euclid(Pt2dr(anX,anY),aZ)
              << "\n";
}
   if (!mMMTP->Inside(anX,anY,aZ))
     return;
   aCptR[1] ++;

   if (Final && (! mMMTP->InMasqFinal(Pt2di(anX,anY))))
      return;
   aCptR[2] ++;

   cCelTiep & aCel =  mMMTP->Cel(anX,anY);


   // std::cout << "NBCCCEL " << aCel.NbCel() << " " << aZ << "\n";

   if (aCel.ZIsExplored(aZ)) 
      return;
   aCptR[3] ++;
   aCel.SetZExplored(aZ);

   cResCorTP aCost = CorrelMasqTP(aMATP,anX,anY,aZ) ;
   // std::cout << "Cots " << aCost.CSom() << " " << aCost.CMax() << " " << aCost.CMed()  << "\n";
   double aCSom = aCost.CSom();
   if (
         (     (aCSom > aMATP.SeuilSomCostCorrel()) 
            || (aCost.CMax() > aMATP.SeuilMaxCostCorrel()) 
            || (aCost.CMed() > aMATP.SeuilMedCostCorrel()) 
         )
         && (! Final)
      )
   {
      return ;
   }
   aCptR[4] ++;
   if (aCSom < aCel.CostCorel())
   {
        aCel.SetCostCorel(aCSom);
        aCel.SetZ(aZ);
        ShowPoint(Pt2dr(anX,anY),aZ*10,1);
        mMMTP->MajOrAdd(aCel);
   }

#if (ELISE_X11)
  int aPer =  100000;
  static int aCpt=0; aCpt++;
  if ((aCpt%aPer)==0)
  {
     std::cout << "CPT= " << aCpt << "\n";
     if (0)
        TheWTiePCor->DumpImage("DumpMMTieP_"+ToString(aCpt/aPer)+".tif");
  }
#endif
}

/********************************************************************/
/********************************************************************/
/********************************************************************/

void  cAppliMICMAC::MakeDerivAllGLI(int aX,int aY,int aZ)
{
   for (int aKIm=0 ; aKIm<int(mVLI.size()) ; aKIm++)
   {
       mVLI[aKIm]->MakeDeriv(aX,aY,aZ);
   }
}

void  cAppliMICMAC::OneIterFinaleMATP(const cMasqueAutoByTieP & aMATP,bool Final)
{
   std::cout << "IN ITER FINAL " << mMMTP->NbInHeap() << " FINAL " << Final << "\n";
   cCelTiepPtr aCPtr;
   while (mMMTP->PopCel(aCPtr))
   {
        Pt3di  aP = aCPtr->Pt();
        int aMxDZ = aMATP.DeltaZ();
        Pt3di aDP;

        MakeDerivAllGLI(aP.x,aP.y,aP.z);
        for (aDP.x=-1 ; aDP.x<=1 ; aDP.x++)
        {
            for (aDP.y=-1 ; aDP.y<=1 ; aDP.y++)
            {
                for (aDP.z=-aMxDZ ; aDP.z<=aMxDZ ; aDP.z++)
                {
                    Pt3di aQ = aP+aDP;
                    CTPAddCell(aMATP,aQ.x,aQ.y,aQ.z,Final);
                }
            }
        }
 
   }
   std::cout << "END ITER FINAL " << mMMTP->NbInHeap() << " FINAL " << Final << "\n";
}




void  cAppliMICMAC::DoMasqueAutoByTieP(const Box2di& aBox,const cMasqueAutoByTieP & aMATP)
{

   std::cout <<  "*-*-*-*-*-*- cAppliMICMAC::DoMasqueAutoByTieP    "<< mImSzWCor.sz() << " " << aBox.sz() << mCurEtUseWAdapt << "\n";


   ElTimer aChrono;
   mMMTP = new cMMTP(aBox,*this);

    if (aMATP.TiePMasqIm().IsInit())
    {
       int aDZ = aMATP.TiePMasqIm().Val().DeZoomRel();
       int aDil = aMATP.TiePMasqIm().Val().Dilate();

       std::string aNameMasq = NameImageMasqOfResol(mCurEtape->DeZoomTer()*aDZ);
       Tiff_Im aTM(aNameMasq.c_str());
       Pt2di aSZM = aTM.sz();
       Im2D_Bits<1> aM(aSZM.x,aSZM.y);
       ELISE_COPY(aM.all_pts(),aTM.in(),aM.out());

       Im2D_Bits<1> aNewM = mMMTP->ImMasquageInput();
       ELISE_COPY
       (
             aNewM.all_pts(),
             dilat_32(aM.in(0)[Virgule(FX,FY)/double(aDZ)],aDil*3),
              aNewM.out()
       );
    }


#ifdef ELISE_X11
   if (aMATP.Visu().Val())
   {
       Pt2dr aSzW = Pt2dr(aBox.sz());
       TheScaleW = ElMin(1.0,ElMin(TheMaxSzW.x/aSzW.x,TheMaxSzW.y/aSzW.y));

       // TheScaleW = 0.635;
       aSzW = aSzW * TheScaleW;

       TheWTiePCor= Video_Win::PtrWStd(round_ni(aSzW));
       TheWTiePCor=  TheWTiePCor->PtrChc(Pt2dr(0,0),Pt2dr(TheScaleW,TheScaleW),true);
       for (int aKS=0 ; aKS<mVLI[0]->NbScale() ; aKS++)
       {
           Im2D_REAL4 * anI = mVLI[0]->FloatIm(aKS);
           ELISE_COPY(anI->all_pts(),Max(0,Min(255,anI->in()/50)),TheWTiePCor->ogray());
       }
/*
       {
           ELISE_COPY(TheWTiePCor->all_pts(),mMMTP->ImMasquageInput().in(),TheWTiePCor->odisc());
           std::cout << "HERISE THE MAKSE \n"; getchar();
       }
*/
   }
#endif 
   std::string  aNamePts = mICNM->Assoc1To1
                           (
                              aMATP.KeyImFilePt3D(),
                              PDV1()->Name(),
                              true
                           );
   mTP3d = StdNuage3DFromFile(WorkDir()+aNamePts);

   cMasqBin3D * aMasq3D = 0;
   if (aMATP.Masq3D().IsInit())
   {
         aMasq3D  = cMasqBin3D::FromSaisieMasq3d(WorkDir()+aMATP.Masq3D().Val());
         std::vector<Pt3dr> aNewVec;
         for (int aK=0 ; aK<int(mTP3d->size()) ; aK++)
         {
              Pt3dr aP = (*mTP3d)[aK];
              if (aMasq3D->IsInMasq(aP))
                aNewVec.push_back(aP);
         }
         *mTP3d = aNewVec;
   }


   std::cout << "== cAppliMICMAC::DoMasqueAutoByTieP " << aBox._p0 << " " << aBox._p1 << " Nb=" << mTP3d->size() << "\n"; 
   std::cout << " =NB Im " << mVLI.size() << "\n";


   cXML_ParamNuage3DMaille aXmlN =  mCurEtape->DoRemplitXML_MTD_Nuage();


   {
       cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromParam(aXmlN,FullDirMEC());
       for (int aK=0 ; aK<int(mTP3d->size()) ; aK++)
       {
           Pt3dr aPE = (*mTP3d)[aK];
           Pt3dr aPL2 = aNuage->Euclid2ProfPixelAndIndex(aPE);

           int aXIm = round_ni(aPL2.x);
           int aYIm = round_ni(aPL2.y);
           int aZIm = round_ni(aPL2.z) ;


           MakeDerivAllGLI(aXIm,aYIm,aZIm);
           CTPAddCell(aMATP,aXIm,aYIm,aZIm,false);

           ShowPoint(Pt2dr(aXIm,aYIm),P8COL::red,0);
       }
       if (aMasq3D)
       {
           mMMTP->SetMasq3D(aMasq3D,aNuage);
           mGLOBMasq3D = aMasq3D;
           mGLOBNuage = aNuage;
       }
   }



   OneIterFinaleMATP(aMATP,false);
   const cComputeAndExportEnveloppe * aCAEE = aMATP.ComputeAndExportEnveloppe().PtrVal();
   if (aCAEE)
   {
       mMMTP->ConputeEnveloppe(*aCAEE);
       if (aCAEE->EndAfter().Val()) return;
   }


   // std::cout << "TIME CorTP " << aChrono.uval() << "\n";
   std::cout << " XML " << aXmlN.Image_Profondeur().Val().Image() << "\n";

   mMMTP->DoMasqAndProfInit(aMATP);

/*
{
    Im2D_INT2 anIm = anIpRes;
    Pt2di aSz = anIm.sz();
    Pt2di aP;
    int aNbIn = 0;
    int aNbOut = 0;
    for (aP.x=0 ; aP.x <aSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y <aSz.y ; aP.y++)
        {
            INT aZ= anIm.GetI(aP);
            if (aTIL.get(aP))
            {
               aNbIn++;
               if (! mGLOBMasq3D->IsInMasq(mGLOBNuage->IndexAndProfPixel2Euclid(Pt2dr(aP),aZ)))
               {
                  aNbOut++;
                  //std::cout << "OUUTttt CCCCC " << aP << "\n";
               }
            }
        }
    }
    std::cout <<  "FINN IN " << aNbIn << " ;; Out " << aNbOut << "\n";
}
*/


   if (aMATP.DoImageLabel().Val())
   {
        Tiff_Im::Create8BFromFonc
        (
             FullDirMEC() + "LabelTieP_Num_" + ToString(mCurEtape->Num()) + ".tif",
             mMMTP->ImProf().sz(),
             mMMTP->ImProf().in()%256
        );
   }

   Im2D_Bits<1>  aIL =  TestLabel(mMMTP->ImProf(),cCelTiep::TheNoZ);
   TIm2DBits<1>  aTIL(aIL);

   // Filtrage final
   Fonc_Num aF = aIL.in(0);
   int aSzM =2 ;
   for (int aK=0 ; aK<3 ;aK++)
       aF = rect_som(aF,aSzM) / ElSquare(1.0+2*aSzM);
   aF = aF>0.5;

   aF =  close_32(aF,18);

   ELISE_COPY(aIL.all_pts(),aF,aIL.out());

 
   if (TheWTiePCor)
   {
        ELISE_COPY
        (
              aIL.all_pts(),
              Virgule(aIL.in(),mMMTP->ImMasqInit().in(),0)*255,
              TheWTiePCor->orgb()
        );

        std::cout << "DONE MASQ !! \n";
        // getchar();
        ELISE_COPY(select(aIL.all_pts(),mMMTP->ImMasqFinal().in()),5,TheWTiePCor->odisc());
        // getchar();
   }


   Im2D_INT2   anIpRes = mMMTP->ImProf();

   switch(aMATP.ImPaintResult().Val())
   {
        case eImpaintL2 :
std::cout << "AAAAAAAAAAAAAAAa\n";
             anIpRes = ImpaintL2(mMMTP->ImMasqInit(),aIL,anIpRes,16);
        break;
        case eImpaintMNT :
std::cout << "BBBBBBBB\n";
             ComplKLipsParLBas(mMMTP->ImMasqInit(),aIL,anIpRes,aMATP.ParamIPMnt().Val());
        break;
        default :
std::cout << "CCCCCCC\n";
        break;
   }



{
    Im2D_INT2 anIm = anIpRes;
    Pt2di aSz = anIm.sz();
    Pt2di aP;
    int aNbIn = 0;
    int aNbOut = 0;
    for (aP.x=0 ; aP.x <aSz.x ; aP.x++)
    {
        for (aP.y=0 ; aP.y <aSz.y ; aP.y++)
        {
            INT aZ= anIm.GetI(aP);
            if (aTIL.get(aP))
            {
               aNbIn++;
               if (! mGLOBMasq3D->IsInMasq(mGLOBNuage->IndexAndProfPixel2Euclid(Pt2dr(aP),aZ)))
               {
                  aNbOut++;
                  //std::cout << "OUUTttt CCCCC " << aP << "\n";
               }
            }
        }
    }
    std::cout <<  "FINN IN " << aNbIn << " ;; Out " << aNbOut << "\n";
}


//  ==================  SAUVEGARDE DES DONNEES POU FAITE UN NUAGE ====================

   // std::string aNameMasq = FullDirMEC() +aXmlN.Image_Profondeur().Val().Masq();
   std::string aNameMasq =  NameImageMasqOfResol(mCurEtape->DeZoomTer());
   ELISE_COPY(aIL.all_pts(), aIL.in(), Tiff_Im(aNameMasq.c_str()).out());

   
   std::string aNameImage = FullDirMEC() +aXmlN.Image_Profondeur().Val().Image();
   ELISE_COPY(select(anIpRes.all_pts(),anIpRes.in()==cCelTiep::TheNoZ),0,anIpRes.out());
   ELISE_COPY ( anIpRes.all_pts(), anIpRes.in(), Tiff_Im(aNameImage.c_str()).out());

{
   int aVMax,aVMin;
   ELISE_COPY ( anIpRes.all_pts(), anIpRes.in(), VMax(aVMax)|VMin(aVMin));
   std::cout << "MaxMin " << aVMax << " :: " << aVMin << "\n";
}
   //cElNuage3DMaille * aNuage = ;

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
