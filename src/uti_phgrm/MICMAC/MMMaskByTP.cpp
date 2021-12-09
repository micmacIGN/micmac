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
#include "../src/uti_phgrm/MICMAC/MICMAC.h"
#ifdef MAC
// Modif Greg pour avoir le nom de la machine dans les log
#include <sys/utsname.h>
#endif


// Test Mercurial
extern Im2D_Bits<1>  TestLabel(Im2D_INT2 aILabel,INT aLabelOut);




//====================================================

#ifdef ELISE_X11
static Video_Win *  TheWTiePCor = 0;
static double       TheScaleW = 1.0;
static Pt2di TheMaxSzW(1600,1200);
#endif

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
      cMMTP(const Box2di & aBoxInLoc,const Box2di &aBoxInGlob,const Box2di & aBoxOut,cAppliMICMAC &);
      void  ConputeEnveloppe(const cComputeAndExportEnveloppe &,const cXML_ParamNuage3DMaille & aCurNuage);
      // Dequantifie et bouche, calcul les "petits"  trous et les bouches
      void ContAndBoucheTrou();
      void MaskProgDyn(const cParamFiltreDepthByPrgDyn & aParam);
      void MaskRegulMaj(const cParamFiltreDetecRegulProf & aParam);
      void FreeCel();

      // Transform les rsultat du heap n tableua de prof et de masque
      void  ExportResultInit();
      bool Inside(const int & anX,const int & anY,const int & aZ)
      {
         return     (anX>=mP0Tiep.x) 
                 && (anY>=mP0Tiep.y) 
                 && (anX<mP1Tiep.x) 
                 && (anY<mP1Tiep.y) 
                 && (aZ> cCelTiep::TheNoZ)
                 && (aZ< 32000)
                 &&  (mTImMasquageInput.get(Pt2di(anX-mP0Tiep.x,anY-mP0Tiep.y)))
                 &&  ((mMasq3D==0) ||    mMasq3D->IsInMasq(mNuage3D->IndexAndProfPixel2Euclid(Pt2dr(anX,anY)+mP0In,aZ)))
                ;
      }
       
     void SetMasq3D(cMasqBin3D * aMasq3D,cElNuage3DMaille * aNuage3D,Pt2dr aP0In)
     {
           mMasq3D = aMasq3D;
           mNuage3D = aNuage3D;
           mP0In = aP0In;
     }
 
      cCelTiep & Cel(int anX,int anY) { return mTabCTP[anY][anX]; }
      cCelTiep & Cel(const Pt2di & aP) { return Cel(aP.x,aP.y);}

      void MajOrAdd(cCelTiep & aCel)
      {
           mHeapCTP.MajOrAdd(&aCel);
      }
      // Remplit aCPtr avec la meilleure cellule
      bool PopCel(cCelTiepPtr & aCPtr)
      {
         return mHeapCTP.pop(aCPtr);
      }
      int NbInHeap() {return mHeapCTP.nb();}

       Im2D_Bits<1> ImMasquageInput() {return   mImMasquageInput;}
       //Im2D_Bits<1> ImMasqFinal() {return mImMasqFinal;}
       Im2D_Bits<1> ImMasqInit() {return mImMasqInit;}
       Im2D_Bits<1> ImMasqFinal() {return mImMasqFinal;}
       Im2D_INT2    ImProfInit() {return mImProf;}

       Im2D_REAL4   ImProfFinal() {return  mContBT;}   // image dequant et trous bouches

    private :
        Tiff_Im FileEnv(const std::string & aPost ,bool Bin); // Si pas  Bin int2
        void DoOneEnv(Im2D_REAL4 anEnvRed,Im2D_Bits<1> aNewM,bool isMax,const cXML_ParamNuage3DMaille & aTargetNuage,const cXML_ParamNuage3DMaille & aCurNuage,double aFactRed);


        cAppliMICMAC &    mAppli;
        Pt2di             mP0Tiep;
        Pt2di             mP1Tiep;
        Pt2di             mSzTiep;

        Box2di            mBoxInGlob;
        Box2di            mBoxInEnv;
        Box2di            mBoxOutGlob;
        Box2di            mBoxOutEnv;

        cCelTiep **       mTabCTP;
        ElHeap<cCelTiepPtr,cHeap_CTP_Cmp,cHeap_CTP_Index> mHeapCTP;

        Im2D_INT2       mImProf;
        TIm2D<INT2,INT> mTImProf;
        Im2D_Bits<1>    mImMasqInit;   // Result de la propgation par "heap"
        TIm2DBits<1>    mTImMasqInit;   

        Im2D_U_INT1        mImLabel;  // 0 => Rien, 1= > OK, 2=> trou bouche
        TIm2D<U_INT1,INT>  mTLab;
        Im2D_REAL4         mContBT;   // image dequant et trous bouches
        TIm2D<REAL4,REAL>  mTCBT;

        Im2D_Bits<1>    mImMasquageInput;   // eventuel masq de contrainte initial
        TIm2DBits<1>    mTImMasquageInput;   


        Im2D_Bits<1>    mImMasqFinal;   // resultat final si post filtrage
        TIm2DBits<1>    mTImMasqFinal;   

        Pt2dr mP0In;
        cMasqBin3D *       mMasq3D;
        cElNuage3DMaille * mNuage3D;
        std::string mNameTargetEnv;
        double mZoomTargetEnv;
        Pt2di mSzTargetEnv;

        int mDilatPlani;
        int mDilatAlti;
        // Fonc_Num fChCo;
        // Fonc_Num fMasq;
        // Fonc_Num fMasqBin;
};

Tiff_Im  cMMTP::FileEnv(const std::string & aPref,bool Bin) // Si pas  Bin int2
{
   std::string aNameRes  = DirOfFile(mNameTargetEnv) + aPref  + "_DeZoom" + ToString( mAppli.CurEtape()->DeZoomTer()) + ".tif";
   bool isNew;
   return Tiff_Im::CreateIfNeeded
          (
               isNew,
               aNameRes.c_str(),
               mSzTargetEnv,
               Bin ? GenIm::bits1_msbf : GenIm::real4,
               Tiff_Im::No_Compr,
               Tiff_Im::BlackIsZero
          );
}


/*
*/

cMMTP::cMMTP(const Box2di & aBoxLoc,const Box2di & aBoxInGlob,const Box2di & aBoxOut,cAppliMICMAC & anAppli) : 
   mAppli            (anAppli),
   mP0Tiep           (aBoxLoc._p0),
   mP1Tiep           (aBoxLoc._p1),
   mSzTiep           (aBoxLoc.sz()),
   mBoxInGlob        (aBoxInGlob),
   mBoxOutGlob       (aBoxOut),
   mHeapCTP          (TheCmpCTP),
   mImProf           (mSzTiep.x,mSzTiep.y,0),
   mTImProf          (mImProf),
   mImMasqInit       (mSzTiep.x,mSzTiep.y,1),
   mTImMasqInit      (mImMasqInit),
   mImLabel          (mSzTiep.x,mSzTiep.y),
   mTLab             (mImLabel),
   mContBT           (mSzTiep.x,mSzTiep.y),
   mTCBT             (mContBT),
   mImMasquageInput  (mSzTiep.x,mSzTiep.y,1),
   mTImMasquageInput (mImMasquageInput),
   mImMasqFinal      (mSzTiep.x,mSzTiep.y,1),
   mTImMasqFinal     (mImMasqFinal),
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

void cMMTP::FreeCel()
{
   for (int anY = mP0Tiep.y ; anY<mP1Tiep.y ; anY++)
   {
       delete [] (mTabCTP[anY] + mP0Tiep.x);
   }
   delete mTabCTP;
   mTabCTP = 0;
}



extern Im2D_REAL4 ProlongByCont (Im2D_Bits<1> & aMasqRes, Im2D_Bits<1> aIMasq, Im2D_INT2 aInput, INT aNbProl,double aDistAdd,double DMaxAdd);
extern Im2D_REAL4 ProlongByCont (Im2D_Bits<1> & aMasqRes, Im2D_Bits<1> aIMasq, Im2D_REAL4 aInput, INT aNbProl,double aDistAdd,double DMaxAdd);

Fonc_Num  AdaptDynOut(Fonc_Num aFonc,const cXML_ParamNuage3DMaille & aTargetNuage,const cXML_ParamNuage3DMaille & aCurNuage)
{
    const cImage_Profondeur & ipIn = aCurNuage.Image_Profondeur().Val();
    const cImage_Profondeur & ipOut = aTargetNuage.Image_Profondeur().Val();
    aFonc = aFonc*  ipIn.ResolutionAlti() +ipIn.OrigineAlti();
    aFonc = (aFonc-ipOut.OrigineAlti()) / ipOut.ResolutionAlti() ;
    return aFonc;
}

//      anEnvRed.in(aDefVal)     ; aNewM.in(0) , fChCo   , aDefFunc
Fonc_Num  FoncChCoordWithMasq(Fonc_Num aFoncInit,Fonc_Num aMasqInit, Fonc_Num aFoncChCo,Fonc_Num  aDefFunc,Fonc_Num & aMasqBin)
{
    Fonc_Num fMasq = aMasqInit[aFoncChCo];
    aMasqBin  = fMasq>0.5;

    Symb_FNum sMasqBin (aMasqBin);
    Fonc_Num aRes =   sMasqBin * (aFoncInit[aFoncChCo] / Max(fMasq,1e-5))  + (1-sMasqBin) * (aDefFunc);

    return aRes;
}

void cMMTP::DoOneEnv(Im2D_REAL4 anEnvRed,Im2D_Bits<1> aNewM,bool isMax,const cXML_ParamNuage3DMaille & aTargetNuage,const cXML_ParamNuage3DMaille & aCurNuage,double aRedFact)
{
    int aSign  = isMax ? 1 : - 1;
    int aDefVal = -(aSign * 32000);


    Fonc_Num  aFMasqBin;
    Fonc_Num fChCo = Virgule(FX,FY)/ (aRedFact);
    Fonc_Num aRes = FoncChCoordWithMasq(anEnvRed.in(aDefVal),aNewM.in(0),fChCo,aDefVal,aFMasqBin);

    aRes = aRes + mDilatAlti * aSign;
    aRes  =  isMax ? rect_max(aRes,mDilatPlani)  : rect_min(aRes,mDilatPlani);
    aRes = ::AdaptDynOut(aRes,aTargetNuage,aCurNuage);

    Tiff_Im  aFileRes = FileEnv(isMax?"EnvMax":"EnvMin",false);
    ELISE_COPY(rectangle(mBoxOutEnv._p0,mBoxOutEnv._p1),trans(aRes * aFMasqBin,-mBoxInEnv._p0),aFileRes.out());

    if (isMax)
    {
        Tiff_Im  aFileMasq = FileEnv("EnvMasq",true);
        ELISE_COPY(rectangle(mBoxOutEnv._p0,mBoxOutEnv._p1),trans(aFMasqBin,-mBoxInEnv._p0),aFileMasq.out());
    }
}

void  cMMTP::ExportResultInit()
{
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
                 if (mMasq3D && (! mMasq3D->IsInMasq(mNuage3D->IndexAndProfPixel2Euclid(Pt2dr(aPIm),aZ))))
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
}


void  cMMTP::ConputeEnveloppe(const cComputeAndExportEnveloppe & aCAEE,const cXML_ParamNuage3DMaille & aCurNuage)
{

   mNameTargetEnv = mAppli.WorkDir() + TheDIRMergeEPI()  +  mAppli.PDV1()->Name() + "/NuageImProf_LeChantier_Etape_1.xml";

   mNameTargetEnv = aCAEE.NuageExport().ValWithDef(mNameTargetEnv);
   cXML_ParamNuage3DMaille aTargetNuage = StdGetFromSI(mNameTargetEnv,XML_ParamNuage3DMaille);
   mZoomTargetEnv = aTargetNuage.SsResolRef().Val();
   mSzTargetEnv =  aTargetNuage.NbPixel();
   double aZoomRel = mAppli.CurEtape()->DeZoomTer()/mZoomTargetEnv;

   mBoxOutEnv._p0 = round_ni(Pt2dr(mBoxOutGlob._p0) * aZoomRel);
   mBoxOutEnv._p1 = round_ni(Pt2dr(mBoxOutGlob._p1) * aZoomRel);
   mBoxInEnv._p0 = round_ni(Pt2dr(mBoxInGlob._p0) * aZoomRel);
   mBoxInEnv._p1 = round_ni(Pt2dr(mBoxInGlob._p1) * aZoomRel);



   ELISE_ASSERT(mP0Tiep==Pt2di(0,0),"Too lazy to handle box maping");


   double aPasPx =  mAppli.CurEtape()->GeomTer().PasPxRel0();
//=============== READ PARAMS  ====================
   double  aStepSsEch = aCAEE.SsEchFilter().Val();
   int     aSzFiltrer = aCAEE.SzFilter().Val();
   double  aProp = aCAEE.ParamPropFilter().Val();

   int     aDistProl = round_up(  ElMax(aCAEE.ProlResolCur().Val(),aCAEE.ProlResolCible().Val()/aZoomRel) /aStepSsEch);
   double  aDistCum =  (aCAEE.ProlDistAddMax().Val()  / (aPasPx*  aZoomRel));
   double aDistAdd =   (aCAEE.ProlDistAdd().Val()*aStepSsEch )  / (aPasPx);

   std::cout << "DIST CUM " << aDistCum << " DADD " << aDistAdd << "\n";


//===================================

    ElTimer aChrono;

    int     aSeuilNbV = 2 * (1+2*aSzFiltrer); // Au moins une bande de 2 pixel pour inferer qqch
    Pt2di aSzRed = round_up(Pt2dr(mSzTiep)/aStepSsEch);

    Im2D_Bits<1>    aMasqRed(aSzRed.x,aSzRed.y,0);
    TIm2DBits<1>    aTMR(aMasqRed);
/*
    TIm2D<INT2,INT> aPMaxRed(aSzRed);
    TIm2D<INT2,INT> aPMinRed(aSzRed);
*/
    TIm2D<REAL4,REAL> aPMaxRed(aSzRed);
    TIm2D<REAL4,REAL> aPMinRed(aSzRed);

    // Calcul du filtre de reduction
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
             std::vector<REAL> aVVals;
             Pt2di aVoisR1;
             for (aVoisR1.x=anX0 ; aVoisR1.x<=anX1 ; aVoisR1.x++)
             {
                  for (aVoisR1.y=anY0 ; aVoisR1.y<=anY1 ; aVoisR1.y++)
                  {
                     if (mTImMasqInit.get(aVoisR1))
                     {
                        double aVal =  mTCBT.get(aVoisR1);
                        // Rustine, devrait normalement aller voir pourquoi certaines valeurs sont nan !!!
                        if (! std_isnan(aVal))
                        {
                           aVVals.push_back(aVal);
                        }
                        // aVVals.push_back( mTCBT.get(aVoisR1));
                     }
                        // aVVals.push_back( mTImProf.get(aVoisR1));
                  }
             }
             if (int(aVVals.size()) >= aSeuilNbV)
             {
                  REAL4 aVMax = KthValProp(aVVals,aProp);
                  REAL4 aVMin = KthValProp(aVVals,1-aProp);
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
    //Tiff_Im::Create8BFromFonc("TDifInit.tif",aSzRed,Max(0,Min(255,Iconv(aPMaxRed._the_im.in()-aPMinRed._the_im.in()))));

    Im2D_Bits<1> aNewM(1,1);
    Im2D_REAL4  aNewMax = ProlongByCont (aNewM,aMasqRed,aPMaxRed._the_im,aDistProl,aDistAdd,aDistCum);
    Im2D_REAL4  aNewMin = ProlongByCont (aNewM,aMasqRed,aPMinRed._the_im,aDistProl,-aDistAdd,aDistCum);
    ELISE_COPY(select(aNewM.all_pts(),!aNewM.in()),0,aNewMax.out()|aNewMin.out());

    // fChCo = Virgule(FX,FY)/ (aStepSsEch * aZoomRel);
    // fMasq = aNewM.in(0)[fChCo];
    // fMasqBin = fMasq>0.5;


    mDilatPlani = ElMax(aCAEE.DilatPlaniCible().Val(),round_up(aCAEE.DilatPlaniCur().Val()*aZoomRel));
    mDilatAlti  = ElMax(aCAEE.DilatAltiCible ().Val(),round_up(aCAEE.DilatPlaniCur().Val()*aZoomRel));
    
    DoOneEnv(aNewMax,aNewM,true ,aTargetNuage,aCurNuage,aStepSsEch * aZoomRel);
    DoOneEnv(aNewMin,aNewM,false,aTargetNuage,aCurNuage,aStepSsEch * aZoomRel);


    Fonc_Num  aFMasqBin;
    Fonc_Num fChCo = Virgule(FX,FY)/ aZoomRel;

std::cout  << "ZRRRR  " << aZoomRel <<  " 1/Z " << (1/aZoomRel) 
           <<   " ;; " << mAppli.CurEtape()->DeZoomTer() << " , " << mZoomTargetEnv << "\n";
// Tiff_Im::CreateFromIm(mContBT,DirOfFile(mNameTargetEnv)+"CONTBT.tif");


    Fonc_Num aFoncProf = FoncChCoordWithMasq(mContBT.in(0),mImMasqFinal.in(0),fChCo,0,aFMasqBin);
    aFoncProf = ::AdaptDynOut(aFoncProf,aTargetNuage,aCurNuage);

    Tiff_Im aFileProf = FileEnv("Depth",false);
    ELISE_COPY(rectangle(mBoxOutEnv._p0,mBoxOutEnv._p1),trans(aFoncProf,-mBoxInEnv._p0),aFileProf.out());


    Tiff_Im aFileMasq = FileEnv("Masq",true);
    ELISE_COPY(rectangle(mBoxOutEnv._p0,mBoxOutEnv._p1),trans(aFMasqBin,-mBoxInEnv._p0),aFileMasq.out());


#ifdef ELISE_X11
   if (0 && TheWTiePCor)
   {
       ELISE_COPY(TheWTiePCor->all_pts(),aFMasqBin,TheWTiePCor->odisc());
       std::cout << "AAAAAAAAAAAAAAAAAAAAa\n";
       TheWTiePCor->clik_in();
       ELISE_COPY(TheWTiePCor->all_pts(),aFileMasq.in(),TheWTiePCor->odisc());
       std::cout << "bbBBbbBBBBBBBbbb\n";
       TheWTiePCor->clik_in();
   }
#endif
}
             

void cMMTP::ContAndBoucheTrou()
{
   int aDist32Close = 6;
   int aNbErod = 6;

   // 1- Quelques fitre morpho de base, pour calculer les points eligibles au bouche-trou
   int aLabelOut = 0;
   //int aLabelIn = 1;
   int aLabelClose = 2;
   int aLabelFront = 3;

   ELISE_COPY(mImMasqInit.all_pts(),mImMasqInit.in(),mImLabel.out());
   ELISE_COPY(mImLabel.border(2),aLabelOut,mImLabel.out());

      // 1.1 calcul des point dans le fermeture
   ELISE_COPY
   (
          select
          (
             mImLabel.all_pts(),
             close_32(mImLabel.in(0),aDist32Close) && (mImLabel.in()==aLabelOut)
          ),
          aLabelClose,
          mImLabel.out()
   );
   ELISE_COPY(mImLabel.border(2),aLabelOut,mImLabel.out());


      // 1.2 erosion de ces points
   Neighbourhood V4 = Neighbourhood::v4();
   Neighbourhood V8 = Neighbourhood::v8();
   Neigh_Rel aRelV4(V4);

   Liste_Pts_U_INT2 aLFront(2);
   ELISE_COPY
   (
          select
          (
             mImLabel.all_pts(),
             (mImLabel.in(0)==aLabelClose) &&  (aRelV4.red_max(mImLabel.in(0)==aLabelOut))
          ),
          aLabelFront,
          mImLabel.out() | aLFront
    );
    for (int aK=0 ; aK<aNbErod ; aK++)
    {
        Liste_Pts_U_INT2 aLNew(2);
        ELISE_COPY
        (
               dilate
               (
                  aLFront.all_pts(),
                  mImLabel.neigh_test_and_set(Neighbourhood::v4(),2,3,20)
               ),
               aLabelFront,
               aLNew
         );
         aLFront = aLNew;
    }
    ELISE_COPY(select(mImLabel.all_pts(),mImLabel.in()==aLabelFront),0,mImLabel.out());

    // Au cas ou on ferait un export premature
    ELISE_COPY(mImMasqFinal.all_pts(),mImLabel.in()!=0,mImMasqFinal.out());

    int aSomMaskF;
    ELISE_COPY(mImMasqFinal.all_pts(),mImLabel.in()==1,sigma(aSomMaskF));
    if (aSomMaskF < 100) return;
    // std::cout << "aSomMaskFaSomMaskF " << aSomMaskF << "\n";
   // 2- Dequantifiication, adaptee au image a trou

       Im2D_REAL4 aProfCont(mSzTiep.x,mSzTiep.y,0.0);
       {
           Im2D_INT2 aPPV = BouchePPV(mImProf,mImLabel.in()==1);

           ElImplemDequantifier aDeq(mSzTiep);
           aDeq.DoDequantif(mSzTiep,aPPV.in());
           ELISE_COPY(aProfCont.all_pts(),aDeq.ImDeqReelle(),aProfCont.out());

           ELISE_COPY(select(aProfCont.all_pts(),mImLabel.in()!=1),0,aProfCont.out());
       }

       
    //Im2D_REAL4 aImInterp(mSzTiep.x,mSzTiep.y);
    TIm2D<REAL4,REAL8> aTInterp(mContBT);

   // 3- Bouchage "fin" des trour par moinde L2
          // 3.1 Valeur initial

                 // Filtrage gaussien
    Fonc_Num aFMasq = (mImLabel.in(0)==1);
    Fonc_Num aFProf = (aProfCont.in(0) * aFMasq);
    for (int aK=0 ; aK<3 ; aK++)
    {
          aFMasq = rect_som(aFMasq,1) /9.0;
          aFProf = rect_som(aFProf,1) /9.0;
    }

    ELISE_COPY
    (
         mContBT.all_pts(),
         aFProf / Max(aFMasq,1e-9),
         mContBT.out()
    );
                 // On remet la valeur init au point ayant un valeur propre
    ELISE_COPY
    (
         select(mContBT.all_pts(),mImLabel.in()==1),
         aProfCont.in(),
         mContBT.out()
    );
             // Et rien en dehors de l'image
    ELISE_COPY
    (
         select(mContBT.all_pts(),mImLabel.in()==0),
         0,
         mContBT.out()
    );
  
  
  
  
       // 3.2 Iteration pour regulariser les points interpoles
    {
         std::vector<Pt2di> aVInterp;
         {
            Pt2di aP;
            for (aP.x=0 ; aP.x<mSzTiep.x ; aP.x++)
            {
                for (aP.y=0 ; aP.y<mSzTiep.y ; aP.y++)
                {
                   if (mTLab.get(aP)==aLabelClose)
                     aVInterp.push_back(aP);
                }
            }
         }

         for (int aKIter=0 ; aKIter<20 ; aKIter++)
         {
              std::vector<double> aVVals;
              for (int aKP=0 ; aKP<int(aVInterp.size()) ; aKP++)
              {
                   double aSom=0;
                   double aSomPds = 0;
                   Pt2di aPK = aVInterp[aKP];
                   for (int aKV=0 ; aKV<9 ; aKV++)
                   {
                         Pt2di aVois = aPK+TAB_9_NEIGH[aKV];
                         if (mTLab.get(aVois)!=0)
                         {
                             int aPds = PdsGaussl9NEIGH[aKV];
                             aSom +=  aTInterp.get(aVois) * aPds;
                             aSomPds += aPds;
                         }
                   }
                   ELISE_ASSERT(aSomPds!=0,"Assert P!=0");
                   aVVals.push_back(aSom/aSomPds);
              }
              for (int aKP=0 ; aKP<int(aVInterp.size()) ; aKP++)
              {
                 aTInterp.oset(aVInterp[aKP],aVVals[aKP]);
              }
         }
    }
    
/*
*/

#ifdef ELISE_X11
           if(0 && TheWTiePCor)
           {

              ELISE_COPY
              (
                   mImLabel.all_pts(),
                   mContBT.in()*7,
                   TheWTiePCor->ocirc()
              );
              TheWTiePCor->clik_in();
              
              ELISE_COPY
              (
                  mImLabel.all_pts(),
                  nflag_close_sym(flag_front4(mImLabel.in(0)==1)),
                  TheWTiePCor->out_graph(Line_St(TheWTiePCor->pdisc()(P8COL::black)))
              );
              TheWTiePCor->clik_in();

              ELISE_COPY
              (
                  mImLabel.all_pts(),
                  mImLabel.in(0),
                  TheWTiePCor->odisc()
              );
              TheWTiePCor->clik_in();
              ELISE_COPY
              (
                  mImLabel.all_pts(),
                  mImMasqFinal.in(0),
                  TheWTiePCor->odisc()
              );
              TheWTiePCor->clik_in();
           }
#endif
}

void cMMTP::MaskRegulMaj(const cParamFiltreDetecRegulProf & aParam)
{
   double aPasPx =  mAppli.CurEtape()->GeomTer().PasPxRel0();
   Im2D_REAL4 anIm(mSzTiep.x,mSzTiep.y);
   ELISE_COPY(anIm.all_pts(),mImProf.in()*aPasPx,anIm.out());
    
   Im2D_Bits<1>   aNewMasq = FiltreDetecRegulProf(anIm,mImMasqInit,aParam);

#ifdef ELISE_X11
    if(TheWTiePCor)
    {
           std::cout << "SHOW FiltreDetecRegulProf\n";
           ELISE_COPY(mImLabel.all_pts(),mImMasqInit.in(),TheWTiePCor->odisc());
           ELISE_COPY
           (
                 select(mImLabel.all_pts(),aNewMasq.in()),
                 P8COL::green,
                 TheWTiePCor->odisc()
           );
           TheWTiePCor->clik_in();
    }
#endif
   mImMasqInit = aNewMasq;
}


void cMMTP::MaskProgDyn(const cParamFiltreDepthByPrgDyn & aParam)
{
    std::cout << "BEGIN MASK PRGD\n";
    mImMasqFinal = FiltrageDepthByProgDyn(mContBT,mImLabel,aParam);
    std::cout << "END MASK PRGD\n";
    
#ifdef ELISE_X11
    if(TheWTiePCor)
    {
           ELISE_COPY(mImLabel.all_pts(),mImLabel.in(),TheWTiePCor->odisc());
           ELISE_COPY
           (
                 select(mImLabel.all_pts(),mImMasqFinal.in() && (mImLabel.in()==1)),
                 P8COL::green,
                 TheWTiePCor->odisc()
           );
           ELISE_COPY
           (
                 select(mImLabel.all_pts(),mImMasqFinal.in() && (mImLabel.in()==2)),
                 P8COL::blue,
                 TheWTiePCor->odisc()
           );
           TheWTiePCor->clik_in();
    }
#endif
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
    int aNbCpleOk = (int)(aVOk.size() - 1);
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

   // Control la boite, les debordment en Z, le masque 3d, le masque image ...
   if (!mMMTP->Inside(anX,anY,aZ))
     return;
   aCptR[1] ++;

   aCptR[2] ++;
  
   // Recupere la cellule au point X,Y (existe toujours)
   cCelTiep & aCel =  mMMTP->Cel(anX,anY);

   // std::cout << "NBCCCEL " << aCel.NbCel() << " " << aZ << "\n";

   // Pas le peine de perdre du temps si on est deja passe par la
   if (aCel.ZIsExplored(aZ)) 
      return;
   aCptR[3] ++;
   // Memoriser qu'on est deja passe par la
   aCel.SetZExplored(aZ);

   // Calcul le cout  (par du CorMS ?)
   cResCorTP aCost = CorrelMasqTP(aMATP,anX,anY,aZ) ;
   // std::cout << "Cots " << aCost.CSom() << " " << aCost.CMax() << " " << aCost.CMed()  << "\n";
   double aCSom = aCost.CSom();
   // Different type de seuil pour eliminer
   if (
         (     (aCSom > aMATP.SeuilSomCostCorrel()) 
            || (aCost.CMax() > aMATP.SeuilMaxCostCorrel()) 
            || (aCost.CMed() > aMATP.SeuilMedCostCorrel()) 
         )
      )
   {
      return ;
   }
   aCptR[4] ++;
   // Si le cout est meilleur que le meilleur cout courrant on met a jour
   if (aCSom < aCel.CostCorel())
   {
        aCel.SetCostCorel(aCSom);
        aCel.SetZ(aZ);
        ShowPoint(Pt2dr(anX,anY),aZ*10,1);
        // Maj Or Ad => Ajoute si n'existe pas, Mise a jour sinon
        mMMTP->MajOrAdd(aCel);
   }

#if (ELISE_X11)
  // Eventuelle generation d'images pour illustrer
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
   // Les derivees sont precalculees sur toutes les images
   for (int aKIm=0 ; aKIm<int(mVLI.size()) ; aKIm++)
   {
       mVLI[aKIm]->MakeDeriv(aX,aY,aZ);
   }
}

void  cAppliMICMAC::OneIterFinaleMATP(const cMasqueAutoByTieP & aMATP,bool Final)
{
   std::cout << "IN ITER FINAL " << mMMTP->NbInHeap() << " FINAL " << Final << "\n";
   cCelTiepPtr aCPtr;
   // Tant qu'il y a des cellule, prendre la meilleur
   while (mMMTP->PopCel(aCPtr))
   {
        Pt3di  aP = aCPtr->Pt();
        int aMxDZ = aMATP.DeltaZ();
        Pt3di aDP;

        // Parcourir les voisin
        MakeDerivAllGLI(aP.x,aP.y,aP.z);
        for (aDP.x=-1 ; aDP.x<=1 ; aDP.x++)
        {
            for (aDP.y=-1 ; aDP.y<=1 ; aDP.y++)
            {
                for (aDP.z=-aMxDZ ; aDP.z<=aMxDZ ; aDP.z++)
                {
                    Pt3di aQ = aP+aDP;
                    // Mettre tout les voisin a explorer
                    CTPAddCell(aMATP,aQ.x,aQ.y,aQ.z,Final);
                }
            }
        }
 
   }
   std::cout << "END ITER FINAL " << mMMTP->NbInHeap() << " FINAL " << Final << "\n";
}

Fonc_Num FoncHomog(Im2D_REAL4 anIm, int aSzKernelH, double aPertPerPix)
{
    Fonc_Num aFMax = rect_max(anIm.in_proj(),aSzKernelH);
    Fonc_Num aFMin = rect_min(anIm.in_proj(),aSzKernelH);
    Fonc_Num aFHom =  aFMin > (1- aPertPerPix * aSzKernelH) * aFMax;

    return rect_max(aFHom,aSzKernelH);
}



void  cAppliMICMAC::DoMasqueAutoByTieP(const Box2di& aBoxLoc,const cMasqueAutoByTieP & aMATP)
{

   std::cout << "cAppliMICMAC::DoMasqueAutoByTieP " << aBoxLoc << "\n";

   // std::cout <<  "*-*-*-*-*-*- cAppliMICMAC::DoMasqueAutoByTieP    "<< mImSzWCor.sz() << " " << aBox.sz() << mCurEtUseWAdapt << "\n";


   ElTimer aChrono;
   mMMTP = new cMMTP(aBoxLoc,mBoxIn,mBoxOut,*this);

    // Si il faut repartir d'un masque initial calcule a un de zool anterieur
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

    // Si on active le filtre "anti-ciel"
    if (aMATP.mmtpFilterSky().IsInit())
    {
         Im2D_REAL4 * anIm = mPDV1->LoadedIm().FirstFloatIm();
         ELISE_ASSERT(anIm!=0,"Incohe in mmtpFilterSky");
         // Pt2di aSz = anIm->sz();
         Pt2di aSz = mMMTP->ImMasquageInput().sz();

         const cmmtpFilterSky & aFS = aMATP.mmtpFilterSky().Val();
         int aSeuilNbPts = round_ni(aSz.x*aSz.y*aFS.PropZonec().Val());

         Im2D_U_INT1 aImLabel(aSz.x,aSz.y);
         TIm2D<U_INT1,INT> aTLab(aImLabel);

         // Fonction d'homogenite , est homogene si sur un voisinage le Min est superieur a une proportion du max
         Fonc_Num FHGlob = FoncHomog(*anIm,aFS.SzKernelHom().Val(),aFS.PertPerPix().Val());
         ELISE_COPY(aImLabel.all_pts(),FHGlob,aImLabel.out());
         FiltrageCardCC(true,aTLab,1,2,aSeuilNbPts);

         Im2D_Bits<1> aNewM = mMMTP->ImMasquageInput();
         ELISE_COPY(select(aImLabel.all_pts(),aImLabel.in()==1),0,aNewM.out());
          
    }

 #ifdef ELISE_X11
   // Cree les fenetre qui permettront la visualisation progressive
   if (aMATP.Visu().Val())
   {
       Pt2dr aSzW = Pt2dr(aBoxLoc.sz());
       TheScaleW = ElMin(1000.0,ElMin(TheMaxSzW.x/aSzW.x,TheMaxSzW.y/aSzW.y));  // Pour l'instant on accepts Zoom>1 , donc => 1000

       // TheScaleW = 0.635;
       aSzW = aSzW * TheScaleW;

       TheWTiePCor= Video_Win::PtrWStd(round_ni(aSzW));
       TheWTiePCor=  TheWTiePCor->PtrChc(Pt2dr(0,0),Pt2dr(TheScaleW,TheScaleW),true);
       for (int aKS=0 ; aKS<mVLI[0]->NbScale() ; aKS++)
       {
           Im2D_REAL4 * anI = mVLI[0]->FloatIm(aKS);
           ELISE_COPY(anI->all_pts(),Max(0,Min(255,anI->in()/50)),TheWTiePCor->ogray());
       }
   }
#endif 
   std::string  aNamePts = mICNM->Assoc1To1
                           (
                              aMATP.KeyImFilePt3D(),
                              PDV1()->Name(),
                              true
                           );
   // Lecture des germes de l'appariement, ce sont des points 3D genere a
   // a partir des points homologues dans "AperoChImSecMM"
   mTP3d = StdNuage3DFromFile(WorkDir()+aNamePts);

   // Filtre avec le masque 3D
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

   std::cout << "== cAppliMICMAC::DoMasqueAutoByTieP " << aBoxLoc._p0 << " " << aBoxLoc._p1 << " Nb=" << mTP3d->size() << "\n"; 
   std::cout << " =NB Im " << mVLI.size() << "\n";

   cXML_ParamNuage3DMaille aXmlN =  mCurEtape->DoRemplitXML_MTD_Nuage();


   // On rentre tous les germes
   {
       // On lit le nuage qui permet de faire les conversions geometriques
       // pour "rasteriser" les points 3D
       cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromParam(mPDV1->Name(),aXmlN,FullDirMEC());
       if (aMasq3D)
       {
           mMMTP->SetMasq3D(aMasq3D,aNuage,Pt2dr(mBoxIn._p0));
           // A priori ces deux la ne servent plus, mais ont ete utiles pour du debugage, on laisse
           mGLOBMasq3D = aMasq3D;
           mGLOBNuage = aNuage;
       }

       for (int aK=0 ; aK<int(mTP3d->size()) ; aK++)
       {
           Pt3dr aPE = (*mTP3d)[aK];
           Pt3dr aPL2 = aNuage->Euclid2ProfPixelAndIndex(aPE);

           int aXIm = round_ni(aPL2.x) - mBoxIn._p0.x;
           int aYIm = round_ni(aPL2.y) - mBoxIn._p0.y;
           int aZIm = round_ni(aPL2.z) ;

           // calclul les derivees par differences finies pour accelerer 
           // la geometrie
           MakeDerivAllGLI(aXIm,aYIm,aZIm);
           // Ajoute le point germe
           CTPAddCell(aMATP,aXIm,aYIm,aZIm,false);

           ShowPoint(Pt2dr(aXIm,aYIm),P8COL::red,0);
       }
   }

   // Fonction qui contient la boucle principale
   OneIterFinaleMATP(aMATP,false);
   // Export ....
   mMMTP->ExportResultInit();
   mMMTP->FreeCel();
 #ifdef ELISE_X11
   if (TheWTiePCor)
   {
       std::cout << "End croissance \n";
       TheWTiePCor->clik_in();
   }
 #endif
   const cComputeAndExportEnveloppe * aCAEE = aMATP.ComputeAndExportEnveloppe().PtrVal();


   if (aMATP.ParamFiltreRegProf().IsInit())
      mMMTP->MaskRegulMaj(aMATP.ParamFiltreRegProf().Val());
   mMMTP->ContAndBoucheTrou();
   if (aMATP.FilterPrgDyn().IsInit())
      mMMTP->MaskProgDyn(aMATP.FilterPrgDyn().Val());


   if (aCAEE)
   {
       mMMTP->ConputeEnveloppe(*aCAEE,aXmlN);
       if (aCAEE->EndAfter().Val()) return;
   }


/*
   if (aMATP.ParamFiltreRegProf().IsInit())
      mMMTP->MaskRegulMaj(aMATP.ParamFiltreRegProf().Val());
   mMMTP->ContAndBoucheTrou();
   if (aMATP.FilterPrgDyn().IsInit())
      mMMTP->MaskProgDyn(aMATP.FilterPrgDyn().Val());
*/



   // A CONSERVER , SAUV FINAL ...:

   std::string aNameMasq =  NameImageMasqOfResol(mCurEtape->DeZoomTer());

   Im2D_Bits<1> aImMasq0 = mMMTP->ImMasqFinal();
   ELISE_COPY(aImMasq0.all_pts(), aImMasq0.in(), Tiff_Im(aNameMasq.c_str()).out());
   
   std::string aNameImage = FullDirMEC() +aXmlN.Image_Profondeur().Val().Image();
   // Pour forcer le resultat flotant 
   Tiff_Im::CreateFromIm(mMMTP->ImProfFinal(),aNameImage.c_str());
/*
   ELISE_COPY(aImProf.all_pts(), aImProf.in(), Tiff_Im(aNameImage.c_str()).out());

       Im2D_REAL4   ImProfFinal() {return  mContBT;}   // image dequant et trous bouches
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
