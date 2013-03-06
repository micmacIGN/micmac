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

void ShowPoint(Pt2dr aP,int aCoul,int aModeCoul)
{
#ifdef ELISE_X11
   if (TheWTiePCor)
   {
       Pt2dr aP0 = aP - Pt2dr(0,0);
       Pt2dr aP1 = aP + Pt2dr(1,1);
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

namespace NS_ParamMICMAC
{



class cCelTiep
{
   public :
      static const INT2  TheNoZ = -32000;
      cCelTiep() :
         mHeapInd  (HEAP_NO_INDEX),
         mZ        (TheNoZ)
      {
              SetCostCorel(4.0);
      }

      void InitCel()
      {
          SetCostCorel(4.0);
          mExploredIndex.clear();
          mZ =  TheNoZ;
      }

      bool ZIsExplored(int aZ)
      {
            return mExploredIndex.find(aZ) != mExploredIndex.end();
      }
      void SetZExplored(int aZ)
      {
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
      bool Inside(const int & anX,const int & anY,const int & aZ)
      {
         return ((anX>=mP0Tiep.x) && (anY>=mP0Tiep.y) && (anX<mP1Tiep.x) && (anY<mP1Tiep.y) && (aZ> cCelTiep::TheNoZ));
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
      void DoMasqAndProfInit(const cMasqueAutoByTieP & aMATP);

       Im2D_Bits<1> ImMasqFinal() {return mImMasqFinal;}
       Im2D_Bits<1> ImMasqInit() {return mImMasqInit;}
       Im2D_INT2    ImProf() {return mImProf;}
       bool InMasqFinal(const Pt2di & aP) const {return  mTImMasqFinal.get(aP);}

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
   mAppli        (anAppli),
   mP0Tiep       (aBox._p0),
   mP1Tiep       (aBox._p1),
   mSzTiep       (aBox.sz()),
   mHeapCTP      (TheCmpCTP),
   mImProf       (mSzTiep.x,mSzTiep.y),
   mTImProf      (mImProf),
   mImMasqInit   (mSzTiep.x,mSzTiep.y),
   mTImMasqInit  (mImMasqInit),
   mImMasqFinal   (mSzTiep.x,mSzTiep.y),
   mTImMasqFinal  (mImMasqFinal)
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
      cResCorTP (double aCSom,double aCMax) :
           mCostSom (aCSom),
           mCostMax (aCMax)
      {
      }
      double  CSom() const {return  mCostSom;}
      double  CMax() const {return  mCostMax;}
    private :
       double mCostSom;
       double mCostMax;
};

cResCorTP cAppliMICMAC::CorrelMasqTP(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ)
{

    std::vector<int> aVOk;
    bool             Ok0 = false;
    for (int aKI=0 ; aKI<mNbIm ; aKI++)
    {
         double aSomIm = 0; 
         double aSomI2 = 0; 
         bool AllOk = true;
         for (int aKS=0 ; aKS<mNbScale ; aKS++)
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
             double aSomP = mVScaIm[0][aKI]->SomPdsMS();
             aSomIm /= aSomP;
             aSomI2 /= aSomP;
             double anEct = aSomI2-ElSquare(aSomIm);
             aSomI2 = sqrt(ElMax(1e0,anEct));
             for (int aKS=0 ; aKS<mNbScale ; aKS++)
             {
                 cGPU_LoadedImGeom * aLI = mVScaIm[aKS][aKI];
                 cStatOneImage * aStat = aLI->VignetteDone();
                 aStat->Normalise(aSomIm,aSomI2);
             }
         }
    }

    if ((! Ok0) || (aVOk.size() < 2))
       return cResCorTP(4,4);

    double aSomDistTot = 0;
    double aMaxDistTot = 0;
    for (int aKK=1 ; aKK<int(aVOk.size()) ; aKK++)
    {
         int aK0 = 0;
         int aK1 = aVOk[aKK];
         double aDistLoc = 0;
         for (int aKS=0 ; aKS<mNbScale ; aKS++)
         {
             cGPU_LoadedImGeom * aLI0 = mVScaIm[aKS][aK0];
             cGPU_LoadedImGeom * aLI1 = mVScaIm[aKS][aK1];
             cStatOneImage * aStat0 = aLI0->VignetteDone();
             cStatOneImage * aStat1 = aLI1->VignetteDone();
 
             aDistLoc += aStat0->SquareDist(*aStat1) * aLI0->PdsMS() ;
         }
         aSomDistTot += aDistLoc;
         aMaxDistTot = ElMax(aMaxDistTot,aDistLoc);
    }

    aSomDistTot /=  (mVScaIm[0][0]->SomPdsMS() * (aVOk.size()-1));
    aMaxDistTot /=  mVScaIm[0][0]->SomPdsMS();

    //  std::cout << "DIISTTT :: " << aMaxDistTot << " " << aSomDistTot << "\n";
    
    return cResCorTP(aSomDistTot,aMaxDistTot);
}

/*
*/

void cAppliMICMAC::CTPAddCell(const cMasqueAutoByTieP & aMATP,int anX,int anY,int aZ,bool Final)
{
   if (!mMMTP->Inside(anX,anY,aZ))
     return;


   if (Final && (! mMMTP->InMasqFinal(Pt2di(anX,anY))))
      return;

   cCelTiep & aCel =  mMMTP->Cel(anX,anY);

   if (aCel.ZIsExplored(aZ)) 
      return;
   aCel.SetZExplored(aZ);

   cResCorTP aCost = CorrelMasqTP(aMATP,anX,anY,aZ) ;
   double aCSom = aCost.CSom();
   if (
         (     (aCSom > aMATP.SeuilSomCostCorrel()) 
            || (aCost.CMax() > aMATP.SeuilMaxCostCorrel()) 
         )
         && (! Final)
      )
   {
      return ;
   }
   if (aCSom < aCel.CostCorel())
   {
        aCel.SetCostCorel(aCSom);
        aCel.SetZ(aZ);
        ShowPoint(Pt2dr(anX,anY),aZ*10,1);
        mMMTP->MajOrAdd(aCel);
   }


  static int aCpt=0; aCpt++;
  if ((aCpt%10000)==0)
     std::cout << "CPT= " << aCpt << "\n";

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
   cCelTiepPtr aCPtr;
   while (mMMTP->PopCel(aCPtr))
   {
        // std::cout << "CCC:: " << aCPtr->CostCorel() << "\n";
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
}




void  cAppliMICMAC::DoMasqueAutoByTieP(const Box2di& aBox,const cMasqueAutoByTieP & aMATP)
{
   ElTimer aChrono;
   mMMTP = new cMMTP(aBox,*this);
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
   }
#endif 
   std::string  aNamePts = mICNM->Assoc1To1
                           (
                              aMATP.KeyImFilePt3D(),
                              PDV1()->Name(),
                              true
                           );
   mTP3d = StdNuage3DFromFile(WorkDir()+aNamePts);


   std::cout << "== cAppliMICMAC::DoMasqueAutoByTieP " << aBox._p0 << " " << aBox._p1 << " Nb=" << mTP3d->size() << "\n"; 
   std::cout << " =NB Im " << mVLI.size() << "\n";


   cXML_ParamNuage3DMaille aXmlN =  mCurEtape->DoRemplitXML_MTD_Nuage();


   cElNuage3DMaille *  aNuage = cElNuage3DMaille::FromParam(aXmlN,FullDirMEC());




   // ElTimer aChrono;
   for (int aK=0 ; aK<int(mTP3d->size()) ; aK++)
   {
       Pt3dr aPE = (*mTP3d)[aK];
       // Pt3dr aPL = aNuage->Euclid2ProfAndIndex(aPE);
       Pt3dr aPL2 = aNuage->Euclid2ProfPixelAndIndex(aPE);

       int aXIm = round_ni(aPL2.x);
       int aYIm = round_ni(aPL2.y);
       int aZIm = round_ni(aPL2.z) ;

       MakeDerivAllGLI(aXIm,aYIm,aZIm);
       CTPAddCell(aMATP,aXIm,aYIm,aZIm,false);

   }


   OneIterFinaleMATP(aMATP,false);
   // std::cout << "TIME CorTP " << aChrono.uval() << "\n";
   std::cout << " XML " << aXmlN.Image_Profondeur().Val().Image() << "\n";

   mMMTP->DoMasqAndProfInit(aMATP);



   Im2D_Bits<1>  aIL =  TestLabel(mMMTP->ImProf(),cCelTiep::TheNoZ);

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
        ELISE_COPY(aIL.all_pts(),aIL.in()+2*mMMTP->ImMasqInit().in(),TheWTiePCor->odisc());

        std::cout << "DONE MASQ !! \n";
        getchar();
        ELISE_COPY(select(aIL.all_pts(),mMMTP->ImMasqFinal().in()),5,TheWTiePCor->odisc());
        getchar();
   }






//  ==================  SAUVEGARDE DES DONNEES POU FAITE UN NUAGE ====================

   std::string aNameMasq = FullDirMEC() +aXmlN.Image_Profondeur().Val().Masq();
   ELISE_COPY
   (
        mMMTP->ImMasqFinal().all_pts(),
        mMMTP->ImMasqFinal().in(),
        Tiff_Im(aNameMasq.c_str()).out()
   );

   
   std::string aNameImage = FullDirMEC() +aXmlN.Image_Profondeur().Val().Image();
   ELISE_COPY
   (
        mMMTP->ImProf().all_pts(),
        mMMTP->ImProf().in(),
        Tiff_Im(aNameImage.c_str()).out()
   );
   //cElNuage3DMaille * aNuage = ;


   std::cout << "SAUV DONNEE " << aChrono.uval() << "\n";
   getchar();
}






};

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
