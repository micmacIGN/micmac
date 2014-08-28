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
// #include "anag_all.h"

/*
void f()
{
    FILE * aFP = ElFopen(MMC,"w");
    ElFclose(aFP);
}

*/


#include "StdAfx.h"
#include "hassan/reechantillonnage.h"


#if (ELISE_X11)




#if (0)
#endif



// To put in bench file

void Bench_Rank()
{
    std::cout << "Begin Bench Rank \n";
   
    for (int aTime=0 ; aTime<10000; aTime++)
    {
        int aNb = round_ni(1 + ElSquare(10*NRrandom3()));
        std::vector<double> aV;
        for (int aK=0 ; aK<aNb ; aK++)
           aV.push_back(NRrandC());

        for (int aK=0 ; aK < aNb ; aK++)
        {
            for (int aK2=0 ; aK2 < 3 ; aK2++)
                if (NRrandom3()<0.2)
                   aV.push_back(aV[aK]);

            for (int aK2=0 ; aK2 < int(aV.size()) ; aK2++)
                if (NRrandom3()<0.02)
                   aV[aK2] = aV[aK];
         }

        aNb = aV.size();

        std::vector<double>  aV2 = aV;
        std::vector<double>  aV3 = aV;

         int aRnk = NRrandom3(aNb);

         double aVK =KthVal(VData(aV),aNb,aRnk);

         std::sort(aV2.begin(),aV2.end());
         double aVK2 = aV2[aRnk];

         // std::cout << "Bench Rank " << aVK-aVK2 << "\n";
         ELISE_ASSERT(ElAbs(aVK-aVK2)<1e-10,"Bench rnk");

/*
   Ne marche pas : la valeur RrnK est n'importe ou

         SplitArrounKthValue(VData(aV3),aNb,aRnk);
         double aVK3 = aV3[aRnk];
         std::cout << "Bench Rank " << aVK-aVK2 << " " << aVK-aVK3<< "\n";
         ELISE_ASSERT(ElAbs(aVK-aVK2)<1e-10,"Bench rnk");
         ELISE_ASSERT(ElAbs(aVK-aVK3)<1e-10,"Bench rnk");
*/

    }
    std::cout << "OK BENCH RANK \n";
}




Fonc_Num Correl(Fonc_Num aF1,Fonc_Num aF2,int aNb)
{
   Symb_FNum aM1 (Moy(aF1,aNb));
   Symb_FNum aM2 (Moy(aF2,aNb));

   Fonc_Num aEnct1 = Moy(Square(aF1),aNb) -Square(aM1);
   Fonc_Num aEnct2 = Moy(Square(aF2),aNb) -Square(aM2);


   return (Moy(aF1*aF2,aNb)  -aM1*aM2) / sqrt(Max(1e-5,aEnct1*aEnct2));
}


void AutoCorrel(const std::string & aName)
{
   Tiff_Im aTF(aName.c_str());
   Pt2di aSz = aTF.sz();
   Im2D_REAL4 anI(aSz.x,aSz.y);
   ELISE_COPY(aTF.all_pts(),aTF.in(),anI.out());

   int aNb = 2;

   Fonc_Num aF = 1.0;
   for (int aK=0 ; aK<4 ; aK++)
   {
      aF = Min(aF,Correl(anI.in(0),trans(anI.in(0),TAB_4_NEIGH[aK])*(aNb*2),aNb));
   }

   Tiff_Im::Create8BFromFonc
   (
       StdPrefix(aName)+"_AutoCor.tif",
       aSz,
       Min(255,Max(0,(1+aF)*128))
   );
}


Im2D_REAL4 Conv2Float(Im2DGen anI)
{
   Pt2di aSz = anI.sz();
   Im2D_REAL4 aRes(aSz.x,aSz.y);
   ELISE_COPY(anI.all_pts(),anI.in(),aRes.out());
   return aRes;
}




void TestKL()
{
   Pt2di aSZ(200,200);
   Im2D_Bits<1> aImMasqF(aSZ.x,aSZ.y,1);

   Im2D_Bits<1> aImMasqDef(aSZ.x,aSZ.y,1);
   ELISE_COPY(rectangle(Pt2di(70,0),Pt2di(130,200)),0,aImMasqDef.out());

   Im2D<U_INT2,INT> aImVal(aSZ.x,aSZ.y);
   ELISE_COPY(aImVal.all_pts(),FX,aImVal.out());

   Video_Win aW=Video_Win::WStd(aSZ,3.0);
   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());
   ELISE_COPY(aW.all_pts(),aImMasqDef.in(),aW.odisc());
   getchar();


   aImVal = ImpaintL2(aImMasqDef,aImMasqF,aImVal);

   // NComplKLipsParLBas(aImMasqDef,aImMasqF,aImVal,1.0);

   ELISE_COPY(aW.all_pts(),aImVal.in(),aW.ogray());

   Tiff_Im::Create8BFromFonc("toto.tif",aSZ,aImVal.in());
   getchar();
}
#if (0)
#endif


extern void TestDigeoExt();


void TestXMLNuageNodData()
{
    std::string aN1 = "/media/data2/Aerien/Euro-SDR/Munich/Cloud-Fusion/CF-42_0502_PAN.xml";
    std::string aN2 = "/media/data2/Aerien/Euro-SDR/Munich/MEC2Im-true-42_0502_PAN.tif-41_0420_PAN.tif/NuageImProf_LeChantier_Etape_8.xml";
    for (int aK=0 ; aK<1000 ; aK++)
    {
       cElNuage3DMaille * aC1 =  NuageWithoutData(aN1);
       // cElNuage3DMaille * aC2 =  NuageWithoutData(aN2);
       cElNuage3DMaille * aC2 =  NuageWithoutDataWithModel(aN2,aN1);
       cElNuage3DMaille * aC3 =  NuageWithoutData(aN2);
       std::cout << "C1= " << aC1->SzData()  << " " << aC2->SzData()  << "\n";
       std::cout << "C1= " << aC1->SzGeom()  << " " << aC2->SzGeom()  << "\n";
       std::cout << "COMPAT " <<   GeomCompatForte(aC1,aC2) << "\n";
       std::cout << "COMPAT " <<   GeomCompatForte(aC1,aC3) << "\n";
       // std::cout << "C1= " << aC1->SzUnique()  << " " << aC2->SzUnique()  << "\n";
       getchar();
       delete aC1;
    }
}

void TestRound()
{
   while(1)
   {
       double aV,aBig;
       cin >> aV >> aBig;
       cDecimal aD  = StdRound(aV);
       double Arrond = aD.Arrondi(aBig);
       printf("%9.9f %9.9f\n",aD.RVal(),Arrond);
       std::cout << "Round " << aD.RVal() << "\n";
   }
}



void Test_Arrondi_LG()
{
    Pt2di aSz(100,100);
    double aVTest = 117;

    Im2D_REAL16 anIm(aSz.x,aSz.y);
    TIm2D<REAL16,REAL16> aTIm(anIm);

    ELISE_COPY(anIm.all_pts(),aVTest,anIm.out());

    while (1)
    {
         Pt2dr aP0 = Pt2dr(10,10) + Pt2dr(NRrandom3(),NRrandom3()) *50.123456701765;
         double aV0 = aTIm.getr(aP0);
         double aV1 = Reechantillonnage::biline(anIm.data(),aSz.x,aSz.y,aP0);

         std::cout << " TEST " << (aV0-aVTest) * 1e50 << " " << (aV1-aVTest) * 1e50  << " " << aP0 << "\n";
         getchar();
    }
}



void PbHom(const std::string & anOri)
{
   const std::string & aDir = "/media/data1/Calib-Sony/FacadePlane-2000/";
   const std::string & aIm1 = "DSC05180.ARW";
   const std::string & aIm2 = "DSC05182.ARW";


   cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
   std::string    aKeyOri =  "NKS-Assoc-Im2Orient@-" + anOri;


    std::string aNameOri1 =  aICNM->Assoc1To1(aKeyOri,aIm1,true);
    std::string aNameOri2 =  aICNM->Assoc1To1(aKeyOri,aIm2,true);


    CamStenope * aCS1 = CamOrientGenFromFile(aNameOri1,aICNM);
    CamStenope * aCS2 = CamOrientGenFromFile(aNameOri2,aICNM);

    Pt2dr aP1 (774,443);
    Pt2dr aP2 (5541,3758);

    Pt3dr aTer1  = aCS1->ImEtProf2Terrain(aP1,1.0);
    Pt2dr aProj1 = aCS1->R3toF2(aTer1);

    std::cout << "P & Proj Init1" << aP1 << aProj1 << " " << euclid(aP1-aProj1) << "\n";


    Pt3dr aTer2  = aCS2->ImEtProf2Terrain(aP2,1.0);
    Pt2dr aProj2 = aCS2->R3toF2(aTer2);

    std::cout << "P & Proj Init2" << aP2 << aProj2 << " " << euclid(aP2-aProj2) << "\n";


    double aDist;
    Pt3dr aTerInter = aCS1->PseudoInter(aP1,*aCS2,aP2,&aDist);

    aProj1 = aCS1->R3toF2(aTerInter);
    aProj2 = aCS2->R3toF2(aTerInter);

    std::cout << "Proj Inter " << aDist << " " << (aP1-aProj1) << " " << (aP2-aProj2) << "\n";



    std::cout << "\n";
}

void DebugDrag()
{
   std::string aDir = "/media/data1/Jeux-Tests/Dragon-2/MEC2Im-Epi_Im1_Right_IMGP7511_IMGP7512.tif-Epi_Im2_Left_IMGP7511_IMGP7512.tif/";
   std::string aNamePx = "Px1_Num6_DeZoom2_LeChantier.tif";
   std::string aNameMasq = "AutoMask_LeChantier_Num_5.tif";

   Tiff_Im aTP = Tiff_Im::StdConv(aDir+aNamePx);
   Tiff_Im aTM = Tiff_Im::StdConv(aDir+aNameMasq);

   double aMinPx;
   ELISE_COPY
   (
        aTP.all_pts(),
        aTP.in() * aTM.in(),
        VMin(aMinPx)
   );

   std::cout << "MIN PX " << aMinPx << "\n";
}

// extern  void F2Test();
// extern  void Ftest();

void TestRandomSetOfMesureSegDr()
{
    std::string aInput="/media/data1/ExempleDoc/Test-CompDrAnalogik/MesureLineImageOri.xml";
    std::string aOut="/media/data1/ExempleDoc/Test-CompDrAnalogik/MesureLineImage.xml";


    cSetOfMesureSegDr aSMS = StdGetFromPCP(aInput,SetOfMesureSegDr);

   for
   (
       std::list<cMesureAppuiSegDr1Im>::iterator itIm=aSMS.MesureAppuiSegDr1Im().begin();
       itIm!=aSMS.MesureAppuiSegDr1Im().end();
       itIm++
   )
   {
      std::string aNameIm = itIm->NameIm();
      {

         for
         (
            std::list<cOneMesureSegDr>::iterator itMes=itIm->OneMesureSegDr().begin();
            itMes!=itIm->OneMesureSegDr().end();
            itMes++
         )
         {
             Pt2dr aP1 = itMes->Pt1Im();
             Pt2dr aP2 = itMes->Pt2Im();
             SegComp aSeg(aP1,aP2);
             itMes->Pt1Im() = aSeg.from_rep_loc(Pt2dr(0.6+NRrandC(),0));
             itMes->Pt2Im() = aSeg.from_rep_loc(Pt2dr(0.4-NRrandC(),0));
         }
      }
   }

    MakeFileXML(aSMS,aOut);

    exit(0);
}


template <class T1,class T2,class Action> int BugOneZC
                                 (
                                      const Pt2di & aPGerm, bool V4,
                                      T1 & aIm1,int aV1Sel,int aV1Aff,
                                      T2 & aIm2,int aV2Sel,
                                      Action & aOnNewPt
                                 )
{
   Pt2di * aTabV = V4 ? TAB_4_NEIGH : TAB_8_NEIGH ;
   int aNbV = V4 ? 4 : 8;

   std::vector<Pt2di>  aVec1;
   std::vector<Pt2di>  aVec2;

   std::vector<Pt2di> * aVCur = &aVec1;
   std::vector<Pt2di> * aVNext = &aVec2;

   if ((aIm1.get(aPGerm)==aV1Sel) && (aIm2.get(aPGerm)==aV2Sel))
   {
      aIm1.oset(aPGerm,aV1Aff);
      aVCur->push_back(aPGerm);
      aOnNewPt.OnNewPt(aPGerm);
   }
   int aNbStep = 1;

  int aNbTot = 0;
   while (! aVCur->empty())
   {
       aOnNewPt.OnNewStep();
       if (aOnNewPt.StopCondStep())
          return aNbTot;
       int aNbCur = aVCur->size();
       aNbTot += aNbCur;

       for (int aKp=0 ; aKp<aNbCur ; aKp++)
       {
           Pt2di aP = (*aVCur)[aKp];
           for (int aKv=0; aKv<aNbV ; aKv++)
           {
                 Pt2di aPV = aP+aTabV[aKv];
                 if ((aIm1.get(aPV)==aV1Sel) && (aIm2.get(aPV)==aV2Sel))
                 {
                    aIm1.oset(aPV,aV1Aff);
                    aVNext->push_back(aPV);
                    aOnNewPt.OnNewPt(aPV);
                 }
           }
       }

       ElSwap(aVNext,aVCur);
       aVNext->clear();
       aNbStep++;
   }

   return aNbTot;
}




class cCC_NbMaxIter : public  cCC_NoActionOnNewPt
{
   public  :
       cCC_NbMaxIter(int aNbMax) :
          mNbIter (0),
          mNbMaxIter (aNbMax)
       {
       }


       void OnNewStep() { mNbIter++;}
       void  OnNewPt(const Pt2di & aP) 
       {
           mVPts.push_back(aP);
       }
       bool  StopCondStep() {return mNbIter>=mNbMaxIter;}

    
       std::vector<Pt2di> mVPts;
       int                mNbIter;
       int                mNbMaxIter;
};

class cParamFiltreDetecRegulProf
{
    public :
      cParamFiltreDetecRegulProf()  :
         mSzCC (2),
         mPondZ (2.0),
         mPente (0.5),
         mSeuilReg (0.5),
         mV4    (false),
         mNbCCInit  (5),
         mNameTest  ("Test.tif")
      {
      }
      int  SzCC() const {return mSzCC;}
      double PondZ() const {return  mPondZ;}
      double Pente() const {return  mPente;}
      double SeuilReg() const {return  mSeuilReg;}
      bool V4() const {return mV4;}
      bool NbCCInit() const {return mNbCCInit;}
      const std::string & NameTest() const {return mNameTest;}
    private :
        int mSzCC;
        double mPondZ;
        double mPente;
        double mSeuilReg;
        bool   mV4;
        int    mNbCCInit;
        std::string mNameTest;
};


template <class tNum,class tNBase>  Im2D_Bits<1> TplFiltreDetecRegulProf
                                        (
                                             TIm2D<tNum,tNBase> aTProf, 
                                             TIm2DBits<1>  aTMasq,
                                             const cParamFiltreDetecRegulProf & aParam
                                        )
{
    FiltrageCardCC(true,aTMasq,1,0, aParam.NbCCInit());

    Pt2di aSz = aTProf.sz();
    Im2D_Bits<1> aIMasq = aTMasq._the_im;

    Im2D_Bits<1> aMasqTmp = ImMarqueurCC(aSz);
    TIm2DBits<1> aTMasqTmp(aMasqTmp);
    bool V4= aParam.V4();

    Im2D_REAL4 aImDif(aSz.x,aSz.y);
    TIm2D<REAL4,REAL8> aTDif(aImDif);

    ELISE_COPY(aIMasq.border(1),0,aIMasq.out());

    Pt2di aP;
    int aSzCC = aParam.SzCC();
    for (aP.x =0 ; aP.x < aSz.x ; aP.x++)
    {
        for (aP.y =0 ; aP.y < aSz.y ; aP.y++)
        {
             if (aTMasq.get(aP))
             {
                 cCC_NbMaxIter aCCParam(aSzCC);
                 BugOneZC(aP,V4,aTMasqTmp,1,0,aTMasq,1,aCCParam);
                 tNBase aZ0 =  aTProf.get(aP);
                 double aSomP = 0;
                 int    aNbP  = 0;
                 for (int aKP=0 ; aKP<int(aCCParam.mVPts.size()) ; aKP++)
                 {
                     const Pt2di & aQ = aCCParam.mVPts[aKP];
                     aTMasqTmp.oset(aQ,1);
                     if (aKP>0)
                     {
                         double aDist = euclid(aP,aQ);
                         double aDZ = ElAbs(aZ0-aTProf.get(aQ));
                         double aAttZ = aParam.PondZ() + aParam.Pente() * aDist;
                         double aPds  = 1 / (1 + ElSquare(aDZ/aAttZ));
                         aNbP++;
                         aSomP += aPds;
                     }
                 }
                 aNbP = ElMax(aNbP,aSzCC*(1+aSzCC));
                 aTDif.oset(aP,aSomP/aNbP);
             }
        }
    }
    if (aParam.NameTest()!="")
    {
       Tiff_Im::Create8BFromFonc(aParam.NameTest(),aSz,aImDif.in()*255);
    }

    Im2D_Bits<1> aIResult(aSz.x,aSz.y);
    ELISE_COPY(aIResult.all_pts(),(aImDif.in()> aParam.SeuilReg()) && (aIMasq.in()) , aIResult.out());
    return aIResult;
    
}

Im2D_Bits<1>  FiltreDetecRegulProf(Im2D_REAL4 aImProf,Im2D_Bits<1> aIMasq,const cParamFiltreDetecRegulProf & aParam)
{
   return TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aIMasq),aParam);
}

Im2D_Bits<1> FiltreDetecRegulProf(Im2D_REAL4 aImProf,Im2D_Bits<1> aIMasq)
{
   cParamFiltreDetecRegulProf aParam;
   return FiltreDetecRegulProf(aImProf,aIMasq,aParam);
}

void TestFiltreRegul()
{
   Pt2di aP0(2000,500);
   Pt2di aSz(500,500);

  // Video_Win * aW = Video_Win::PtrWStd(aSz*2,true,Pt2dr(2,2));
   Video_Win * aW = 0;


   Tiff_Im aFileProf ("/home/marc/TMP/EPI/EXO1-Fontaine/MTD-Image-CIMG_2489.JPG/Fusion_NuageImProf_LeChantier_Etape_1.tif");
   Tiff_Im aFileMasq ("/home/marc/TMP/EPI/EXO1-Fontaine/MTD-Image-CIMG_2489.JPG/Fusion_NuageImProf_LeChantier_Etape_1_Masq.tif");

   Im2D_REAL4    aImProf(aSz.x,aSz.y);
   Im2D_Bits<1>  aMasq(aSz.x,aSz.y);


   ELISE_COPY(aImProf.all_pts(),trans(aFileProf.in(0),aP0),aImProf.out());
   ELISE_COPY(aMasq.all_pts(),trans(aFileMasq.in(0),aP0),aMasq.out());

   if (aW)
   {
       ELISE_COPY(aImProf.all_pts(),aImProf.in()*5,aW->ocirc());
       ELISE_COPY(select(aMasq.all_pts(),!aMasq.in()),P8COL::black,aW->odisc());
   }

   cParamFiltreDetecRegulProf aParam;
   //TplFiltreDetecRegulProf(TIm2D<REAL4,REAL8>(aImProf),TIm2DBits<1>(aMasq),aParam);
std::cout << "AAAaaaA\n";
   FiltreDetecRegulProf(aImProf,aMasq,aParam);
std::cout << "BBBbBb\n";
getchar();
}


// cConvExplicite GlobMakeExplicite(eConventionsOrientation aConv);


int MPDtest_main (int argc,char** argv)
{
    TestFiltreRegul();
  
   return 0;

}

#endif

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
