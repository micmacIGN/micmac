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


// Test Git


#include "StdAfx.h"
#include "hassan/reechantillonnage.h"

#include "../uti_phgrm/NewOri/NewOri.h"


#if (ELISE_X11)




/*
template <class TVal,class tFVal,class tFPds> 
         void RecursWeightedSplitArrounKthValue(TVal * Data,const tFPds & aFVal,const tFPds & aFPds,int aNb,U_INT8 aPds)
{
   if (aNb==0) return;
   if (aPds<=0) return;
 // std::cout << " SplitArrounKthValue " << aNb << " " << aKth << "\n";
   // On calcule la moyenne
   TVal aMoy(0);
   U_INT8 aSomP=0;

   for (int aKv=0 ; aKv<aNb ; aKv++)
   {
      U_INT8 aCurPds = aFPds(Data[aKv]);
      aMoy = aMoy+FVal(Data[aKv]) * aCurPds;
      aSomP += aCurPds;
   }
   if (aPds>=aSomP) return;

   aMoy = aMoy / aSomP;

   // On permut de maniere a ce que les valeur du debut soit < Moy  et celle de la fin >=Moy
   int aK0 =0;
   int aK1 = aNb-1;
   U_INT8 aP0Moins = 0;
   U_INT8 aP0Plus  = 0;
   while (aK0 < aK1)
   {
        while ((aK0<aK1) && (aFVal(Data[aK0]) <  aMoy))
        {
            aP0Moins += aFPds(Data[aK0]);
            aK0++;
        }
        while ((aK0<aK1) && (aFVal(Data[aK1]) >= aMoy))
        {
            aP0Plus += aFPds(Data[aK1]);
            aK1--;
        }
        if (aK0 < aK1) 
        {
           ElSwap(Data[aK0],Data[aK1]);
        }
   }
   ELISE_ASSERT(aK0==aK1,"Verif in SplitArrounKthValue");
   ELISE_ASSERT((aP0Moins+aP0Plus)==aSomP,"Verif in SplitArrounKthValue");

   // Si le cas, on n'a pas progresse, toute les valeur sont egale
   if  (aK0==0)
   {
       return;
   }

   if (aK0 == aKth)  
   {
      return;
   }

   if (aK0 < aKth)
   {
      RecursWeightedSplitArrounKthValue(Data+aK0,aNb-aK0,aKth-aK0);
   }
   else           
   {
      RecursWeightedSplitArrounKthValue(Data,aK0,aKth);
   }
}
*/


/*
template <class TVal> void WeightedSplitArrounKthValue(TVal * Data,int aNb,int aKth)
{
}

*/














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

void FiltreRemoveBorderHeter(Im2D_REAL4 anIm,Im2D_U_INT1 aImMasq,double aCostRegul,double aCostTrans)
{
    Pt2di aSz = anIm.sz();
    double aVMax,aVMin;

    ELISE_COPY(aImMasq.border(1),0,aImMasq.out());
    ELISE_COPY(aImMasq.all_pts(),aImMasq.in()!=0,aImMasq.out());
    ELISE_COPY(anIm.all_pts(),anIm.in(),VMax(aVMax)|VMin(aVMin));
    Video_Win * aW = Video_Win::PtrWStd(aSz);
    ELISE_COPY(anIm.all_pts(),(anIm.in()-aVMin) * (255.0/(aVMax-aVMin)),aW->ogray());
    std::cout << "VMAX " << aVMax << "\n";

    //ELISE_COPY(aW->all_pts(),aImMasq.in(),aW->odisc());
    //aW->clik_in();

    ELISE_COPY
    (
          aW->all_pts(),
          nflag_close_sym(flag_front8(aImMasq.in_proj()!=0)),
          aW->out_graph(Line_St(aW->pdisc()(P8COL::red)))
    );

    cParamFiltreDepthByPrgDyn aParam =  StdGetFromSI(Basic_XML_MM_File("DefFiltrPrgDyn.xml"),ParamFiltreDepthByPrgDyn);
    aParam.CostTrans() = aCostTrans;
    aParam.CostRegul() = aCostRegul;

    Im2D_Bits<1>  aNewMasq =  FiltrageDepthByProgDyn(anIm,aImMasq,aParam);
  
    ELISE_COPY
    (
         select(aNewMasq.all_pts(),aNewMasq.in()),
         2,
         aImMasq.out()
    );
    TIm2D<U_INT1,INT> aTMasq(aImMasq);
    FiltrageCardCC(false,aTMasq,2,0,100);

    Neighbourhood aNV4=Neighbourhood::v4();
    Neigh_Rel     aNrV4 (aNV4);

    ELISE_COPY
    (
           conc
           (
               select(select(aImMasq.all_pts(),aImMasq.in()==1),aNrV4.red_sum(aImMasq.in()==0)),
               aImMasq.neigh_test_and_set(aNV4,1,0,256)
           ),
           3,
           Output::onul()
    );



    ELISE_COPY
    (
         aNewMasq.all_pts(),
         aImMasq.in(),
         aW->odisc()
    );

/*
    ELISE_COPY
    (
          aW->all_pts(),
          nflag_close_sym(flag_front8(aNewMasq.in_proj())),
          aW->out_graph(Line_St(aW->pdisc()(P8COL::green)))
    );
*/


    aW->clik_in();

}


void FiltreRemoveFlou(const std::string & aNameIm,const std::string & aNameMasq)
{
    std::cout << "NameIm= " << aNameIm << "\n";
    Im2D_REAL4 anIm = Im2D_REAL4::FromFileStd(aNameIm);
    Im2D_U_INT1 aImMasq = Im2D_U_INT1::FromFileStd(aNameMasq);
    FiltreRemoveBorderHeter(anIm,aImMasq,1.0,10.0);
}





extern void TestFiltreRegul();
#if ELISE_QT
extern void Test3dQT();
#endif

extern Fonc_Num sobel(Fonc_Num f);

void SobelTestNtt(const std::string &aName)
{
    Tiff_Im aTF = Tiff_Im::StdConvGen(aName,1,true);

    Pt2di aSz = aTF.sz();
    Im2D_REAL4 aI0(aSz.x,aSz.y);
    ELISE_COPY( aTF.all_pts(),aTF.in(),aI0.out());

    Video_Win * aW=0;
    // aW = Video_Win::PtrWStd(aSz);

    if (aW)
    {
        ELISE_COPY(aW->all_pts(),Min(255,aI0.in()/256.0),aW->ogray());
        aW->clik_in();
    }
  
    Fonc_Num aF1 = sobel(aI0.in_proj());

    Fonc_Num aF2 = aI0.in_proj();
    for (int aK=0 ; aK<3 ; aK++)
        aF2 = rect_som(aF2,1) / 9.0;
    aF2 = sobel(aF2);

    if (aW)
    {
        ELISE_COPY(aW->all_pts(),Min(255, 200 * (aF1/Max(aF2,1e-7))),aW->ogray());
        aW->clik_in();
    }

    double aSF1,aSF2,aSomPts;
    ELISE_COPY(aI0.all_pts(),Virgule(aF1,aF2,1.0),Virgule(sigma(aSF1),sigma(aSF2),sigma(aSomPts)));

    std::cout << "Indice " << aSF1 / aSF2 << "\n";
  
}

void TestNtt(const std::string &aName)
{
    Tiff_Im aTF = Tiff_Im::StdConvGen(aName,1,true);

    Pt2di aSz = aTF.sz();
    Im2D_REAL4 aI0(aSz.x,aSz.y);
    ELISE_COPY( aTF.all_pts(),aTF.in(),aI0.out());

    int aWSz=2;

    TIm2D<REAL4,REAL8> aTIm(aI0);

     double aSomGlob=0.0;
     double aNbGlob=0.0;

     for (int aKdx=-aWSz ; aKdx<=aWSz ; aKdx+=aWSz)
     {
         printf("## ");
         for (int aKdy=-aWSz ; aKdy<=aWSz ; aKdy+=aWSz)
         {
             int aDx = aKdx;
             int aDy = aKdy;
             Pt2di aDep(aDx,aDy);
             Pt2di aP;
             RMat_Inertie aMat;
             for (aP.x = aWSz ; aP.x<aSz.x-aWSz ; aP.x++)
             {
                 for (aP.y=aWSz ; aP.y<aSz.y-aWSz ; aP.y++)
                 {
                      aMat.add_pt_en_place(aTIm.get(aP),aTIm.get(aP+aDep));
                 }
             }
             double aC = aMat.correlation();
             aC = 1-aC;
             if (dist8(aDep) == aWSz)
             {
                aSomGlob += aC;
                aNbGlob ++;
             }
             printf(" %4d",round_ni(10000*(aC)));
         }
         printf("\n");
     }
     aSomGlob /= aNbGlob;
     std::cout  <<  " G:" << aSomGlob << "\n";
     printf("\n\n");
  
}








extern void getKeypointFilename( const string &i_basename, int i_resolution, string &o_keypointsName );



int Jeremy_main( int argc, char **argv )
{
    if ( argc<2 ) return EXIT_FAILURE;

    Tiff_Im tiff(argv[1]);
    cout << '[' << argv[1] << "]: sz = " << tiff.sz() << 'x' << tiff.nb_chan() << ' ' << eToString(tiff.type_el()) << endl;
    Im2DGen image = tiff.ReadIm();
    cout << '[' << argv[1] << "]: sz = " << image.sz() << ' ' << eToString(image.TypeEl()) << endl;

    ELISE_COPY
    (
        image.all_pts(),
        Virgule( image.in(), image.in(), image.in() ),
        Tiff_Im(
            "toto.tif",
            image.sz(),
            image.TypeEl(),
            Tiff_Im::No_Compr,
            Tiff_Im::RGB,
            ArgOpTiffMDP(argv[1])/*Tiff_Im::Empty_ARG*/ ).out()
    );

    return EXIT_SUCCESS;
}


/*
void LoadTrScaleRotate
     (
          const std::string & aNameIn,
          const std::string & aNameOut,
          const Pt2di & aP1Int,
          const Pt2di & aP2Int,
          const Pt2di & aP1Out,
          double      aScale,  // Par ex 2 pour image 2 fois + petite
          int         aRot
     )
{
     Tiff_Im aTifIn(aNameIn.c_str());
     Tiff_Im aTifOut(aNameOut.c_str());

     int aNbCh = aTifIn.nb_chan();
     ELISE_ASSERT(aTifOut.nb_chan()==aNbCh,"LoadTrScaleRotate nb channel diff");


     Pt2dr aVIn  = Pt2dr(aP2Int-aP1Int);
     Pt2di aSzOutInit = round_ni(aVIn / aScale);

     std::vector<Im2DGen *>   aVOutInit = aTifOut.VecOfIm(aSzOutInit);
     
     ELISE_COPY
     (
          aVOutInit[0]->all_pts(),
          StdFoncChScale(aTifIn.in_proj(),Pt2dr(aP1Int),Pt2dr(aScale,aScale)),
          StdOut(aVOutInit)
     );

     std::vector<Im2DGen *>   aVOutRotate;
     for (int aK=0 ; aK<int(aVOutInit.size()) ; aK++)
          aVOutRotate.push_back(aVOutInit[aK]->ImRotate(aRot));

     Pt2di aSzOutRotat = aVOutRotate[0]->sz();


     ELISE_COPY
     (
         rectangle(aP1Out,aP1Out+aSzOutRotat),
         trans(StdInput(aVOutRotate), -aP1Out),
         aTifOut.out()
     );
}
*/

extern void TestOriBundle();
extern void TestSVD3x3();
extern void Bench_NewOri();

void BenchSort3()
{
   for (int aK=0 ; aK<1000000 ; aK++)
   {
       int Tab[3];
       for (int aK=0 ; aK<3 ; aK++)
           Tab[aK] = NRrandom3(4);

       int aRnk[3]; 
       Rank3(aRnk,Tab[0],Tab[1],Tab[2]);
       int Sort[3];
      
       for (int aK=0 ; aK<3 ; aK++)
           Sort[aRnk[aK]] = Tab[aK];
       ELISE_ASSERT(Sort[0] <= Sort[1],"BenchSort3");
       ELISE_ASSERT(Sort[1] <= Sort[2],"BenchSort3");

       cTplTriplet<int> aTT(Tab[0],Tab[1],Tab[2]);
       ELISE_ASSERT(aTT.mV0 <= aTT.mV1,"BenchSort3");
       ELISE_ASSERT(aTT.mV1 <= aTT.mV2,"BenchSort3");

       cTplTripletByRef<int> aPtrTT(Tab[0],Tab[1],Tab[2]);
       ELISE_ASSERT(*aPtrTT.mV0 <= *aPtrTT.mV1,"BenchSort3");
       ELISE_ASSERT(*aPtrTT.mV1 <= *aPtrTT.mV2,"BenchSort3");
   }
   std::cout << "DONE BenchSort3\n";
}

void PartitionRenato(int argc,char** argv)
{
    std::string aName;
    double  aPropSzW=0.1,aSeuil=75;
    double aPropExag = 0.1;
    int aNbIter = 3;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aName,"Name Input"),
        LArgMain()  <<  EAM(aPropSzW,"PropSzW",true,"Prop Size of W, def =0.1")
                     <<  EAM(aSeuil,"Seuil",true,"Threshold beetween Black & White, Def=75")
    );
    

    Tiff_Im aTIn = Tiff_Im::UnivConvStd(aName);
    Pt2di aSz = aTIn.sz();
    int aSzW = round_ni((euclid(aSz)*aPropSzW) / sqrt(aNbIter));
    Im2D_REAL4 anIm0(aSz.x,aSz.y);
    Im2D_REAL4 anIm1(aSz.x,aSz.y);
    Im2D_U_INT1 aImInside(aSz.x,aSz.y,1);

    ELISE_COPY(anIm0.all_pts(),255-aTIn.in(),anIm0.out());

    int aNbF = 3;
    for (int aKF=0 ; aKF<aNbF ; aKF++)
    {
        Im2D_REAL4 anImFond(aSz.x,aSz.y);
        Fonc_Num aFIn = anIm0.in(0);
        for (int aK=0 ; aK<aNbIter ; aK++)
           aFIn = (rect_som(aFIn,aSzW)*aImInside.in(0)) / Max(1.0,rect_som(aImInside.in(0),aSzW));

       ELISE_COPY(anImFond.all_pts(),aFIn,anImFond.out());
       if (aKF == (aNbF-1))
       {
              Fonc_Num aF = anIm0.in()-anImFond.in();
              aF = aF / aSeuil;
              aF = (aF -0.1) / (1-2*aPropExag);
              aF = Max(0.0,Min(1.0,aF));
              ELISE_COPY(anIm1.all_pts(),255.0 *(1-aF),anIm1.out());
       }
       else
       {
            ELISE_COPY
            (
                 aImInside.all_pts(),
                 anIm0.in() < anImFond.in()+aSeuil,
                 aImInside.out()
            );
       }
       
    }

    Tiff_Im::Create8BFromFonc(std::string("Bin-")+StdPrefix(aName)+".tif",aTIn.sz(),anIm1.in());
}



void PdBump()
{
   std::string aPref = "/home/mpd/MMM/culture3d/Documentation/OriRel-IMGP0492.JPG.xml";

   cXml_Ori2Im  aXmlOri = StdGetFromSI(aPref,Xml_Ori2Im);

}


void TestMax(int argc,char** argv)
{
    std::string aNameIm; 
    int aSzW = 32;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aNameIm,"Name Input"),
        LArgMain() <<  EAM(aSzW,"SzW",true,"Taille de la fenetre")
    );

    Tiff_Im aTIn = Tiff_Im::StdConvGen(aNameIm,1,true);
    Pt2di aSz = aTIn.sz();

    Im2D_U_INT2 aIm(aSz.x,aSz.y);
    ELISE_COPY(aTIn.all_pts(),aTIn.in(),aIm.out());


    Im2D_U_INT1 aImMax(aSz.x,aSz.y,0);
    Im2D_U_INT1 aImMin(aSz.x,aSz.y,0);
    

    Fonc_Num aF = aIm.in(-10000) + FX/100.0 + FY/1000.0;

    ELISE_COPY(aImMax.all_pts(),aF==rect_max(aF,aSzW),aImMax.out());

     
    Tiff_Im aTIn8B = Tiff_Im::StdConvGen(aNameIm,1,false);


    std::string aNameMax ="Max-" + aNameIm;
    Tiff_Im aResMax
            (
                aNameMax.c_str(),
                aSz,
                GenIm::u_int1,
                Tiff_Im::No_Compr,
                Tiff_Im::RGB
            );

   ELISE_COPY
   (
        aResMax.all_pts(),
        Virgule
        (
            aTIn8B.in(),
            aTIn8B.in(),
            255*dilat_32(aImMax.in(0),5)
        ),
        aResMax.out()
   );


}

extern void TestBGC3M2D();

void TestGridCam()
{
    /// std::string aNameCam = "/home/marc/TMP/VolSozo2/Ori-MEP/Orientation-IMG_1577.CR2.xml";

    std::string aDir = "/home/marc/TMP/VolSozo2/";
    std::string aNameIm =  "IMG_1577.CR2";
    std::string aNameOri =  "MEP";
    cInterfChantierNameManipulateur*  anICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::string aFullName = anICNM->NameOriStenope(aNameOri,aNameIm);

    CamStenope * aCS0 =   anICNM->StdCamStenOfNames(aNameIm,aNameOri);
    CamStenope *aCSGrid =   CamStenope::StdCamFromFile(true,aFullName,anICNM);

    std::cout << "SZ " << aCS0->Sz() << " " << aCSGrid->Sz() << " ISGR " << aCSGrid->IsGrid() << "\n";

    int aNb = 10;
    for (int aK = 0 ; aK<=aNb ; aK++)
    {
        Pt2dr aP = Pt2dr(aCS0->Sz()) * (aK/double(aNb));
        Pt2dr aUD1 = aCS0->F2toC2(aP);
        Pt2dr aUD2 = aCSGrid->F2toC2(aP);

        std::cout << aP << aUD1 << aUD2 << "\n";
    }

}


// Test de fit de courbe analytique sur un polynome 
class cTestCircleFit
{
    public :

cTestCircleFit()
{
    mTetaMax = 0.5;
    mNbEch = 100;
    mDegre = 4;
    L2SysSurResol mSys(mDegre+1);

    for (int aK=0 ; aK<mNbEch ; aK++)
    {
        double aTeta = Teta(aK);
        std::vector<double> aVC;
        for (int aD=0 ; aD<=mDegre ; aD++)
        {
            aVC.push_back(pow(aTeta,aD));
        }
        mSys.V_GSSR_AddNewEquation(1.0,VData(aVC),F(aTeta));
    }

    bool Ok;
    Im1D_REAL8  aSol = mSys.V_GSSR_Solve(&Ok);
    double * aDS = aSol.data();

    
    for (int aK=0 ; aK<mNbEch ; aK++)
    {
        double aTeta = Teta(aK);
        std::vector<double> aVC;
        double aRes = 0.0;
        for (int aD=0 ; aD<=mDegre ; aD++)
        {
             aRes += pow(aTeta,aD) * aDS[aD];
        }
        aRes -= F(aTeta);

        std::cout << "RES " << ((aRes>0) ? "+" : "-" ) << "\n";
    }

    getchar();
}

    double F(double aTeta)
    {
          return sqrt(1-aTeta*aTeta + aTeta);
    }


    double Teta(int aK) { return ((mTetaMax) /mNbEch) * aK;}

    double mTetaMax ;
    int    mNbEch;
    int    mDegre;
};






void TestBitmFont(const std::string & aStr,Pt2di aSpace,Pt2di aRab,int aCenter)
{
    cElBitmFont & aFont=  cElBitmFont::BasicFont_10x8();
    Im2D_Bits<1> anIm = aFont.MultiLineImageString(aStr,aSpace,aRab,aCenter);
    Tiff_Im::Create8BFromFonc("toto.tif",anIm.sz(),255*anIm.in());
    std::cout << "DONE " << aStr << "\n";
    getchar();
}

void TestFont()
{
    // cElBitmFont & aFont=  cElBitmFont::BasicFont_10x8();
    //TestSplitStrings("Abcd");
	    //TestSplitStrings("");
	    // TestSplitStrings("Ab\ncd");
    // TestSplitStrings("Ab\n\ncd");
    // TestSplitStrings("Ab\n\ncd\n");

    // Im2D_Bits<1> anIm = ImageString(aFont,"01AB",2);
    TestBitmFont("01AB\n1\n\n1234567",Pt2di(2,10),Pt2di(5,5),0);
exit(0);
}


void TestFoncReelle(Fonc_Num aF,const std::string & aName,Pt2di aSz)
{
    Im2D_REAL8 anIm(aSz.x,aSz.y);
    ELISE_COPY(anIm.all_pts(),aF,anIm.out());
    std::vector<Im2DGen> aV;
    aV.push_back(anIm);
    Tiff_Im::CreateFromIm(aV,aName);

}


void TestcFixedMergeStruct()
{
    cCMT_NoVal aValArc;
    cStructMergeTieP< cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> >  aFMS(3,false);

    aFMS.AddArc(Pt2dr(0,0),0,Pt2dr(1,1),1,aValArc);
    aFMS.AddArc(Pt2dr(1,1),1,Pt2dr(1,2),2,aValArc);
    aFMS.AddArc(Pt2dr(1,2),2,Pt2dr(0,0),0,aValArc);

    aFMS.AddArc(Pt2dr(10,10),0,Pt2dr(11,11),1,aValArc);
    aFMS.AddArc(Pt2dr(11,11),1,Pt2dr(12,12),2,aValArc);
    aFMS.AddArc(Pt2dr(12,12),2,Pt2dr(11,11),1,aValArc);
//    aFMS.AddArc(Pt2dr(12,12),1,Pt2dr(10,10),0);
    aFMS.AddArc(Pt2dr(11,11),1,Pt2dr(10,10),0,aValArc);
//    aFMS.AddArc(Pt2dr(12,12),1,Pt2dr(10,10),0);

    aFMS.AddArc(Pt2dr(7,7),0,Pt2dr(8,8),1,aValArc);
    aFMS.AddArc(Pt2dr(8,8),1,Pt2dr(7,7),0,aValArc);

    aFMS.AddArc(Pt2dr(4,4),0,Pt2dr(5,5),1,aValArc);
    aFMS.AddArc(Pt2dr(5,5),1,Pt2dr(4,4),0,aValArc);
    aFMS.AddArc(Pt2dr(5,5),1,Pt2dr(6,6),2,aValArc);

    aFMS.AddArc(Pt2dr(77,77),0,Pt2dr(88,88),2,aValArc);
    aFMS.AddArc(Pt2dr(88,88),2,Pt2dr(99,99),1,aValArc);

    aFMS.AddArc(Pt2dr(777,777),0,Pt2dr(888,888),1,aValArc);
    aFMS.AddArc(Pt2dr(888,888),1,Pt2dr(999,999),2,aValArc);

    aFMS.DoExport();

    const std::list<cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> *> &  aLM = aFMS.ListMerged();
 
    std::cout << "NB ITEM " << aLM.size() << "\n";

    for
    (
        std::list<cFixedSizeMergeTieP<3,Pt2dr,cCMT_NoVal> *>::const_iterator itM=aLM.begin();
        itM != aLM.end();
        itM++
    )
    {
          std::cout << "NbS=" << (*itM)->NbSom() << " NbA=" << (*itM)->NbArc()<<endl;
          for (int i=0; i<(*itM)->NbSom(); i++)
          {
            std::cout << (*itM)->IsInit(i)<<" " ;
            std::cout << " " << (*itM)->GetVal(i) ;
            std::cout << "\n";
          }

    }
}


void TestXmlX11();

void TestEllips();

/*
cElXMLTree * GetRemanentFromFileAndTag(const std::string & aNameFile,const std::string & aNameTag)
{
   static std::map<std::string,cElXMLTree *> DicoHead;
   if (DicoHead[aNameFile] ==0)
       DicoHead[aNameFile] = new cElXMLTree(aNameFile);

   return DicoHead[aNameFile]->GetOneOrZero(aNameTag);
}

template <class Type> bool InitObjFromXml
                           (
                                 Type & anObj,
                                 const std::string & aNameFile,
                                 const std::string& aFileSpec,
                                 const std::string & aNameTagObj,
                                 const std::string & aNameTagType
                           )
{
   if (GetRemanentFromFileAndTag(StdGetFileXMLSpec(aFileSpec),aNameTagType))
   {
       anObj = StdGetObjFromFile<Type>(aNameFile,StdGetFileXMLSpec(aFileSpec),aNameTagObj,aNameTagType);
       return true;
   }
   return false;
}

template <class Type> bool StdInitObjFromXml
                           (
                                 Type & anObj,
                                 const std::string & aNameFile,
                                 const std::string & aNameTagObj,
                                 const std::string & aNameTagType
                           )
{
     return 
              InitObjFromXml(anObj,aNameFile,"ParamApero.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"ParamMICMAC.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"SuperposImage.xml",aNameTagObj,aNameTagType)
         ||   InitObjFromXml(anObj,aNameFile,"ParamChantierPhotogram.xml",aNameTagObj,aNameTagType)
     ;
}



template <class Type> void MyBinUndumpObj
                          (
                              Type & anObj,
                              const std::string & aFile,
                              const std::string & aNameTagObj,
                              const std::string & aNameTagType
                          )
{
     ELISE_fp aFPIn(aFile.c_str(),ELISE_fp::READ);
     int aNum;

     BinaryUnDumpFromFile(aNum,aFPIn);
     // NumHgRev doesn't work with the new Git version
     //if (aNum!=NumHgRev())
     //{
     //}

     std::string aVerifMangling;
     BinaryUnDumpFromFile(aVerifMangling,aFPIn);
     if (aVerifMangling!=Mangling((Type*)0))
     {
        std::string aXmlName = StdPrefix(aFile)+".xml";
        if (ELISE_fp::exist_file(aXmlName))
        {
           std::cout << "Dump version problem for "<<  aFile << " , try to recover from xml\n";
           if (StdInitObjFromXml(anObj,aXmlName,aNameTagObj,aNameTagType))
           {
               MakeFileXML(anObj,aFile);
               std::cout << "    OK recovered " << aFile << "\n";
               return;
           }
        }
        std::cout << "For file " << aFile << "\n";
        ELISE_ASSERT(false,"Type has changed between Dump/Undump")
     }


     BinaryUnDumpFromFile(anObj,aFPIn);
     aFPIn.close();
}

*/

void TestUnDump()
{
    std::string aNameFile = "/home/ubuntu/Desktop/Data/Amphorus//Tmp-MM-Dir/IMG_20160716_194704.jpg-MDT-4227.dmp";

    cXmlXifInfo aV;
    // MyBinUndumpObj(aV,aNameFile,"XmlXifInfo","XmlXifInfo");


/*
    ELISE_fp aFPIn(aNameFile.c_str(),ELISE_fp::READ);
    int aNum;

     BinaryUnDumpFromFile(aNum,aFPIn);
     std::cout << "NUM=" << aNum << "\n";
     std::string aVerifMangling;
     BinaryUnDumpFromFile(aVerifMangling,aFPIn);
     std::cout << "MnglDmp=" << aVerifMangling <<"\n";
     std::cout << "MnglCalc=" << Mangling((cXmlXifInfo*)0) <<"\n";

     bool Ok= 
     StdInitObjFromXml
     (
             aV,
            "/home/ubuntu/Desktop/Data/Amphorus//Tmp-MM-Dir/IMG_20160716_194704.jpg-MDT-4227.xml",
             "XmlXifInfo",
             "XmlXifInfo"
     );

     std::cout << "VVVVV " << Ok << "\n";

     MakeFileXML(aV,"toto.xml");
*/

/*
  StdGetFromPCP
  (
        "/home/ubuntu/Desktop/Data/Amphorus//Tmp-MM-Dir/IMG_20160716_194704.jpg-MDT-4227.dmp",
        XmlXifInfo
  );
*/
}

void TestMemory()
{
    int aCpt=0;

    while (1)
    {
        int aNbInc = 100;
        L2SysSurResol aSys(aNbInc);

        for (int aKEq=0 ; aKEq<(2*aNbInc) ; aKEq++)
        {
             std::vector<double> aVCoeff;
             for (int aKInc=0 ; aKInc<aNbInc ; aKInc++)
             {
                 aVCoeff.push_back(NRrandC());
             }
             aSys.AddEquation(1.0,&(aVCoeff[0]),NRrandC());
        }
        aSys.Solve(0);

        aCpt++;
        if ((aCpt%100)==0) std::cout << "CPT=" << aCpt << "\n";
    }
}

extern void TMA();

void TestClipBundle()
{
   std::string aNFull = "Ori-RPC-d1-gcp-OriNew/GB-Orientation-IMG_PHR1B_P_201301260750566_SEN_IPU_20130612_0919-003_R1C1.JP2.tif.xml";

   std::string aNClip = "Ori-RPC-d1-gcp-OriNew-Cliped_1/GB-Orientation-Cliped_1-IMG_PHR1B_P_201301260750566_SEN_IPU_20130612_0919-003_R1C1.JP2.tif.xml";

   int aType = 0 ;

   cBasicGeomCap3D * aCFull = cBasicGeomCap3D::StdGetFromFile(aNFull,aType);
   cBasicGeomCap3D * aCClip = cBasicGeomCap3D::StdGetFromFile(aNClip,aType);

   std::cout << "SZZZZ ||  Ful : " << aCFull->SzBasicCapt3D() << " ; Clip : "<< aCClip->SzBasicCapt3D() << "\n";

   Pt2dr aPIm0 = Pt2dr(500,500);
   Pt3dr aPTer = aCClip->ImEtZ2Terrain(aPIm0,1000.0);
   Pt2dr aPIm1 = aCFull->Ter2Capteur(aPTer);

   std::cout << "PIM1111 " << aPIm1 << " " << aCClip->Ter2Capteur(aPTer)  << "\n";


   std::cout << " Def Alti " << aCClip->AltisSolIsDef() << " " << aCFull->AltisSolIsDef() << "\n";
   std::cout << " MinMax " << aCClip->GetAltiSolMinMax() << " " << aCFull->GetAltiSolMinMax() << "\n";
   std::cout << " ZzzZZzz " << aCClip->GetAltiSol() << " " << aCFull->GetAltiSol() << "\n";
   
   exit(EXIT_SUCCESS);
}

void TestFileTxtBin(bool ModeBin)
{
    std::string aName = ModeBin ? "toto.dat" : "toto.txt";
    ELISE_fp aFp(aName.c_str(),ELISE_fp::WRITE,false, ModeBin ? ELISE_fp::eBinTjs : ELISE_fp::eTxtTjs);

    aFp.SetFormatFloat("%.3f");
    aFp.write_U_INT4(1);
    aFp.PutLine();
    aFp.write_U_INT4(10);
    aFp.PutCommentaire("Ceci est un joli commentaire");
    aFp.write_line("Version=0");
    aFp.write_REAL4(1/3.0);
    aFp.PutLine();
    aFp.PutCommentaire("t");
    aFp.PutCommentaire("toto2");
    aFp.write_REAL8(4/3.0);
    aFp.write_U_INT4(10);
    aFp.close();

    ELISE_fp aFp2(aName.c_str(),ELISE_fp::READ,false, ModeBin ? ELISE_fp::eBinTjs : ELISE_fp::eTxtTjs);

    std::cout << "NME " << aName << "\n";
    std::cout << aFp2.read_U_INT4()  << "\n" ;
    std::cout << aFp2.read_U_INT4()  << "\n" ;
    std::cout << aFp2.std_fgets() << "\n";
    std::cout << aFp2.read_REAL4()  << "\n" ;
    std::cout << aFp2.read_REAL8()  << "\n\n" ;


}

void TestFileTxtBin()
{
   TestFileTxtBin(true);
   TestFileTxtBin(false);

   exit(EXIT_SUCCESS);
}

void TestFitPol()
{
    for (int aR=0 ; aR < 3 ; aR++)
    {
        for (int aD=1 ; aD <10 ; aD++)
        {
              std::vector<Pt2dr> aVS;
              for (int aKS=0 ; aKS<= aD+aR ; aKS++)
                 aVS.push_back(Pt2dr(1e5*NRrandC(),1e3*NRrandC()));

              LeasSqFit(aVS,aD);
              // std::cout << "TestFitPol D=" << aD << " RAB=" << aR << "\n";
              // getchar();
        }
    }
}


extern void TestMap2D();
extern void TestEcartTypeStd();

void TestHomogr()
{
   ElPackHomologue aPack;
  
   aPack.Cple_Add(ElCplePtsHomologues(Pt2dr(0,0),Pt2dr(0,0)));
   aPack.Cple_Add(ElCplePtsHomologues(Pt2dr(0,1),Pt2dr(0,1)));
   aPack.Cple_Add(ElCplePtsHomologues(Pt2dr(1,0),Pt2dr(1,0)));

   cElHomographie aHAff (aPack,true);

   aPack.Cple_Add(ElCplePtsHomologues(Pt2dr(1,1),Pt2dr(2,10)));

   cElHomographie aVraiH (aPack,true);

   int aNB = 10 ;
   for (int aK =0 ; aK<= aNB  ; aK++)
   {
       double A = aK / double(aNB);
       Pt2dr aP(A,1-A);

       std::cout << aP << " " << aHAff(aP)  << " " << aVraiH(aP) << "\n";
   }
}


extern void TestFilterGauss();
extern void TestCondStereo();

extern void TestcBiaisedRandGenerator();
extern void OneTestcPrediCoord();


extern void TestcGeneratorEqColLin();
extern void TestPrime();

int MPDtest_main (int argc,char** argv)
{
    std::cout << "USER "  << MMUserEnv().UserName().ValWithDef("toto")  << " MPD=" << MPD_MM() << "\n";
    {
        TestcGeneratorEqColLin();
        exit(EXIT_SUCCESS);
    }
    {
        TestPrime();
        exit(EXIT_SUCCESS);
    }
    {
        OneTestcPrediCoord();
        exit(EXIT_SUCCESS);
    }
    {
        TestcBiaisedRandGenerator();
        exit(EXIT_SUCCESS);
    }
    {
        TestCondStereo();
        exit(EXIT_SUCCESS);
    }
    {
        TestFilterGauss();
        exit(EXIT_SUCCESS);
    }
    {
        Pt2di aP(101.999999,101.000001);
        std::cout <<  "ppppppppppp "  << aP << "\n";
        getchar();
    }
    std::cout << "MPDtest_main in " << __FILE__ << "\n";
    {
       TestHomogr();
       exit(EXIT_SUCCESS);
    }
    {
       TestEcartTypeStd();
       exit(EXIT_SUCCESS);
    }
    {
       TestFitPol();
       exit(EXIT_SUCCESS);
    }
    
    {
       double aTD0 = 2e9;
       float  aFT0 = 2e9;

       double aTD1 = aTD0 + 1e-5;
       float  aFT1 = aFT0 + 1e-5;
    

       std::cout << "DOUBLE "  <<  aTD1 - aTD0  << " " << aFT1 - aFT0 << "\n";


       exit(EXIT_SUCCESS);
    }
    {
       TestEllips();
       exit(EXIT_SUCCESS);
    }
    TestMap2D();
    TestFileTxtBin();

    TestClipBundle();
    {
        double aU0 = 0.5;
        for (int aK=1 ; aK< 100000 ; aK++)
        {
            aU0 = (exp(aU0)-1) / (exp(aU0) -aU0);
            std::cout << aU0 << " " << aK << "\n";
            getchar();
        }
    }


    {
       //TestMemory();
       TMA();
       exit(EXIT_SUCCESS);
    }
    {
       TestUnDump();
       exit(EXIT_SUCCESS);
    }
    {
        int aNx=3;
        int aNy=4;
        int aNxy = ElMax(aNx,aNy);
        ElMatrix<double> aM(aNxy,aNxy);
        for (int aKx=0 ; aKx<aNxy ; aKx++)
        {
           for (int aKy=0 ; aKy<aNxy ; aKy++)
           {
             if ((aNx<=aNxy) && (aNy<=aNxy))
                aM(aKx,aKy) = 1.0 / (1+aKx + aKy*aKy);
             else
                aM(aKx,aKy) = 0.0;
           }
        }

        ElMatrix<double> aU(1,1),aDiag(1,1),aV(1,1);
      
        svdcmp(aM,aU,aDiag,aV,false);
        exit(EXIT_SUCCESS);
    }



    {
       double aFact =  1;
       int aNb=0;
       std::cout << "Entrer l'entier max\n";
       std::cin >> aNb;
       for (int i=1 ; i<=aNb ; i++)
           aFact *= i;

       std::cout << "Produit de 1 a " << aNb << " = " << aFact << "\n";

       exit(EXIT_SUCCESS);
    }


    {
        std::cout << "ELISE_X11=" << ELISE_X11 << "\n";
    }
    // TestXmlX11();
    // TestcFixedMergeStruct();
    // TestFoncReelle(FX/100.0,"FXDiv100.tif",Pt2di(500,500));

   // TestFoncReelle(FY/10.0,"FYDiv10.tif");

   // TestFoncReelle(FX-1000,"FX.tif");
   // TestFoncReelle(500*sin(FX/50.0)*sin(FY/70.0),"SinSin.tif");

    // TestFoncReelle(500*sin(FX/50.0 + 20*sin((FY+(FX*FY)*1e-5)/500.0)),"SinPer.tif", Pt2di (8000,12000));
    exit(1);
    // cTestCircleFit aTCF;
/*
     TestFont();
   for (int aK=0 ; aK< 100 ; aK++)
   {
        Pt3dr anAxe(NRrandom3(),NRrandom3(),NRrandom3());
        anAxe = vunit(anAxe);
        ElMatrix<REAL>  aRot = VectRotationArroundAxe(anAxe,10*NRrandom3());
        Pt3dr  anAxe2 = AxeRot(aRot);
        if (scal(anAxe2,anAxe) < 0) anAxe2 = - anAxe2;

        std::cout << anAxe -anAxe2 << "\n";
 
   }
*/

   ElTimer aChrono;
   int aCpt=0;
   while (1)
   {
       if (ELISE_fp::exist_file("toto" + ToString(aCpt)))
       {
             std::cout << "GET ONE " << aCpt << "\n";
             getchar();
       }
       if ((aCpt%100000) ==1) std::cout << "TIME " << aChrono.uval() << " " << aCpt << " " << 1e6*(aChrono.uval() / aCpt)<< "\n";
       aCpt ++;
   }

/*
   TestMax(argc,argv);
   cXml_ScanLineSensor  aSensor = StdGetFromSI("/home/marc/TMP/EPI/TestSens.xml",Xml_ScanLineSensor);

   MakeFileXML(aSensor,"/home/marc/TMP/EPI/TestSens.dmp");

   aSensor =  StdGetFromSI("/home/marc/TMP/EPI/TestSens.dmp",Xml_ScanLineSensor);


   std::cout <<  aSensor.Lines().begin()->Rays().begin()->P1() << "\n";
*/

/*
    PdBump();
cXml_Ori2Im aXmlOri0;
MakeFileXML(aXmlOri0,aName);
std::string aName = "Test.xml";
cXml_Ori2Im  aXmlOri = StdGetFromSI(aName,Xml_Ori2Im);
*/


/*
   std::string aName = "/home/marc/TMP/EPI/EXO1-Fontaine/Ori2ImAll/DirIm_AIMG_2470.JPG/AIMG_2472.JPG.xml";
   cXml_Ori2Im  aXmlOri = StdGetFromSI(aName,Xml_Ori2Im);
*/





   // PartitionRenato(argc,argv);
/*
   B/EenchSort3();
    Bench_NewOri();

    TestSVD3x3(); 
   std::cout << "Hello Matis\n";
   std::string aFile = "/home/prof/Bureau/FORM-DEV-2015/Toulouse/Indent.xml";
   aFile = "/home/prof/Bureau/FORM-DEV-2015/Toulouse/Toulouse-131010_0716-simplified.ori.xml";
   //aFile = "/home/prof/Bureau/FORM-DEV-2015/Toulouse/Toulouse-131010_0716-00-00001_0000000.ori.xml";
   corientation anOri = StdGetFromPCP(aFile,orientation);

   std::cout << "V " << anOri.version().Val() << "\n";

   corientation anOri2;
   MakeFileXML(anOri2,"/home/prof/Bureau/FORM-DEV-2015/Toulouse/V3.xml");
*/

/*
   std::cout << anOri.sommet().easting() << "\n";
   anOri.sommet().easting() = 0;
   MakeFileXML(anOri,"/home/prof/Bureau/FORM-DEV-2015/Toulouse/V2.xml");
*/
   

/*
   
    corientation anOri =  StdGetFromPCP("/home/prof/Bureau/FORM-DEV-2015/Toulouse/Simple.xml",orientation);

   std::cout << "V = " << anOri.version().Val() << "\n";
    cXML_TestImportOri aXIM =  StdGetFromSI("/home/mpd/TMP/Test.xml",XML_TestImportOri);
    std::cout << "x " << aXIM.x() << "\n";
    aXIM.Tree().mTree->StdShow(" ");
    MakeFileXML(aXIM,"/home/mpd/TMP/Test2.xml");
*/


/*
    cMasqBin3D::FromSaisieMasq3d("/home/marc/TMP/EPI/EXO1-Fontaine/AperiCloud_All_selectionInfo.xml");


     LoadTrScaleRotate
     (
          "/media/data2/Jeux-Test/img_0762.cr2_Ch3.tif",
          "/media/data2/Jeux-Test/img_0762.cr2_Ch3_Scaled.tif",
          Pt2di(1500,1500),
          Pt2di(2100,2400),
          Pt2di(500,500),
          3.0,
          1
     );
*/
          
/*
   TestOriBundle();
    Jeremy_main(argc,argv);
   cCalibrationInterneRadiale aXmlDr;
   aXmlDr.CDist() = Pt2dr(3,4);
*/


/*
   for (int aK=0 ; aK<argc ; aK++)
      std::cout << argv[aK] << "\n";

   std::string aDir,aFile;

   SplitDirAndFile(aDir,aFile,argv[1]);

   for (int aResol = -1 ; aResol <2000 ; aResol +=500)
   {
       std::string aTest;
       getPastisGrayscaleFilename(aDir,aFile,aResol,aTest);
       std::cout << "RESS " << aResol << " => " << aTest << "\n";
   }
*/

   
/*
    ELISE_ASSERT(argc==3,"MPDtest_main");
    FiltreRemoveFlou(argv[1],argv[2]);
*/
    // std::cout << "ARC " << argv[1] << " " << argv[2] << "\n";
    cElWarning::ShowWarns("MPDTest.txt");
  
   return 0;

}

std_unique_ptr<char> toto;




#endif


int SysCalled_main (int argc,char** argv)
{
    int aResul;
    bool ByExit= true;
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aResul,"result val"),
        LArgMain()  << EAM(ByExit,"BE",true,"By Exit")
    );

    if (ByExit)
       exit(aResul);

    return aResul;
}

int SysCall_main (int argc,char** argv)
{
    for (int aK=-2 ; aK<= 2 ; aK++)
    {
        for (int aBE=0 ; aBE<=1 ; aBE++)
        {
            std::string aCom = "mm3d TestLib SysCalled "+ ToString(aK) + " BE=" + ToString(aBE!=0);
            int aV = system(aCom.c_str());
            std::cout << "aK= " << aK << " Got=" << aV  << " " << (int)((char*)&aV)[1] << "  BE=" << aBE << "\n";
        }
    }
    return EXIT_SUCCESS;
}


int CPP_DebugAI4GeoMasq (int argc,char** argv)
{
   std::string aName("./Pyram/MasqIm_Dz1_M56685x56681_EpipIm-RPC-deg1-Renamed-adj-22APR15WV03.tif-19DEC15WV03.tif_Masq.tif");
   Tiff_Im aTifM(aName.c_str());

   double aV0;
   Pt2di aSz(1060,1129), aP0(0,38037);
   ElInitArgMain
   (
        argc,argv,
        LArgMain()  ,
        LArgMain()  <<  EAM(aSz,"Sz",true,"Sz of rect, def=1060,1129")
                    <<  EAM(aP0,"P0",true,"P0 of Box, def = 0,38037")
   );

   std::cout << "Sz of file " << aTifM.sz() << "\n";

   ELISE_COPY
   (
       rectangle(Pt2di(0,0),aSz),
       trans(aTifM.in(),aP0),
       sigma(aV0)
   );


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
