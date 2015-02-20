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
#if (ELISE_QT_VERSION >= 4)
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








extern void getPastisGrayscaleFilename(const std::string & aParamDir, const string &i_baseName, int i_resolution, string &o_grayscaleFilename );
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

int MPDtest_main (int argc,char** argv)
{
    Im2D_REAL4 anIm(200,200);
    TIm2D<REAL4,REAL> aTIm(anIm);

    Pt2dr aP(100,102);
    aTIm.getr(aP/5.0); // Interpole
    aTIm.getr(aP/5.0,2); // Interpole et return 2 si en dehors
    aTIm.getprojR(aP/5.0); // Interpole et prolonge par continuite si en dehors
    aTIm.get(round_ni(Pt2dr(3.6,3)));

    ELISE_COPY
    (
        anIm.all_pts(),
        anIm.in()[Virgule(FX/5,FY/5)],
        anIm.out()
    );


/*
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
#if (ELISE_QT_VERSION >= 4)

#endif
  
   return 0;

}

std_unique_ptr<char> toto;

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
