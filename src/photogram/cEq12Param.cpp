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

#define NoTemplateOperatorVirgule
#define NoSimpleTemplateOperatorVirgule


#include "StdAfx.h"

static bool    SHOW_11P =true;

/*
 class cQual12Param
{
     public :
       cQual12Param();

       void Show();

       double mMoyReproj;
       double mMoyBundleProj;
       double mMoyBundleIm;
       double mPropVis;
};
*/
cQual12Param::cQual12Param() :
   mMoyReproj (0.0),
   mMoyBundleProj (0.0),
   mMoyBundleIm (0.0),
   mPropVis (0.0)
{
}
void cQual12Param::Show()  const
{
    std::cout <<   " ErProj=" << mMoyReproj
              <<   " BundP="  << mMoyBundleProj
              <<   " BundI="  << mMoyBundleIm
              <<   " PVis="  << mPropVis
    ;
}


/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)
*/


cEq12Parametre::cEq12Parametre() :
       mSys         (12,true),
       mIndFixArb   (10),
       mValueFixArb (-1)
{
}

void cEq12Parametre::AddObs(const Pt3dr & aPGround,const Pt2dr & aPPhgr,const double&  aPds)
{
  mVPG.push_back(aPGround);
  mVPPhgr.push_back(aPPhgr);
  mVPds.push_back(aPds);
}


std::pair<ElMatrix<double>,ElRotation3D > cEq12Parametre::ComputeOrtho(bool *Ok)
{

   if (Ok) 
      *Ok=true;
   std::pair<ElMatrix<double>,Pt3dr> aPair = ComputeNonOrtho();

   std::pair<ElMatrix<double>,ElMatrix<double> > aRQ = RQDecomp(gaussj(aPair.first));
   ElMatrix<double> aR = aRQ.first;
   ElMatrix<double> aQ = aRQ.second;

   double aC22 = aR(2,2);

   for (int anX=0 ; anX<3; anX++)
   {
      for (int anY=0 ; anY<3; anY++)
      {
           aR(anX,anY) /= aC22 ;
      }
   }

   Pt3dr aCenter = aPair.second;
   ElRotation3D aRotC2M (aCenter,aQ.transpose(),true);
   int aNbGoodSide=0;
   int aNbBadSide=0;
   for (int aK = 0 ; aK<int(mVPG.size()) ; aK++)
   {
        Pt3dr aPCam = aRotC2M.ImRecAff(mVPG[aK]);
        if (aPCam.z>0)  
           aNbGoodSide++;
        else
           aNbBadSide++;
   }
    if (SHOW_11P)
    {
       std::cout << "        (11p-Sides : G=" <<aNbGoodSide   << " B=" << aNbBadSide  << ")\n";
    }
             // cElWarning::OnzeParamSigneIncoh.AddWarn("",__LINE__,__FILE__);

   if ((aNbGoodSide!=0) && (aNbBadSide==0))
   {
       cElWarning::OnzeParamSigneIncoh.AddWarn("",__LINE__,__FILE__);
   }


   if (aNbBadSide > aNbGoodSide)
   {
     
/*
       Pt3dr aI = aRotC2M.ImVect(Pt3dr(1,0,0));
       Pt3dr aJ = aRotC2M.ImVect(Pt3dr(0,1,0));
       Pt3dr aK = aRotC2M.ImVect(Pt3dr(0,0,1));
*/
       if (Ok)
       {
          *Ok=false;
       }
       else
       {
          ELISE_ASSERT(false,"In 11 param, probable sign error in coordinates ");
       }

   }

/*
   std::cout << "ComputeOrthoComputeOrtho C=" << aC << "\n";
   for (int anY=0 ; anY<3; anY++)
   {
      for (int anX=0 ; anX<3; anX++)
      {
           std::cout << aR(anX,anY)  << " " ;
      }
      std::cout <<  " \n" ;
   }
   getchar();
*/

   //return std::pair<ElMatrix<double>,ElRotation3D> (aR,ElRotation3D(aPair.second,aQ,true));

   if (SHOW_11P)
   {
       // ShowMatr("Tri", aR)  ;
       // ShowMatr("Rot", aRotC2M.Mat())  ;
       double aSomEc =0.0;
       for (int aK=0 ; aK <int(mVPG.size()) ; aK++)
       {
              Pt3dr aP3 = mVPG[aK];
              Pt2dr aPIm = mVPPhgr[aK];

	      Pt3dr aPLoc = aRotC2M.ImRecAff(aP3);
              aPLoc = aR * aPLoc;

	      Pt2dr aPProj (aPLoc.x/aPLoc.z,aPLoc.y/aPLoc.z);
              // std::cout << " DecOrth Pt3=" << aP3  << " PIm=" << aPIm  << " pp=" << aPProj   << "\n";
	      aSomEc +=   euclid(aPIm-aPProj);
       }
       std::cout << "  DecOrtho=" << aSomEc/mVPG.size()  << "\n";
   }


   // return std::pair<ElMatrix<double>,ElRotation3D> (aR,ElRotation3D(aPair.second,aQ.transpose(),true));
   return std::pair<ElMatrix<double>,ElRotation3D> (aR,aRotC2M);
   
}

static double  CoeffHom3(const double * aDS,const Pt3dr & aP)
{
     return aDS[0]*aP.x+aDS[1]*aP.y + aDS[2]*aP.z + aDS[3];
}

static Pt2dr  PtHom3(const double * aDS,const Pt3dr & aP)
{
   double aDen = CoeffHom3(aDS+8,aP);

   return Pt2dr(CoeffHom3(aDS+0,aP) / aDen,  CoeffHom3(aDS+4,aP) /aDen);
}

/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)
*/


std::pair<ElMatrix<double>,Pt3dr> cEq12Parametre::ComputeNonOrtho()
{
    mSys.GSSR_Reset(false);

    // Pour limiter les erreurs numériques, on se remet sur des coordonnees centrees
    Pt3dr aMoyP(0,0,0);
    double aSomPds = 0;
    for (int aK=0 ; aK <int(mVPG.size()) ; aK++)
    {
       aMoyP = aMoyP + mVPG[aK] * mVPds[aK];
       aSomPds += mVPds[aK];
    }
    aMoyP = aMoyP / aSomPds;


    for (int aK=0 ; aK <int(mVPG.size()) ; aK++)
    {
         ComputeOneObs(mVPG[aK]-aMoyP,mVPPhgr[aK],mVPds[aK]);
    }


    ElMatrix<double> aMat(3,3);
    Pt3dr aC;

    bool aOk;
    Im1D_REAL8  aISol = mSys.GSSR_Solve(&aOk);
    double * aDS = aISol.data();

    // Tout est a un facteur pres, on normalise pour que la matrice 3x3 soit de norme L2
    // (a permis de tester pour les rotations pure si OK ...)
    double aSomL2=0;
    for (int aK=0 ; aK<12 ; aK+=4)
    {
        aSomL2 += Square(aDS[aK]) + Square(aDS[aK+1])  + Square(aDS[aK+2]) ;
    }
    aSomL2 = sqrt(aSomL2/3);
    for (int aK=0 ; aK<12 ; aK++)
    {
        aDS[aK  ] /= aSomL2;
    }

    //  Test correctness on homographie computation
    if (SHOW_11P)
    {
         // std::cout << "NBPTS=" << mVPG.size()<< "\n" ;
	 double aSomEc =0.0;
         for (int aK=0 ; aK <int(mVPG.size()) ; aK++)
         {
              Pt3dr aP3 = mVPG[aK]-aMoyP;
              Pt2dr aPIm = mVPPhgr[aK];
              Pt2dr aPProj = PtHom3(aDS,aP3);
              // std::cout << " Pt3=" << aP3  << " PIm=" << aPIm  << " pp=" << aPProj   << " DSArb=" << aDS[mIndFixArb] << "\n";
	      aSomEc +=   euclid(aPIm-aPProj);
         }
	 std::cout << "  EcHom=" << aSomEc/mVPG.size()  << "\n";
    }
/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)

    I = (p0 x + p1 y + p2 z + p3)  /  (p8 x + p9 y + p10 z + p11)
    J = (p4 x + p5 y + p6 z + p7)  /  (p8 x + p9 y + p10 z + p11)


                   (     (p0  p1  p2  )  (x)     (p3)  )
    (I,J)  = Proj  (     (p4  p5  p6  )  (y)  +  (p7)  )  = Proj ( M*P +Q0) =  Proj( M*(P-(-M-1*Q0)))
                   (     (p8  p9  p10 )  (z)     (p11) )
*/

    // Le centre est le point ou les trois terme sont nul (et les deux ratio indetermines)

    Pt3dr aPInc(-aDS[3],-aDS[7],-aDS[11]);
    for (int aKy=0 ; aKy<3 ; aKy++)
    {
        for (int aKx=0 ; aKx<3 ; aKx++)
        {
            aMat(aKx,aKy) = aDS[aKx+4*aKy];
        }
    }
    aC = gaussj(aMat) * aPInc + aMoyP;
   
    //  Test correctness on matrix computation
    if (SHOW_11P)
    {
	 double aSomEc =0.0;
         for (int aK=0 ; aK <int(mVPG.size()) ; aK++)
         {
              Pt3dr aP3 = mVPG[aK];
              Pt2dr aPIm = mVPPhgr[aK];

	      Pt3dr aPLoc = aMat*(aP3-aC);
	      Pt2dr aPProj (aPLoc.x/aPLoc.z,aPLoc.y/aPLoc.z);

              // std::cout << " Pt3=" << aP3  << " PIm=" << aPIm  << " pp=" << aPProj   << " DSArb=" << aDS[mIndFixArb] << "\n";
	      aSomEc +=   euclid(aPIm-aPProj);
         }
	 std::cout << "  EcMatrix(NO)=" << aSomEc/mVPG.size()  << "\n";
    }

    return std::pair<ElMatrix<double>,Pt3dr> (gaussj(aMat),aC);
}


/*   
    0 = (p0 x + p1 y + p2 z + p3) - I (p8 x + p9 y + p10 z + p11)
    0 = (p4 x + p5 y + p6 z + p7) - J (p8 x + p9 y + p10 z + p11)
*/
void cEq12Parametre::ComputeOneObs(const Pt3dr & aPG,const Pt2dr & aPPhgr,const double&  aPds)
{
    double aC[12];
  
    aC[0] = aPG.x;
    aC[1] = aPG.y;
    aC[2] = aPG.z;
    aC[3] = 1;
    for (int aK=0 ; aK<4 ; aK++)
    {
       aC[aK+4] = 0.0;
       aC[aK+8] = -aC[aK] * aPPhgr.x;
    }
    mSys.GSSR_AddNewEquation(aPds,aC,0.0,(double *)0);

    for (int aK=0 ; aK<4 ; aK++)
    {
       aC[aK+4] = aC[aK];
       aC[aK+8] = -aC[aK] * aPPhgr.y;
       aC[aK] = 0;
    }
    mSys.GSSR_AddNewEquation(aPds,aC,0.0,(double *)0);

    for (int aK=0 ; aK<12 ; aK++)
        aC[aK] = (aK==mIndFixArb);
    mSys.GSSR_AddNewEquation(aPds,aC,mValueFixArb,(double *)0);
}

CamStenope * cEq12Parametre::Camera11Param
             (
	         cQual12Param & aQual,
                 const Pt2di&               aSzCam,
                 bool                       isFraserModel,
                 const std::vector<Pt3dr> & aVCPCur,
                 const std::vector<Pt2dr> & aVImCur,
                 double & Alti ,
                 double & Prof
             )
{
    cEq12Parametre anEq12;
    Pt3dr aPMoy(0,0,0);
    for (int aK=0 ; aK<int(aVCPCur.size()) ; aK++)
    {
        anEq12.AddObs(aVCPCur[aK],aVImCur[aK],1.0);
        aPMoy = aPMoy+aVCPCur[aK];
    }
    aPMoy = aPMoy/double(aVCPCur.size());
    bool Ok;
    std::pair<ElMatrix<double>,ElRotation3D > aPair = anEq12.ComputeOrtho(&Ok);
    if (!Ok) 
       return 0;
    ElMatrix<double> aMat = aPair.first;
    ElRotation3D aR = aPair.second;

    double aFX =  aMat(0,0);
    double aFY =  aMat(1,1);
    Pt2dr aPP(aMat(2,0),aMat(2,1));
    double aSkew =  aMat(1,0);



    Pt3dr aCenter =  aR.ImAff(Pt3dr(0,0,0));
    Alti = aPMoy.z;
    Prof = euclid(aPMoy-aCenter);


    Pt2dr aRSz = Pt2dr(aSzCam);

    ElDistRadiale_PolynImpair aDR((1.1*euclid(aRSz))/2.0,aPP);

    CamStenope * aCS=0;
    std::vector<double> aPAF;
    if (isFraserModel)
    {
        cDistModStdPhpgr aDPhg(aDR);
        aDPhg.b1() = (aFX-aFY)/ aFY;
        aDPhg.b2() = aSkew / aFY;
        aCS = new cCamStenopeModStdPhpgr(false,aFY,aPP,aDPhg,aPAF);
    }
    else
    {
        aCS = new cCamStenopeDistRadPol(false,(aFX+aFY)/2.0,aPP,aDR,aPAF);
    }

    if (aCS)
    {
        aCS->SetOrientation(aR.inv());

	aQual = cQual12Param();
        if (true)
	{
             aCS->SetOrientation(aR.inv());
	     aCS->SetSz(aSzCam);
             // ShowMatr("Tri", aMat)  ;
             // std::cout  <<  "FocalX=" << aFX   << " " << " FocalY=" << aFY << "\n";
             // std::cout  <<  "Skew=" <<  aSkew << "\n";
	     int aNbPts=aVCPCur.size();
             for (size_t aKPt=0 ; aKPt<aVCPCur.size() ; aKPt++)
             {
                 Pt3dr aP3 = aVCPCur[aKPt];
                 Pt2dr aPIm = aVImCur[aKPt];
                 Pt2dr aPProj = aCS->Ter2Capteur(aP3);

		 cArgOptionalPIsVisibleInImage aArgV;

                 // std::cout << "CSSSS Pt3=" << aP3  << " PIm=" << aPIm  << " pp=" << aPProj   << " D3=" << aDB << " Vis=" << isVis   <<  " Mes=" << aArgV.mWhy << "\n";
	         aQual.mMoyReproj     +=  euclid(aPIm-aPProj);
	         aQual.mMoyBundleProj +=  aCS->Capteur2RayTer(aPProj).DistDoite(aP3);
	         aQual.mMoyBundleIm   +=  aCS->Capteur2RayTer(aPIm  ).DistDoite(aP3);
	         aQual.mPropVis       +=  aCS->PIsVisibleInImage(aP3,&aArgV); 
             }
	     aQual.mMoyReproj     /= aNbPts;
	     aQual.mMoyBundleProj /= aNbPts;
	     aQual.mMoyBundleIm   /= aNbPts;
	     aQual.mPropVis       /= aNbPts;
	     // aQual.Show(); std::cout << "\n"; getchar();
	}
    }

    return aCS;
}


CamStenope * cEq12Parametre::RansacCamera11Param
                            (
	                        cQual12Param & aBestQual,
                                const Pt2di&               aSzCam,
                                bool                       isFraserModel,
                                const std::vector<Pt3dr> & aVCPTot,
                                const std::vector<Pt2dr> & aVImTot,
                                double & Alti ,
                                double & Prof,
                                int    aNbTest,
                                double  aPropInlier,
                                int     aNbMaxTirage
                            )
{
     int aNbTot = aVCPTot.size();
     double aScoreMin = 1e60;
     CamStenope * aBestSol=0;
     for (int aKTest=0 ; aKTest<aNbTest ; aKTest++)
     {
          int aNbPts = 6 + aKTest%ElMax(1,(1+aNbMaxTirage-6));
          cRandNParmiQ aSel(aNbPts,aNbTot);

          std::vector<Pt3dr> aVCPSel;
          std::vector<Pt2dr> aVImSel;
          for (int aKp=0 ; aKp<aNbTot ; aKp++)
          {
              if (aSel.GetNext())
              {
                  aVCPSel.push_back(aVCPTot[aKp]);
                  aVImSel.push_back(aVImTot[aKp]);
              }
          }
          double AltiTest,ProfTest;
          cQual12Param  aQual;
          CamStenope * aSolTest = Camera11Param(aQual,aSzCam,isFraserModel,aVCPSel,aVImSel,AltiTest,ProfTest);
          if (aSolTest)
          {
              std::vector<double> aVDist;
              for (int aKp=0 ; aKp<aNbTot ; aKp++)
              {
                   double aResIm = euclid(aVImTot[aKp]-aSolTest->Ter2Capteur(aVCPTot[aKp]));
                   aVDist.push_back(aResIm);
              }
              double aVPerc = KthValProp(aVDist,aPropInlier);

              double aSomP = 0;
              for (int aKp=0 ; aKp<aNbTot ; aKp++)
              {
                   double aPds = aVDist[aKp] / (aVDist[aKp] +aVPerc);
                   aSomP += aPds;
              }
              double aScore = aVPerc * aSomP;
              if (aScore<aScoreMin)
              {
                  aBestQual = aQual;
                  aScoreMin = aScore;
                  aBestSol = aSolTest;
                  Alti = AltiTest;
                  Prof = ProfTest;
              }
          }
     }

     return aBestSol;
}



static std::string aTCS_PrefGen="TmpChSysCo";
void AffinePose(ElCamera & aCam,const std::vector<Pt2dr> & aVIm,const std::vector<Pt3dr> & aVPts)
{

    std::string aNameCam =  aCam.NameIm() ;
    std::string aDirOriTmp = "-"+aTCS_PrefGen;

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::Glob();
    ELISE_ASSERT(aICNM!=0,"ICNM Null in AffinePose");
    std::string aDir = aICNM->Dir();
    std::string aKeyOriTmp =  "NKS-Assoc-Im2Orient@"+aDirOriTmp;
    std::string aNameOriTmp = aDir+aICNM->Assoc1To1(aKeyOriTmp,aNameCam,true);
    cOrientationConique anOC =  aCam.StdExportCalibGlob(false);
    MakeFileXML(anOC,aNameOriTmp);
                
    cSetOfMesureAppuisFlottants aS2D;
    aS2D.MesureAppuiFlottant1Im().push_back(cMesureAppuiFlottant1Im());
    cMesureAppuiFlottant1Im  & aMAF = aS2D.MesureAppuiFlottant1Im().back();
    aMAF.NameIm() = aNameCam;

    cDicoAppuisFlottant aDic;

    for (int aK=0 ; aK<int(aVIm.size()) ; aK++)
    {
        cOneMesureAF1I aOM;
        aOM.NamePt() = "Pt-"+ToString(aK);
        aOM.PtIm() = aVIm[aK];
        aMAF.OneMesureAF1I().push_back(aOM);

        cOneAppuisDAF  anAp;
        anAp.Pt() = aVPts[aK];
        anAp.NamePt() = aOM.NamePt() ;
        anAp.Incertitude() = Pt3dr(1,1,1);
        aDic.OneAppuisDAF().push_back(anAp);
    }
    std::string aName2D = aDir+aTCS_PrefGen+ aNameCam + "-S2D.xml";
    std::string aName3D = aDir+aTCS_PrefGen+ aNameCam + "-S3D.xml";
    MakeFileXML(aS2D,aName2D);
    MakeFileXML(aDic,aName3D);

    std::string aCom =    MM3dBinFile_quotes( "Apero" )
                        + XML_MM_File("Apero-Optim-ChSysCo-Rot.xml")
                        + " DirectoryChantier=" +  aDir
                        + " +Im=" + aNameCam;

    System(aCom.c_str());



    std::string aKeyOriTmpOut =  "NKS-Assoc-Im2Orient@"+aDirOriTmp +"-OUT";
    std::string aNameOriTmpOut = aDir+aICNM->Assoc1To1(aKeyOriTmpOut,aNameCam,true);



    ElCamera * aCamOut =  CamOrientGenFromFile(aNameOriTmpOut,aICNM);


    aCam.SetOrientation(aCamOut->Orient());

    ELISE_fp::RmFile(aName2D);
    ELISE_fp::RmFile(aName3D);


    ELISE_fp::PurgeDir(aDir+"Ori-TmpChSysCo/",true);
    ELISE_fp::PurgeDir(aDir+"Ori-TmpChSysCo-OUT/",true);


    // getchar();
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
