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


#include "NewRechPH.h"

std::string NameFileNewPCarac(const std::string & aNameGlob,bool Bin,const std::string & anExt)
{
    std::string aDirGlob = DirOfFile(aNameGlob);
    std::string aDirLoc= "NewPH" + anExt + "/";
    ELISE_fp::MkDirSvp(aDirGlob+aDirLoc);
    return aDirGlob+aDirLoc + NameWithoutDir(aNameGlob) + (Bin ? ".dmp" : ".xml");
}


void  cAppli_NewRechPH::Clik()
{
   if (mW1) mW1->clik_in();
}

void cAppli_NewRechPH::AddScale(cOneScaleImRechPH * aI1,cOneScaleImRechPH *)
{
    mVI1.push_back(aI1);
}

void cAppli_NewRechPH::AddBrin(cBrinPtRemark * aBr)
{
   mVecB.push_back(aBr);
}



cPtSc CreatePtSc(const Pt2dr & aP,double aSc)
{
    cPtSc aRes;
    aRes.Pt()    = aP;
    aRes.Scale() = aSc;
    return aRes;
}

double Som_K_ApK(int aN,double a)
{
    double aSom = 0;
    for (int aK=1 ; aK<= aN ; aK++)
        aSom +=  pow(a,aK) * aK;
    return aSom;
}

/*
void TestSom(int aN,double a)
{
    // Verifie Som k=1,N  { a^k * k}

     double aCheck = Som_K_ApK(aN,a);
     // double aFormul = ((1-pow(a,aN+1)
}
*/
/*
double Sigma2FromFactExp(double a);
double FactExpFromSigma2(double aS2);


void TestSigma2(double a)
{
   int aNb = 100 + 100/(1-a);
   double aSom=0;
   double aSomK2=0;
   for (int aK=-aNb ; aK<= aNb ; aK++)
   {
       double aPds = pow(a,ElAbs(aK));
       aSom += aPds;
       aSomK2 += aPds * ElSquare(aK);
   }

   double aSigmaExp =  aSomK2 / aSom;
   // double aSigmaTh = (2*a) /((1+a) * ElSquare(1-a));
   // double aSigmaTh = (2*a) /(ElSquare(1-a));
   double aSigmaTh = Sigma2FromFactExp(a);

   double aATh = FactExpFromSigma2(aSigmaTh);

   std::cout << "TestSigma2 " << aSigmaExp << " " << aSigmaTh/aSigmaExp - 1 << " aaa " << a << " " << aATh << "\n";
}
*/

cAppli_NewRechPH::cAppli_NewRechPH(int argc,char ** argv,bool ModeTest) :
    mPowS        (pow(2.0,1/5.0)),
    mNbS         (40),
    mNbSR        (7),
    mDeltaSR     (2),
    mMaxLevR     (mNbS - (mNbSR-1) * mDeltaSR),
    mS0          (1.0),
    mScaleStab   (4.0),
    mSeuilAC     (0.95),
    mSeuilCR     (0.6),
    mW1          (0),
    mModeTest    (ModeTest),
    mDistMinMax  (3.0),
    mDoMin       (true),
    mDoMax       (true),
    mDoPly       (false),
    mPlyC        (0),
    mHistLong    (1000,0),
    mHistN0      (1000,0),
    mExtSave     ("Std"),
    mBasic       (false),
    mAddModeSift (true),
    mAddModeTopo (true),
    mLapMS       (false),
    mTestDirac   (false),
    mSaveFileLapl (false),
    // mPropCtrsIm0 (0.1),
    mNbSpace           (0),
    mNbScaleSpace      (0),
    mNbScaleSpaceCstr  (0),
    mDistAttenContr    (100.0),
    mPropContrAbs      (0.3),
    mSzContrast        (2),
    mPropCalcContr     (0.05),
    mIm0               (1,1),
    mTIm0              (mIm0),
    mImContrast        (1,1),
    mTImContrast       (mImContrast)
{
   cSinCardApodInterpol1D * aSinC = new cSinCardApodInterpol1D(cSinCardApodInterpol1D::eTukeyApod,5.0,5.0,1e-4,false);
   mInterp = new cTabIM2D_FromIm2D<tElNewRechPH>(aSinC,1000,false);

/*
   TestSigma2(0.1);
   TestSigma2(0.5);
   TestSigma2(0.9);
   TestSigma2(0.95);
   TestSigma2(0.99);
   getchar();
*/

   double aSeuilPersist = 1.0;

   MMD_InitArgcArgv(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mName, "Name Image",  eSAM_IsPatFile),
         LArgMain()   << EAM(mPowS, "PowS",true,"Scale Pow")
                      << EAM(mNbS,  "NbS",true,"Number of level")
                      << EAM(mS0,   "S0",true,"ScaleInit, Def=1")
                      << EAM(mDoPly, "DoPly",true,"Generate ply file, for didactic purpose")
                      << EAM(mBox, "Box",true,"Box for computation")
                      << EAM(mModeTest, "Test",true,"if true add W")
                      << EAM(aSeuilPersist, "SP",true,"Threshold persistance")
                      << EAM(mBasic, "Basic",true,"Basic")
                      << EAM(mAddModeSift, "Sift",true,"Add SIFT Mode")
                      << EAM(mAddModeTopo, "Topo",true,"Add Topo Mode")

                      << EAM(mLapMS, "LapMS",true,"MulScale in Laplacian, def=false")
                      << EAM(mTestDirac, "Dirac",true,"Test with dirac image")
                      << EAM(mSaveFileLapl, "SaveLapl",true,"Save Laplacian file, def=false")
                      << EAM(mScaleStab, "SS",true,"Scale of Stability")
   );

   if (! EAMIsInit(&mExtSave))
   {
        mExtSave  = mBasic ? "Basic" : "Std";
   }
   if (! EAMIsInit(&mNbS))
   {
       if (mBasic) 
           mNbS = 1;
   }
   mNbInOct = log(2) / log(mPowS);

   if (mDoPly)
   {
      mPlyC = new cPlyCloud;
   }

   Pt2di aP0(0,0);
   Pt2di aP1 = mTestDirac ? Pt2di(1000,1000) : Pt2di(-1,-1);
   if (EAMIsInit(&mBox))
   {
       aP0 = mBox._p0;
       aP1 = mBox._p1;
   }
   // Create top scale
   AddScale(cOneScaleImRechPH::FromFile(*this,mS0,mName,aP0,aP1),nullptr);

   // Create matr of link, will have do it much less memory consuming (tiling of list ?)
   mIm0         = mVI1.back()->Im();
   mTIm0        = tTImNRPH(mIm0);
   mSzIm = mIm0.sz();
   mImContrast  = tImNRPH(mSzIm.x,mSzIm.y);
   mTImContrast = tTImNRPH(mImContrast);
   ComputeContrast();
   mBufLnk  = std::vector<std::vector<cPtRemark *> >(mSzIm.y,std::vector<cPtRemark *>(mSzIm.x,(cPtRemark *)0));

   double aScaleMax = mS0*pow(mPowS,mNbS);
   // Used for  nearest point researh
   mVoisLnk = SortedVoisinDisk(-1,aScaleMax+4,true);


   if (! EAMIsInit(&mDZPlyLay))
   {
      mDZPlyLay = ElMin(mSzIm.x,mSzIm.y)/ double(mNbS);
   }
   if (mModeTest)
   {
      mW1 = Video_Win::PtrWStd(mSzIm);
   }
   // mVI1.back()->Show(mW1);

   for (int aK=0 ; aK<mNbS ; aK++)
   {
        // Init from low resol
        if (aK!=0)
        {
           AddScale
           (
              cOneScaleImRechPH::FromScale(*this,*mVI1.back(),mS0*pow(mPowS,aK)),
              0
           );

        }
        std::cout << "DONE SCALE " << aK << " on " << mNbS << "\n";
   }
   cSetPCarac aSPC;
   if (mAddModeSift)
   {
       for (int aK=0 ; aK<mNbS-1 ; aK++)
       {
            mVI1[aK]->SiftMakeDif(mVI1[aK+1]);
       }
       for (int aK=1 ; aK<mNbS-1 ; aK++)
       {
            mVI1[aK]->SiftMaxLoc(mVI1[aK-1],mVI1[aK+1],aSPC);
       }
   }
   if (mAddModeTopo)
   {
       for (int aK=0 ; aK<mNbS ; aK++)
       {
            // Compute point of scale
            mVI1[aK]->CalcPtsCarac(mBasic);
            mVI1[aK]->Show(mW1);
          
            // Links the point at different scale
            if (aK!=0)
            {
               mVI1[aK]->CreateLink(*(mVI1[aK-1]));
            }
            std::cout << "DONE CARAC " << aK << " on " << mNbS << "\n";
       }


       Clik();

       for (int aK=0 ; aK<mNbS ; aK++)
       {
           mVI1[aK]->Export(aSPC,mPlyC);
       }

       if (mPlyC)
       {
           mPlyC->PutFile("NewH.ply");
       }

   }

   for (auto & aPt : aSPC.OnePCarac())
       aPt.OK() = true;

   std::list<cOnePCarac> aNewL;
   for (auto & aPt : aSPC.OnePCarac())
   {
       if (aPt.OK())  
          ComputeContrastePt(aPt);
       // 
       if (aPt.OK())
       {
          mVI1[aPt.NivScale()]->ComputeDirAC(aPt);
       }
       //  ComputeDirAC(cBrinPtRemark &);

       if (aPt.OK())
       {
          mVI1[aPt.NivScale()]->AffinePosition(aPt);
       }

       if (aPt.OK())
       {
          CalvInvariantRot(aPt);
       }

       // Put in global coord
       aPt.Pt() =  aPt.Pt() + Pt2dr(aP0);
       if (aPt.OK())
          aNewL.push_back(aPt);
   }
   aSPC.OnePCarac() = aNewL;


   // MakeFileXML(aSPC,NameFileNewPCarac(mName,true,mExtSave));
   MakeFileXML(aSPC,NameFileNewPCarac(mName,true,mExtSave));
}

bool  cAppli_NewRechPH::ComputeContrastePt(cOnePCarac & aPt)
{
   
   Symb_FNum aFI0  (mIm0.in_proj());
   Symb_FNum aFPds (1.0);
   double aS0,aS1,aS2;
   ELISE_COPY
   (
       disc(aPt.Pt(),aPt.Scale()),
       Virgule(aFPds,aFPds*aFI0,aFPds*Square(aFI0)),
       Virgule(sigma(aS0),sigma(aS1),sigma(aS2))
   );

   aS1 /= aS0;
   aS2 /= aS0;
   aS2 -= ElSquare(aS1);
   aS2 = sqrt(ElMax(aS2,1e-10)) * (aS0 / (aS0-1.0));
   aPt.Contraste() = aS2;
   aPt.ContrasteRel() = aS2 / mTImContrast.getproj(round_ni(aPt.Pt()));
    
   aPt.OK() = aPt.ContrasteRel() > mSeuilCR;

   return aPt.OK();
}

bool  cAppli_NewRechPH::CalvInvariantRot(cOnePCarac & aPt)
{
   if (aPt.NivScale() >= mMaxLevR) 
   {
      return aPt.OK() = false;
   }

   return true;
}


void cAppli_NewRechPH::ComputeContrast()
{
   Symb_FNum aFI0  (mIm0.in_proj());
   Symb_FNum aFPds (1.0);
   Symb_FNum aFSom (rect_som(Virgule(aFPds,aFPds*aFI0,aFPds*Square(aFI0)),mSzContrast));

   Symb_FNum aS0 (aFSom.v0());
   Symb_FNum aS1 (aFSom.v1()/aS0);
   Symb_FNum aS2 (Max(0.0,aFSom.v2()/aS0 -Square(aS1)));

   tImNRPH aImC0  (mSzIm.x,mSzIm.y);
   double aNbVois = ElSquare(1+2*mSzContrast);
   // compute variance of image
   ELISE_COPY
   (
        mIm0.all_pts(),
       // ect_max(mIm0.in_proj(),mSzContrast)-rect_min(mIm0.in_proj(),mSzContrast),
        sqrt(aS2) * (aNbVois/(aNbVois-1.0)),
        mImContrast.out() | aImC0.out()
   );
   

   // Calcul d'une valeur  moyenne robuste
   std::vector<double> aVC;
   int aStep = 2*mSzContrast+1;
   for (int aX0=0 ; aX0<mSzIm.x ; aX0+= aStep)
   {
      for (int aY0=0 ; aY0<mSzIm.y ; aY0+= aStep)
      {
          int aX1 = ElMin(aX0+aStep,mSzIm.x);
          int aY1 = ElMin(aY0+aStep,mSzIm.y);
          // Calcul de la moyenne par carre
          double aSom = 0.0;
          double aSomF = 0.0;
          for (int aX=aX0 ; aX<aX1; aX++)
          {
              for (int aY=aY0 ; aY<aY1; aY++)
              {
                  aSom++;
                  aSomF += mTImContrast.get(Pt2di(aX,aY));
              }
          }
          aVC.push_back(aSomF/aSom);
      }
   }
   double aV0 = KthValProp(aVC,mPropCalcContr);
   double aV1 = KthValProp(aVC,1.0-mPropCalcContr);

   double aSom=0.0;
   double aSomF=0.0;
   for (const auto & aV : aVC)
   {
       if ((aV>=aV0) && (aV<=aV1))
       {
           aSom  ++;
           aSomF += aV;
       }
   }
   double aMoy = aSomF / aSom;
   double aFact = 1.0-1.0/mDistAttenContr;
   FilterExp(mImContrast,aFact);
   Im2D_REAL4 aIP1(mSzIm.x,mSzIm.y,1.0);
   FilterExp(aIP1,aFact);
   ELISE_COPY(mImContrast.all_pts(),mImContrast.in()/aIP1.in(),mImContrast.out());

   ELISE_COPY
   (
      mImContrast.all_pts(),
      mImContrast.in()*(1-mPropContrAbs)+mPropContrAbs*aMoy,
      mImContrast.out()
   );
   std::cout << "MOY = " << aMoy << "\n";


   if (1)
   {
      Tiff_Im::CreateFromIm(aImC0,"ImC0.tif");
      Tiff_Im::CreateFromIm(mImContrast,"ImSeuilContraste.tif");
      Tiff_Im::CreateFromFonc("ImCRatio.tif",mSzIm,aImC0.in()/Max(1e-10,mImContrast.in()),GenIm::real4);
   }
}




bool cAppli_NewRechPH::OkNivStab(int aNiv)
{
   return mVI1.at(aNiv)->Scale() >= mScaleStab;
}

bool cAppli_NewRechPH::Inside(const Pt2di & aP) const
{
    return (aP.x>=0) && (aP.y>=0) && (aP.x<mSzIm.x) && (aP.y<mSzIm.y);
}

tPtrPtRemark &  cAppli_NewRechPH::PtOfBuf(const Pt2di & aP)
{
     
    ELISE_ASSERT(Inside(aP),"cAppli_NewRechPH::PtOfBuf"); 

    return mBufLnk[aP.y][aP.x];
}

double  cAppli_NewRechPH::DistMinMax(bool Basic) const  
{
   if (Basic)
   {
       return  60;
   }
   return mDistMinMax;
}




tPtrPtRemark  cAppli_NewRechPH::NearestPoint(const Pt2di & aP,const double & aDist)
{
   double aD2 = ElSquare(aDist);
   for (int aKV=0 ; aKV<int(mVoisLnk.size()) ; aKV++)
   {
       const Pt2di & aVois = mVoisLnk[aKV];
       if (square_euclid(aVois) > aD2)
          return 0;
       Pt2di aPV = aP + aVois;
       if (Inside(aPV))
       {
           tPtrPtRemark  aRes = mBufLnk[aPV.y][aPV.x];
           if (aRes) return aRes;
       }
   }
   return 0;
}

const Pt2di & cAppli_NewRechPH::SzIm() const  {return mSzIm;}

double cAppli_NewRechPH::ScaleOfNiv(const int & aNiv) const
{
   return mVI1.at(aNiv)->Scale();
}

bool cAppli_NewRechPH::OkNivLapl(int aNiv)
{
   return (aNiv < int(mVI1.size())-2) ;
}

double cAppli_NewRechPH::GetLapl(int aNiv,const Pt2di & aP,bool &Ok)
{
   Ok = false;
   if (! OkNivLapl(aNiv))
      return 0;
   double aV1 = mVI1.at(aNiv)->GetVal(aP,Ok);
   if (!Ok)  return 0;
   double aV2 = mVI1.at(aNiv+1)->GetVal(aP,Ok);
   if (!Ok)  return 0;
   return aV1 - aV2;
}




int Test_NewRechPH(int argc,char ** argv)
{
   cAppli_NewRechPH anAppli(argc,argv,false);


   return EXIT_SUCCESS;

}

/*
int Generate_ImagSift(int argc,char ** argv)
{
     Pt2di aSz(1000,1000);
     Im2D_REAL4 aIm(aSz.x,aSz.y);

     for (int aKx=0 ; aKx<10 ; aKx++)
     {
         for (int aKy=0 ; aKy<10 ; aKy++)
         {
             Pt2di aP0(aKx*100,aKy*100);
             Pt2di aP1((aKx+1)*100,(aKy+1)*100);
             Pt2dr aMil = Pt2dr(aP0+aP1) / 2.0;

             double aSigma = (0.5*aKx+1.5*aKy+1);
             double aSign = ((aKx+aKy) % 2)   ? 1 : -1;

             ELISE_COPY
             (
                  rectangle(aP0,aP1),
                  128 * (1+aSign * exp(-  ( Square(FX-aMil.x) + Square(FY-aMil.y)) / Square(aSigma))),
                  aIm.out()
             );

         }
     }
     Tiff_Im::CreateFromIm(aIm,"TestSift.tif");
}
*/


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
aooter-MicMac-eLiSe-25/06/2007*/
