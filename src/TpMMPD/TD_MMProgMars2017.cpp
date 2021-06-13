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
#include "TpPPMD.h"


/********************************************************************/
/*                                                                  */
/*         cTD_Camera                                               */
/*                                                                  */
/********************************************************************/

/*
   1- Hello word
   2- Lire string points d'appuis + points de liaison
   3- Creer une classe cAppli_BasculeRobuste
   4- Decouvrir le XML, les classe C++, "SetOfMesureAppuisFlottants"

        * include/XML_GEN/ParamChantierPhotogram.xml
        * include/XML_GEN/ParamChantierPhotogram.h

   5-  Ajout orientation + pattern

   6-  Creer les points 3D
*/

class cImageBasculeRobuste;
class cPointBascRobust;
class cAppli_BasculeRobuste;

class cImageBasculeRobuste
{
    public :
        cImageBasculeRobuste
        (
             const std::string  & aNameIm,
             cBasicGeomCap3D *    aCam
        ) :
          mNameIm (aNameIm),
          mCam    (aCam),
          mCamS   (aCam->DownCastCS()),
          mRA     (mCam->ResolutionAngulaire()),
          mC      (mCamS ? mCamS->PseudoOpticalCenter() : Pt3dr(0,0,0) )
          // mP1     (mCam->ImEtProf2Terrain(
        {
        }

        std::string         mNameIm;
        cBasicGeomCap3D *   mCam;
        CamStenope *        mCamS;
        double              mRA;
        Pt3dr               mC;
        Pt3dr               mP1;
};


class cPointBascRobust
{
    public :
        cPointBascRobust(const std::string & aNamePt,const Pt3dr & aPTer) :
           mNamePt (aNamePt),
           mPTer   (aPTer)
        {
        }
    
        Pt3dr PIntRand() ;
        double DistReproj(const Pt3dr & aPLoc,double aMaxEr,double Exp);
        void   Finish(double aMaxEr);

        std::string                            mNamePt;
        Pt3dr                                  mPTer;
        Pt3dr                                  mPInter;
        std::vector<cImageBasculeRobuste *>    mIms;
        std::vector<Pt2dr >                    mPtIms;
        std::vector<ElSeg3D >                  mSegs;
        std::vector<Pt3dr >                    mTgtN;
        double        mDMaxTer;
        double        mMoyDTer;
        double        mDMaxFais;
        double        mMoyDFais;
        std::string   mImWortTer;
        std::string   mImWortFais;
        
};



class cAppli_BasculeRobuste
{
    public :
       cAppli_BasculeRobuste(int argc,char ** argv);
    private :
       bool DoOneTirage();
       cPointBascRobust * RanPBR(); // Biaise

       std::string                   mPatIm;
       std::string                   mOri;
       double                        mMaxEr;

       cSetOfMesureAppuisFlottants   mXML_MesureIm;
       cDicoAppuisFlottant           mXML_MesureTer;
       cElemAppliSetFile             mSetFile;
       cInterfChantierNameManipulateur *            mICNM;

       std::map<std::string,cImageBasculeRobuste *> mDicoIm;
       std::map<std::string,cPointBascRobust *>     mDicoPt;
       std::vector<cPointBascRobust *>              mVecPt;
 
       //  Pour biaiser le tirage mets un point autant de fois qu'il a de (N-1) points
       std::vector<cPointBascRobust *>              mDecupVecPt;


       int                                          mNbPts;
       int                                          mNbTirage;
       double                                       mBestRes;
       cSolBasculeRig                               mBestSol;
       cSolBasculeRig                               mBestSolInv;
       bool                                         mSIFET;
       double                                       mExpos;
       std::string                                  mInPutSol;
       std::string                                  mOut;
       double                                       mRatioAlertB;
};


/****************************************************/
/*                                                  */
/*                cPointBascRobust                  */
/*                                                  */
/****************************************************/
double cPointBascRobust::DistReproj(const Pt3dr & aPLoc,double aMaxEr,double Exp)
{
   double aSomRes=0;
   for (int aKI=0 ; aKI< int(mIms.size()) ; aKI++)
   {
       // double aDStd = euclid(mPtIms[aKI],mIms[aKI]->mCam->Ter2Capteur(aPLoc));
       double aD3 = euclid(vunit(aPLoc-mIms[aKI]->mC)-mTgtN[aKI]) /  mIms[aKI]->mRA;

       double aD = aD3; // Ou DStd, au choix
       aD = aD *aMaxEr /(aD +aMaxEr);
       aD = pow(aD,Exp);
       aSomRes += aD;
   }

   return aSomRes;
}

void cPointBascRobust::Finish(double aMaxEr)
{
    double aDMin = 1e30;
    for (int aK1=0 ; aK1<int(mSegs.size()) ; aK1++)
    {
        for (int aK2=aK1+1 ; aK2<int(mSegs.size()) ; aK2++)
        {
             Pt3dr aP = mSegs[aK1].PseudoInter(mSegs[aK2]);
             double aD =  DistReproj(aP,aMaxEr,1);
             if (aD<aDMin)
             {
                aDMin = aD;
                mPInter = aP;
             }
        }
    }
}

Pt3dr cPointBascRobust::PIntRand() 
{
    int aK1 = NRrandom3(mSegs.size());
    int aK2 = aK1;

     while (aK2== aK1)
           aK2 =  NRrandom3(mSegs.size());

     Pt3dr aRes = mSegs[aK1].PseudoInter(mSegs[aK2]);

     return aRes;
}

/****************************************************/
/*                                                  */
/*           cAppli_BasculeRobuste                  */
/*                                                  */
/****************************************************/

cAppli_BasculeRobuste::cAppli_BasculeRobuste(int argc,char ** argv)  :
     mMaxEr        (10),
     mNbTirage     (30000),
     mBestRes      (1e30),
     mBestSol      (cSolBasculeRig::Id()),
     mBestSolInv   (cSolBasculeRig::Id()),
     mSIFET        (false),
     mExpos        (1),
     mRatioAlertB  (3)
{
    std::string aName2D,aName3D;

    ElInitArgMain
    (
        argc,argv,
        LArgMain()  
                    << EAMC(mPatIm,"Pattern of images")
                    << EAMC(mOri,"Orientation")
                    << EAMC(aName3D,"Name of 3D Point")
                    << EAMC(aName2D,"Name of 2D Points"),
        LArgMain()   << EAM(mNbTirage,"NbRan",true,"Number of random, Def=30000")
                     << EAM(mSIFET,"SIFET",true,"Special export for SIFET benchmark")
                     << EAM(mExpos,"Expos",true,"Exposure for dist, 1->L1, 2->L2 , def=1")
                     << EAM(mOut,"Out",true,"Export computed orientation, Def=BAR+${-Orientation}")
                     << EAM(mRatioAlertB,"RatioAlertB",true,"Ratio of alert for bundle reproj (Def=3)")
                     << EAM(mMaxEr,"MaxEr",true,"Typical max error image reprojection,Def=10")
    );


    mXML_MesureIm =  StdGetFromPCP(aName2D,SetOfMesureAppuisFlottants);
    mXML_MesureTer =  StdGetFromPCP(aName3D,DicoAppuisFlottant);
/*
    aSMAF = StdGetObjFromFile<cSetOfMesureAppuisFlottants>
            (
                aName2D,
                 StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
                 "SetOfMesureAppuisFlottants",
                 "SetOfMesureAppuisFlottants"
             );
*/

     mSetFile.Init(mPatIm);
     mICNM = mSetFile.mICNM;
     const std::vector<std::string> * aVS =  mSetFile.SetIm();
     StdCorrecNameOrient(mOri,mSetFile.mDir);

     std::cout << "PATTERN " << aVS->size()  << " ORI=" << mOri << "\n";


     for 
     (
           std::list<cOneAppuisDAF>::const_iterator itPt = mXML_MesureTer.OneAppuisDAF().begin() ; 
           itPt!=mXML_MesureTer.OneAppuisDAF().end() ; 
           itPt++
     )
     {
          cPointBascRobust * aPBR = new cPointBascRobust(itPt->NamePt(),itPt->Pt());
          mDicoPt[itPt->NamePt()] = aPBR;
          mVecPt.push_back(aPBR); 
          
     }

     for (int aKIm=0 ; aKIm<int(aVS->size()) ; aKIm++)
     {
          std::string aName = (*aVS)[aKIm];
          cBasicGeomCap3D *  aCam = mICNM->StdCamGenerikOfNames(mOri,aName);
          mDicoIm[aName] = new cImageBasculeRobuste(aName,aCam);
     }


     for 
     (
          std::list<cMesureAppuiFlottant1Im>::const_iterator itMAF = mXML_MesureIm.MesureAppuiFlottant1Im().begin();
          itMAF != mXML_MesureIm.MesureAppuiFlottant1Im().end();
          itMAF++
     )
     {
         const cMesureAppuiFlottant1Im & aMAF = *itMAF;
         cImageBasculeRobuste * aImBR = mDicoIm[aMAF.NameIm()];
         if (aImBR !=0)
         {
             for 
             (
                 std::list<cOneMesureAF1I>::const_iterator itM= aMAF.OneMesureAF1I().begin();
                 itM != aMAF.OneMesureAF1I().end();
                 itM++
             )
             {
                cPointBascRobust * aPtBR = mDicoPt[itM->NamePt()];           
                if (aPtBR!=0)
                {
                    aPtBR->mPtIms.push_back(itM->PtIm());
                    aPtBR->mIms.push_back(aImBR);
                    aPtBR->mSegs.push_back(aImBR->mCam->Capteur2RayTer(itM->PtIm()));
                    aPtBR->mTgtN.push_back(aPtBR->mSegs.back().TgNormee());
                }
             }
         }
     }

     mNbPts = mVecPt.size();
     for (int aKP= 0 ; aKP<mNbPts  ; aKP++)
     {
          int aNb = mVecPt[aKP]->mSegs.size() -1;
          if (aNb>=0)
          {
             for (int aK=0 ; aK<aNb ; aK++)
             {
                 mDecupVecPt.push_back(mVecPt[aKP]);
             }      
          }
          mVecPt[aKP]->Finish(mMaxEr);
     }
     
     //  std::cout << "Bbbbbbbbbbbbbbbbbbbbbb\n";
     for (int aKT=0 ; aKT<mNbTirage ; )
     {
     //  std::cout << "AAAAaaaaaaaaaaaaaaa\n";
         if (DoOneTirage()) 
            aKT++;
     }


     FILE * aFP = FopenNN(mSetFile.mDir+"ResulBar.txt","w","cAppli_BasculeRobuste");

     // double aMed = 0;
     // Premiere fois pour avoir les stats, deuxieme pour les sortir
     Pt3dr aSomEcAbs(0,0,0);
     Pt3dr aSomEc(0,0,0);
     int aNbPtsOk=0;

     cPointBascRobust * aPMaxReprojTer = 0;
     cPointBascRobust * aPMaxReprojBun = 0;
     double aDistMedBun=0;
     for (int aNbStep = 0 ; aNbStep<4 ; aNbStep++)
     {
         
         bool aPrintGlob = (aNbStep==1);
         bool aPrintPt = (aNbStep==2);
         bool aPrintIm = (aNbStep==3);

         if (aPrintGlob && mNbPts)
         {
             aSomEcAbs = aSomEcAbs / aNbPtsOk;
             aSomEc    = aSomEc    / aNbPtsOk;
             fprintf(aFP," ================== Global Stat  ========================== \n");
             fprintf(aFP," ================== Global Stat  ========================== \n");
             fprintf(aFP,"   Aver Dif    Bundle/Ter (%lf %lf %lf )\n",aSomEc.x,aSomEc.y,aSomEc.z);
             fprintf(aFP,"   Aver AbsDif Bundle/Ter (%lf %lf %lf )\n",aSomEcAbs.x,aSomEcAbs.y,aSomEcAbs.z);
          if (aPMaxReprojTer)
             fprintf(aFP,"   Worst reproj Ter   %lf for pt %s on image %s\n",
                          aPMaxReprojTer->mDMaxTer,
                          aPMaxReprojTer->mNamePt.c_str(),
                          aPMaxReprojTer->mImWortTer.c_str()
             );
          if (aPMaxReprojBun)
             fprintf(aFP,"   Worst reproj Bundle %lf for pt %s on image %s\n",
                          aPMaxReprojBun->mDMaxFais,
                          aPMaxReprojBun->mNamePt.c_str(),
                          aPMaxReprojBun->mImWortTer.c_str()
             );
         }

         aSomEcAbs = Pt3dr(0,0,0);
         aSomEc    = Pt3dr(0,0,0);
         aNbPtsOk=0;
         if (aPrintPt)
         {
              fprintf(aFP,"\n ================== Stat per point ========================== \n");
                fprintf(aFP," ================== Stat per point ========================== \n\n");
                fprintf(aFP,"    ---------------------------------------------- \n");
                fprintf(aFP,"[NamePt] :  \n");
                fprintf(aFP,"   D=Dist(GCP,Bundle) : GCP-Bundle\n");
                fprintf(aFP,"   Worst ReprojTer ReprojBundle ");
                fprintf(aFP," ImWorsTer ImWorstBundle \n");
                fprintf(aFP,"   Aver   ReprojTer ReprojBundle  on NbIma \n");
                fprintf(aFP,"    ---------------------------------------------- \n");
         }
         if (aPrintIm)
         {
                fprintf(aFP,"\n ================== Detail per point & image ========================== \n");
                fprintf(aFP," ================== Detail per point & image ========================== \n\n");
                fprintf(aFP,"    ---------------------------------------------- \n");
                fprintf(aFP,"[NamePt] \n");
                fprintf(aFP,"    | ReprojTer | ReprojBundle | Image\n");
                fprintf(aFP,"    | ReprojTer | ReprojBundle | Image\n");
                fprintf(aFP,"    .... \n");
                fprintf(aFP,"    ---------------------------------------------- \n");
         }
         // bool SIFET = false;
         std::vector<double> aVDIstBund;
         for (int aKP=0 ; aKP<mNbPts ; aKP++)
         {
             cPointBascRobust * aPBR = mVecPt[aKP];
             int aNbIm = aPBR->mIms.size();

             if (aNbIm>=2)
             {
                 aNbPtsOk ++;
                 Pt3dr aPIntAv = aPBR->mPInter;
                 Pt3dr aPIntAp =  mBestSolInv(aPIntAv);
                 Pt3dr anEcart = aPBR->mPTer-aPIntAp;
                 double aDist = euclid(anEcart);
                 aSomEc = aSomEc+anEcart;
                 aSomEcAbs = aSomEcAbs+Pt3dr(ElAbs(anEcart.x),ElAbs(anEcart.y),ElAbs(anEcart.z));

                 if (aPrintPt)
                 {
                    if (mSIFET)
                    {
                      fprintf(aFP,"%s %.3f %.3f %.3f\n",aPBR->mNamePt.c_str(),aPIntAv.x,aPIntAv.y,aPIntAv.z);
                    }
                    else
                    {
                       fprintf(aFP,"[%s] :  \n",aPBR->mNamePt.c_str());
                       fprintf(aFP,"   D=%f : (%f %f %f)  \n",
                            aDist, anEcart.x,anEcart.y,anEcart.z
                           );
                       fprintf(aFP,"   Worst %07.3f  %07.3f ",aPBR->mDMaxTer,aPBR->mDMaxFais);
                       fprintf(aFP," %s  %s \n",aPBR->mImWortTer.c_str(),aPBR->mImWortFais.c_str());
                       fprintf(aFP,"   Aver  %07.3f  %07.3f on %d \n",aPBR->mMoyDTer,aPBR->mMoyDFais,aNbIm);
                    }
                 }
             }
             if (aPrintIm && aNbIm)
             {
                if (!mSIFET) 
                {
                    fprintf(aFP,"[%s]\n",aPBR->mNamePt.c_str());
                }
             }
                
             if (aNbStep==0)
             {
                aPBR->mDMaxTer = -1;
                aPBR->mDMaxFais = -1;
             }
             aPBR->mMoyDTer = 0.0;
             aPBR->mMoyDFais = 0.0;
             Pt3dr aPLoc = mBestSol(aPBR->mPTer);
             for (int aKI=0 ; aKI< aNbIm; aKI++)
             {
                  double aDTer = euclid(aPBR->mPtIms[aKI],aPBR->mIms[aKI]->mCam->Ter2Capteur(aPLoc));
                  double aDFais = euclid(aPBR->mPtIms[aKI],aPBR->mIms[aKI]->mCam->Ter2Capteur(aPBR->mPInter));
                  aVDIstBund.push_back(aDFais);
                  std::string aNameIm = aPBR->mIms[aKI]->mNameIm;

                   if (aNbIm==1) aDFais = -1;

                  if (aPrintIm & (!mSIFET))
                  {
                     std::string aMes = "    ";
                     if (aNameIm==aPBR->mImWortTer)
                        aMes[1] = '#';
                     if (aNameIm==aPBR->mImWortFais)
                        aMes[2] = '@';
                     if (aDFais > aDistMedBun * mRatioAlertB)
                        aMes[3] = '+';


                     fprintf(aFP," %s | %07.3f | %07.3f | %s\n",aMes.c_str(),aDTer,aDFais,aNameIm.c_str());
                  }
                  if (aNbStep==0)
                  {
                     if (aPBR->mDMaxTer<aDTer)
                     {
                         aPBR->mDMaxTer = aDTer;
                         aPBR->mImWortTer = aNameIm;
                     }
                     if (aPBR->mDMaxFais<aDFais)
                     {
                         aPBR->mDMaxFais = aDFais;
                         aPBR->mImWortFais = aNameIm;
                     }
                  }

                  aPBR->mMoyDTer += aDTer;
                  aPBR->mMoyDFais += aDFais;
              }

              if (aNbStep==0)
              {
                 if ((aPMaxReprojTer==0)  || (aPMaxReprojTer->mDMaxTer < aPBR->mDMaxTer))
                    aPMaxReprojTer =  aPBR;
                 if ((aPMaxReprojBun==0)  || (aPMaxReprojBun->mDMaxFais < aPBR->mDMaxFais))
                    aPMaxReprojBun =  aPBR;
              }

              if (aNbIm>=2)
              {
                 aPBR->mMoyDFais /= (aNbIm-1);
              }
              else
              {
                  aPBR->mMoyDFais  = -1;
                  aPBR->mDMaxFais  = -1;
              }

              if (aNbIm>=1)
              {
                 aPBR->mMoyDTer /= aNbIm;
              }
              
              
          }
          if (aVDIstBund.size())
             aDistMedBun = MedianeSup(aVDIstBund);
     }


     fclose(aFP);

     {
        if (! EAMIsInit(&mOut))
           mOut = "BAR_"+ mOri;

        std::string aNameBasc = "Basc-"+  mOri  +"-2-"+ mOut + ".xml";
        MakeFileXML(EL2Xml(mBestSolInv),aNameBasc);

        std::string aComBasc = MM3dBinFile_quotes("GCPBascule")
                                + " " + QUOTE( mPatIm)
                                + " " + mOri
                                + " " + mOut
                                + " " + aName3D
                                + " " + aName2D
                                + " ForceSol=" +aNameBasc;
        System(aComBasc);
     }
}


cPointBascRobust * cAppli_BasculeRobuste::RanPBR() // Biaise
{
     return  mDecupVecPt[NRrandom3(mDecupVecPt.size())];
}




bool cAppli_BasculeRobuste::DoOneTirage()
{
   cPointBascRobust * aPB1 =  RanPBR();
   cPointBascRobust * aPB2 =  RanPBR();
   cPointBascRobust * aPB3 =  RanPBR();

   if ((aPB1==aPB2) || (aPB1==aPB3) || (aPB2==aPB3))
      return false;

  
   cSolBasculeRig aTer2Loc = SolElemBascRigid
                             (
                                   aPB1->mPTer,      aPB2->mPTer,      aPB3->mPTer,
                                   aPB1->PIntRand(), aPB2->PIntRand(), aPB3->PIntRand()
                             );

   double aSomRes = 0.0;
   int aNbRes=0;
   for (int aKP=0 ; aKP<int(mVecPt.size()) ; aKP++)
   {
       cPointBascRobust * aPBR = mVecPt[aKP];
       Pt3dr aPLoc = aTer2Loc(aPBR->mPTer);
       aSomRes += aPBR->DistReproj(aPLoc,mMaxEr,mExpos);
       aNbRes += aPBR->mIms.size();
   }

   aSomRes /= aNbRes;

   if (aSomRes< mBestRes)
   {
      mBestRes = aSomRes;
      mBestSol = aTer2Loc;
      mBestSolInv = aTer2Loc.Inv();
      std::cout<< "RES "  << mBestRes << "\n";
   }


   return true;
}



/*
{

     std::vector<cPointBascRobust *>              aVecOk;
     for (int aKP= 0 ; aKP<int(mVecPt.size())  ; aKP++)
     {
          if (mVecPt[aKP]->mSegs.size() >=2)
          {
               bool Ok;
               mVecPt[aKP]->mPInter = InterSeg(mVecPt[aKP]->mSegs,Ok);
               if (Ok)
                  aVecOk.push_back(mVecPt[aKP]);
          }
     }
     mVecPt = aVecOk;
     mNbPts = mVecPt.size();

     double aPropTest = 0.8;
     int aNbPtsTest = round_ni(mNbPts*aPropTest);
     double aDistMin = 1e7;

     cSolBasculeRig aBestSol = cSolBasculeRig::Id();
     for (int aTir=0 ; aTir<mNbTirage  ; aTir++ )
     {
         int aK1 = NRrandom3(mNbPts);
         int aK2 = NRrandom3(mNbPts);
         int aK3 = NRrandom3(mNbPts);

         if ((aK1!=aK2) && (aK1!=aK3) && (aK2!=aK3))
         {

             cSolBasculeRig aSol = SolElemBascRigid
                                   (
                                       mVecPt[aK1]->mPInter, mVecPt[aK2]->mPInter, mVecPt[aK3]->mPInter,
                                       mVecPt[aK1]->mPTer, mVecPt[aK2]->mPTer, mVecPt[aK3]->mPTer
                                   );
             aTir++;

             std::vector<double> aVDist;
             for (int aKp=0 ; aKp<mNbPts ; aKp++)
             {
                 aVDist.push_back(euclid(aSol(mVecPt[aKp]->mPInter)-mVecPt[aKp]->mPTer));
             }
             std::sort(aVDist.begin(),aVDist.end());
             double aSomD = 0.0;
             for (int aK=0 ; aK<aNbPtsTest ; aK++)
                 aSomD += aVDist[aK];

             if (aSomD<aDistMin)
             {
                 aDistMin=aSomD;
                 aBestSol = aSol;
                 std::cout << "DIST MIN " << aDistMin << "\n";
             }
             
         }
     }
     for (int aKp=0 ; aKp<mNbPts ; aKp++)
     {
         Pt3dr aDif = aBestSol(mVecPt[aKp]->mPInter)-mVecPt[aKp]->mPTer;
         std::cout << "Pt="  << mVecPt[aKp]->mNamePt 
                   << " " << euclid(aDif) << "   " << aDif  << "\n";
     }
}
*/


cSolBasculeRig SolElemBascRigid
               (
                    const Pt3dr & aAvant1, const Pt3dr & aAvant2, const Pt3dr & aAvant3,
                    const Pt3dr & aApres1, const Pt3dr & aApres2, const Pt3dr & aApres3
               )
{

   // std::cout << "AVVVVVVVV " << aAvant1 << aAvant2 <<  aAvant3 << "\n";
   // std::cout << "Apppppppp " << aApres1 << aApres2 <<  aApres3 << "\n";

   cRansacBasculementRigide aRBR(false);

   aRBR.AddExemple(aAvant1,aApres1,0,"P1");
   aRBR.AddExemple(aAvant2,aApres2,0,"P2");
   aRBR.AddExemple(aAvant3,aApres3,0,"P3");

   aRBR.CloseWithTrGlob();

   aRBR.ExploreAllRansac();

   return aRBR.BestSol();
}



int BBB_main(int argc,char ** argv)
{
    cAppli_BasculeRobuste anAppli(argc,argv);
    return EXIT_SUCCESS;
}


int BasculeRobuste_main(int argc,char ** argv)
{
    cAppli_BasculeRobuste anAppli(argc,argv);
    return EXIT_SUCCESS;
}




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
