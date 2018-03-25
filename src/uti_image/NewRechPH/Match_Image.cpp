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

class cAFM_Im;
class cAFM_Im_Master;
class cAFM_Im_Sec ;
class cAppli_FitsMatch1Im;

//================================

// NKS-Assoc-CplIm2Hom


class cAFM_Im
{
     public :
         friend class cAFM_Im_Master;
         friend class cAFM_Im_Sec;

         cAFM_Im (const std::string  &,cAppli_FitsMatch1Im &);
         ~cAFM_Im ();
     protected :
         cAppli_FitsMatch1Im & mAppli;
         std::string mNameIm;
         cSetPCarac                             mSetPC;
         std::vector<std::vector<cCompileOPC> > mVOPC;
         std::vector<cCompileOPC> &             mVIndex;
         
};

class cAFM_Im_Master : public  cAFM_Im
{
     public :
         cAFM_Im_Master (const std::string  &,cAppli_FitsMatch1Im &);
         void Match(cAFM_Im_Sec &);

         std::vector<std::vector<cCompileOPC *> > mVTabIndex;
};


class cAFM_Im_Sec : public  cAFM_Im
{
     public :
         cAFM_Im_Sec (const std::string  &,cAppli_FitsMatch1Im &);
};
         


class cAppli_FitsMatch1Im
{
     public :
          cAppli_FitsMatch1Im(int argc,char ** argv);
          const std::string &   ExtNewH () const {return    mExtNewH;}
          const cFitsParam & FitsPm() const {return mFitsPm;}
          std::string NameCple(const std::string & aN1,const std::string & aN2) const;
          int NbBIndex() const;
          int ThreshBIndex() const;

     private :

          cFitsParam         mFitsPm;
          std::string        mNameMaster;
          std::string        mPatIm;
          cElemAppliSetFile  mEASF;
          cAFM_Im_Master *   mImMast;
          cAFM_Im_Sec *      mCurImSec;
          std::string        mNameXmlFits;
          std::string        mExtNewH;
          std::string        mSH;
          std::string        mPostHom;
          int                mNbBIndex;
          int                mThreshBIndex;
};

/*************************************************/
/*                                               */
/*             ::                                */
/*                                               */
/*************************************************/

void InitOneLabelFitsPm(cFitsOneBin & aFB,const std::string & aDir,eTypePtRemark aLab)
{
    std::string aName = aDir + aFB.PrefName() +  eToString(aLab) +  aFB.PostName().Val();


    cCompCB aCB = StdGetFromNRPH(aName,CompCB);
    aFB.CCB().SetVal(aCB);

    // std::cout <<   aCB.CompCBOneBit().size()  << "  "  << aName << "\n";
}

void InitOneFitsPm(cFitsOneLabel & aFOL,const std::string & aDir,eTypePtRemark aLab)
{
    InitOneLabelFitsPm(aFOL.BinIndexed(),aDir,aLab);
    ELISE_ASSERT(aFOL.BinIndexed().CCB().Val().CompCBOneBit().size()<=16,"InitOneFitsPm");
    InitOneLabelFitsPm(aFOL.BinDecision(),aDir,aLab);
}

const  std::string TheDirXmlFits=    string("include")    + ELISE_CAR_DIR
                                      + string("XML_MicMac") + ELISE_CAR_DIR 
                                      + string("Fits")       + ELISE_CAR_DIR;
const  std::string DefNameFitParam =  "FitsParam.xml";

void InitFitsPm(cFitsParam & aFP,const std::string & aDir, const std::string & aName)
{
    aFP = StdGetFromNRPH(aDir+aName,FitsParam);
    InitOneFitsPm(aFP.OverLap(),aDir,aFP.KindOl());
}

void StdInitFitsPm(cFitsParam & aFP)
{
    InitFitsPm(aFP,MMDir() + TheDirXmlFits,DefNameFitParam);
}
/*************************************************/
/*                                               */
/*           cAFM_Im                             */
/*                                               */
/*************************************************/

bool CmpCPOC(const cCompileOPC &aP1,const  cCompileOPC &aP2)
{
   return aP1.mOPC.ScaleStab() > aP2.mOPC.ScaleStab();
}

cAFM_Im::cAFM_Im (const std::string  & aNameIm,cAppli_FitsMatch1Im & anAppli) :
   mAppli  (anAppli),
   mNameIm (aNameIm),
   mVOPC   (int(eTIR_NoLabel)),
   mVIndex (mVOPC.at(int(mAppli.FitsPm().KindOl())))
{
    std::string aNamePC = NameFileNewPCarac(mNameIm,true,anAppli.ExtNewH());

    mSetPC = StdGetFromNRPH(aNamePC,SetPCarac);
    // std::cout << "cAFM_Im::cAFM " << aNameIm << " " << mSetPC.OnePCarac().size() << "\n";

    for (const auto & aPC : mSetPC.OnePCarac())
    {
        mVOPC.at(int(aPC.Kind())).push_back(cCompileOPC(aPC));
    }

    
}

cAFM_Im::~cAFM_Im()
{
   static cSetPCarac TheSetPC;
   mSetPC = TheSetPC;
}

/*************************************************/
/*                                               */
/*           cAFM_Im_Master                      */
/*                                               */
/*************************************************/

cAFM_Im_Master::cAFM_Im_Master(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im     (aName,anApli),
    mVTabIndex  (1<< mAppli.NbBIndex())
{
    std::vector<int> aVFlagVois;
    SetOfFlagInfNbb(aVFlagVois,mAppli.NbBIndex(),mAppli.ThreshBIndex());

/*
    for (int aK=0 ; aK< 16 ; aK++)
    {
         std::cout << "K=" << aK << " " ;
         for (int aB=3 ; aB>=0 ; aB--)
             std::cout << ((aK&(1<<aB)) ? "1" : "0") ;
         std::cout << " NBB=" << NbBitOfShortFlag(aK) << "\n";
    }
    for (const auto & aFlagV : aVFlagVois)
    {
        std::cout << "FllaggVv " << NbBitOfShortFlag(aFlagV) << "\n";
    }
    std::cout << "VOISSSSSS " << aVFlagVois.size() << "\n"; getchar();
*/

    for (auto & aPC : mVIndex)
    {
        aPC.SetFlag(mAppli.FitsPm().OverLap());
        int aFlag = aPC.mShortFlag;
        for (const auto & aFlagV : aVFlagVois)
        {
             mVTabIndex.at(aFlag^aFlagV).push_back(&aPC);
        }
    }
     // std::vector<std::vector<cCompileOPC *> > mVIndex;

     // std::cout << getchar();
}

class cCdtCplHom
{
    public :
       cCdtCplHom(cCompileOPC * aPM,cCompileOPC * aPS,double aCorr,int aShift) :
           mPM    (aPM),
           mPS    (aPS),
           mCorr  (aCorr),
           mShift (aShift)
       {
       }

       cCompileOPC * mPM;
       cCompileOPC * mPS;
       double        mCorr;
       int           mShift;
};

void cAFM_Im_Master::Match(cAFM_Im_Sec & anISec)
{
   int aNbMin = 4;

   const cFitsParam &  aFPM = mAppli.FitsPm();
   eTypePtRemark aT0 = aFPM.KindOl();
   // std::vector<cCompileOPC> & aVM = mVOPC.at(int(aT0));
   std::vector<cCompileOPC> & aVS = anISec.mVOPC.at(int(aT0));

   int aNB1=0;
   // int aNBSup=0;
   ElPackHomologue aPack;

   std::vector<cCdtCplHom> aVCpl;
   for (int aKs=0 ; aKs<(int)aVS.size() ; aKs++)
   {
      cCompileOPC & aPCS = aVS[aKs];
      aPCS.SetFlag(mAppli.FitsPm().OverLap());
      int aFlagS = aPCS.mShortFlag;
      std::vector<cCompileOPC *> & aVSel = mVTabIndex.at(aFlagS);
      
      for (int aKSel=0 ; aKSel<(int)aVSel.size() ; aKSel++)
      {
          int aShift;
          cCompileOPC * aPCM = aVSel[aKSel];
          double aD =  aPCM->Match(aPCS,aFPM,aShift);
          if (aD > 0)
          {
             aPCS.mTmpNbHom++;
             aPCM->mTmpNbHom++;

             aVCpl.push_back(cCdtCplHom(aPCM,&aPCS,aD,aShift));
          }
      }
   }

   {
       std::vector<cCdtCplHom> aNewV;
       for (const auto  & aCpl : aVCpl)
       {
          if ((aCpl.mPM->mTmpNbHom==1) && (aCpl.mPS->mTmpNbHom==1))
             aNewV.push_back(aCpl);
       }
       for (const auto  & aCpl : aVCpl)
       {
          aCpl.mPM->mTmpNbHom=0; 
          aCpl.mPS->mTmpNbHom=0; 
       }
       aVCpl = aNewV;
   }

   // Filtrage directionnel
   {
      double aPropConv = 0.1;
      double  aPropDir = 0.05;

      int aNbDir = 64;
      int   aMul = 100;
       

      int aSeuilDir = ElMax(1,round_ni(aPropDir*aNbDir));
      // Poids du filtre de convolution
      int aNbConv = round_up(aNbDir*aPropConv);


      std::vector<int> aHConv;
      for (int aKC=0 ; aKC<= aNbConv ; aKC++)
      {
           double aVal = aNbDir*aPropConv-aKC;
           if (aVal>0) 
              aHConv.push_back(ElMax(0,round_up(aMul*aVal)));
      }
      aNbConv = aHConv.size() -1;


      // Histo non filtre
      std::vector<int> aHDir(aNbDir,0.0);
      for (const auto  & aCpl : aVCpl)
      {
          ELISE_ASSERT(aCpl.mShift<aNbDir,"Dir over 64");
          aHDir[aCpl.mShift]++;
      }
      // convolution
      std::vector<int> aHConvDir(aNbDir,0);
      for (int aKD=0 ; aKD<aNbDir ; aKD++)
      {
         int aVDir = aHDir[aKD];
         if (aVDir)
         {
             for (int aKC=-aNbConv ; aKC<aNbConv ; aKC++)
             {
                  aHConvDir.at(mod(aKD+aKC,aNbDir)) += aHConv.at(ElAbs(aKC)) * aVDir;
             }
         }
      }

      // calcul du pt max
      int aHMax=-1;
      int aKMax = -1;
      for (int aKD=0 ; aKD<aNbDir ; aKD++)
      {
          if (aHConvDir[aKD]>aHMax)
          {
             aKMax = aKD;
             aHMax = aHConvDir[aKD];
          }
      }

      std::vector<cCdtCplHom> aNewV;
      for (const auto  & aCpl : aVCpl)
      {
          int aDif = mod(aCpl.mShift-aKMax,aNbDir);
          aDif = ElMin(aDif,aNbDir-aDif);
          if (aDif < aSeuilDir)
             aNewV.push_back(aCpl);

      }
      aVCpl = aNewV;
   }

   if (int(aVCpl.size()) <=  aNbMin)
      return;


   for (const auto  & aCpl : aVCpl)
   {
       aNB1++;
       aPack.Cple_Add(ElCplePtsHomologues(aCpl.mPM->mOPC.Pt(),aCpl.mPS->mOPC.Pt()));
   }
  
   

   std::cout << anISec.mNameIm << " " << aNB1 << " Prop1 " << aNB1/double(aVS.size())  << "\n";

   aPack.StdPutInFile(mAppli.NameCple(mNameIm,anISec.mNameIm));

}

/*************************************************/
/*                                               */
/*           cAFM_Im_Sec                         */
/*                                               */
/*************************************************/

cAFM_Im_Sec::cAFM_Im_Sec(const std::string  & aName,cAppli_FitsMatch1Im & anApli) :
    cAFM_Im(aName,anApli)
{
}

/*************************************************/
/*                                               */
/*           cAppli_FitsMatch1Im                 */
/*                                               */
/*************************************************/

   // return  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH +"@" + mPostHom,aN1,aN2,true);

cAppli_FitsMatch1Im::cAppli_FitsMatch1Im(int argc,char ** argv) :
   mImMast      (nullptr),
   mCurImSec    (nullptr),
   mNameXmlFits (DefNameFitParam),
   mExtNewH     (""),
   mSH          (""),
   mPostHom     ("dat")
{
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mNameMaster, "First Image")
                     << EAMC(mPatIm, "Name Image2"),
         LArgMain()  << EAM(mNameXmlFits,"XmlFits",true,"Name of xml file for Fits parameters")
                     <<  EAM(mExtNewH,"ExtPC",true,"Extension for P cararc to NewPH... ")
   );

   if (!EAMIsInit(&mSH) )
   {
        mSH = "HFits"+ mExtNewH;
   }
   
   InitFitsPm(mFitsPm,MMDir()+TheDirXmlFits,mNameXmlFits);
    // const cFitsParam & aFitsPM = anAppli.FitsPm();
   const cFitsOneLabel & aFOL = mFitsPm.OverLap();
   mNbBIndex =  aFOL.BinIndexed().CCB().Val().CompCBOneBit().size();
   mThreshBIndex =  aFOL.BinIndexed().CCB().Val().BitThresh();

   mEASF.Init(mPatIm);


   mImMast = new cAFM_Im_Master(mNameMaster,*this);

   for (const auto &  aName : *(mEASF.SetIm()))
   {
       if (aName != mNameMaster)
       {
           mCurImSec = new  cAFM_Im_Sec(aName,*this);

           mImMast->Match(*mCurImSec);

           delete mCurImSec;
           mCurImSec = nullptr;
       }
   }
}

int cAppli_FitsMatch1Im::NbBIndex() const
{
   return mNbBIndex;
}

int cAppli_FitsMatch1Im::ThreshBIndex() const
{
   return mThreshBIndex;
}

std::string cAppli_FitsMatch1Im::NameCple(const std::string & aN1,const std::string & aN2) const
{
   return  mEASF.mICNM->Assoc1To2("NKS-Assoc-CplIm2Hom@"+ mSH +"@" + mPostHom,aN1,aN2,true);
}

int CPP_FitsMatch1Im(int argc,char ** argv)
{
   cAppli_FitsMatch1Im anAppli(argc,argv);
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
aooter-MicMac-eLiSe-25/06/2007*/
