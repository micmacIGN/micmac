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

extern const std::string PASTIS_MATCH_ARGUMENT_NAME = "Match";
extern const std::string PASTIS_DETECT_ARGUMENT_NAME = "Detect";

extern const std::string PASTIS_IGNORE_MAX_NAME = "NoMax";
extern const std::string PASTIS_IGNORE_MIN_NAME = "NoMin";
extern const std::string PASTIS_IGNORE_UNKNOWN_NAME = "NoUnknown";

#if ELISE_unix
    const std::string TheStrSiftPP = "siftpp_tgi.LINUX";
    const std::string TheStrAnnPP  = "ann_mec_filtre.LINUX";
#elif ELISE_MacOs
    const std::string TheStrSiftPP = "siftpp_tgi.OSX";
    const std::string TheStrAnnPP  = "ann_samplekey200filtre.OSX";
#elif ELISE_windows
    const std::string TheStrSiftPP = "siftpp_tgi.exe";
    const std::string TheStrAnnPP  = "ann_samplekeyfiltre.exe";
#endif


/*********************************************/
/*                                           */
/*                ::                         */
/*                                           */
/*********************************************/


/*

Voici pour le seuil utilisé sur les prises de vue aériennes :
    ./siftpp image.tif --output image.key --threshold=0.015
Cela permet de bien réduire le nombre de points extraits.

Autres options possibles :
 --octaves=O           Number of octaves
 --levels=S            Number of levels per octave
 --first-octave=MINO   Index of the first octave
 --threshold=THR       Keypoint strength threshold : les points pour
lesquels la fonction différence de gaussienne est inférieure �  ce seuil
sont éliminés
 --edge-threshold=THR  On-edge threshold : c'est pour l'élimination des
candidats situés sur des arêtes

J'avais testé des solutions du type de celle que tu proposais au tout
début des tests (alors que l'appariement "brut" prenait près de 5h par
couple d'images aériennes). Il faudrait que je rejette un oeil l�
dessus. Ca pourrait notamment être utile pour de grandes images avec de
très grandes quantités de points (cf images satellites voire images
Marseille 10 cm...).

Arnaud

*/

void Banniere_Pastis()
{
   std::cout << "\n";
   std::cout <<  " *********************************\n";
   std::cout <<  " *     P-rogramme utilisant,     *\n";
   std::cout <<  " *     A-utopano                 *\n";
   std::cout <<  " *     S-ift pour les            *\n";
   std::cout <<  " *     T-ie-points dans les      *\n";
   std::cout <<  " *     I-mage                    *\n";
   std::cout <<  " *     S                         *\n";
   std::cout <<  " *********************************\n";

}


#define NOSFH -1234


/*********************************************/
/*                                           */
/*            cAppliPastis                    */
/*                                           */
/*********************************************/

/*
typedef enum
{
     eModeLeBrisPP,
     eModeAutopano
}  eModeBinSift;
*/

class cCple;

class cAppliPastis : public cAppliBatch
{
    public :
       cAppliPastis(int argc,char ** argv,bool FBD);
       //void DoAll();
       void Exec();

       void ExecSz(double aSz,bool aSSR);


       void Banniere()
       {
            if (! IsRelancedByThis() ) Banniere_Pastis();
       }
    private :

      bool ValideGlobH(const std::list<cCple> & aLC);
      std::list<cCple>  FiltrageHomogr(const std::list<cCple> & aL,double aSeuil);
      std::list<cCple>  FiltrageRot(std::list<cCple>  aL,double aSeuil);

       CamStenope * CamOfIm(const std::string & aNameIm);
       // CamStenope * Std_Cal_From_File

       void GenerateKey(const std::string &,const std::string & aNameIm);
       void GenerateMatch(const std::string & aNI1,const std::string & aNI2);
       void GenerateXML(std::pair<cCompileCAPI,cCompileCAPI> &);

       std::string NameKey(const std::string &);

       // std::string mN1;
       // std::string mN2;
       std::string    mBinDir;
       std::string    mBinDirAux;
       double         mSzPastis;

       int mNbMaxMatch;

       std::string   mNameAPM; // Auto Pano Match
       std::string   mNameHomXML; // Homologue XML
       std::string   mKeyGeom1;
       std::string   mKeyGeom2;

       eModeBinSift   mModeBin;
       std::string    mSiftImplem;
       std::string    mKCal;
       CamStenope *   mCamera1;
       CamStenope *   mCamera2;
       CamStenope *   Cam1(bool Create=true)
       {
            if ((mCamera1==0)   &&Create)
            {
                mCamera1= CamOfIm(CurF1());
            }
            return mCamera1;
       }
       CamStenope *   Cam2(bool Create=true)
       {
            if ((mCamera2==0)  && Create)
            {
                mCamera2= CamOfIm(CurF2());
            }
            return mCamera2;
       }


       cElHomographie mCurHom;
       Pt2di          mSzIm1;
       Pt2di          mSzIm2;

       bool OKP1(const Pt2dr & aP)
       {
             return (aP.x>0)&&(aP.y>0)&&(aP.x<mSzIm1.x)&&(aP.y<mSzIm1.y);
       }
       bool OKP2(const Pt2dr & aP)
       {
             return (aP.x>0)&&(aP.y>0)&&(aP.x<mSzIm2.x)&&(aP.y<mSzIm2.y);
       }

       double        mSeuilFHom;
       double        mSeuilDistEpip;
       double        mSeuilPente;
       double        mSeuilDup;
       int           mNbMinPtsExp;
       bool          mForceByDico;
       int           mNbMinPtsRot;
       int           mExpBin;
       int           mExpTxt;
       int           mOnlyXML;
       int           mFiltreOnlyDupl;
       int           mFiltreOnlyHom;
       int           mSsRes;
       std::string   mExt;
       int           mNbMinValidGlobH;
       int           mNbMaxValidGlobH;
       double        mSeuilHGLOB;
       std::string   mNKS;
       string        mDetectingTool;      // name of the program to be used for dectecting points
       string        mDetectingArguments; // arguments to be passed when calling the detecting tool
       string        mMatchingTool;       // name of the program to be used for matching points
       string        mMatchingArguments;  // arguments to be passed when calling the matching tool
       string        mOutputDirectory;
       bool          mIgnoreMin;
       bool          mIgnoreMax;
       bool          mIgnoreUnknown;
       double        mAnnClosenessRatio;
       bool          mIsSFS;              //force using SFS image instead of original even if size=-1

       Pt2dr Homogr1to2(const Pt2dr & aP1)
       {
          if ((!Cam1()) ||  (! Cam2()))
             return mCurHom.Direct(aP1);

           return  Cam2()->PtDirRayonL3toF2(mCurHom.Direct(Cam1()->F2toPtDirRayonL3(aP1)));
       }
};



std::string cAppliPastis::NameKey(const std::string & aFullName)
{
   std::string aDir,aName;
   SplitDirAndFile(aDir,aName,aFullName);
   return mOutputDirectory+ICNM()->Assoc1To2(mSiftImplem+"-Pastis-PtInt",aName,ToString(mSzPastis),true);
}

void cAppliPastis::GenerateKey(const std::string & aName,const std::string & aNameIm)
{
    if (mOnlyXML)
        return;

    std::string aNK = NameKey(aNameIm);

    std::string aCom ;

    if (mModeBin==eModeLeBrisPP)
    {
        std::string OptOut = " -o ";

        aCom =	protectFilename(g_externalToolHandler.get( mDetectingTool ).callName()) + ' ' + mDetectingArguments + ' ' +
                protectFilename(NameFileStd(aNameIm,1,false)) +
                OptOut +
                protectFilename(aNK);

    }
    else if (mModeBin==eModeAutopano)
    {
        aCom =	std::string("generatekeys ") +
                aNameIm + " " +
                aNK + " " +
                ((mSzPastis >=0) ? ToString(mSzPastis) : std::string(""));
    }

    System(aNK.c_str(),aCom);
}

void cAppliPastis::GenerateMatch(const std::string & aNI1,const std::string & aNI2)
{
  if (mOnlyXML)
     return;

  std::string aCom;

  string protected_named_key1 = protectFilename(NameKey(aNI1)),
         protected_named_key2 = protect_spaces(NameKey(aNI2));

  if (mModeBin==eModeLeBrisPP)
  {
    aCom =	protect_spaces(g_externalToolHandler.get( mMatchingTool ).callName()) + " " + mMatchingArguments + ' ' +
            protected_named_key1 + std::string(" ") +
            protected_named_key2 + std::string(" ") +
            mNameAPM +
            std::string(" -ratio ") + ToString(mAnnClosenessRatio);
  }
  else if (mModeBin==eModeAutopano)
  {
      aCom = std::string("autopano --ransac off ")
             + std::string("--maxmatches ") + ToString(mNbMaxMatch) +  std::string(" ")
             + mNameAPM +  std::string(" ")
             + protected_named_key1 + std::string(" ")
             + protected_named_key2 + std::string(" ");
  }

  System(mNameAPM.c_str(),aCom);
  if (ByMKf())
  {
     GPAO().TaskOfName(mNameAPM).AddDep(protected_named_key1);
     GPAO().TaskOfName(mNameAPM).AddDep(protected_named_key2);
  }

}


CamStenope * cAppliPastis::CamOfIm(const std::string & aNameIm)
{
   std::string aNameCal;

   if ( ELISE_fp::exist_file( mOutputDirectory+mKCal ) )
      aNameCal = mKCal;
   else
      aNameCal = ICNM()->Assoc1To1(mKCal,aNameIm,true);

   if (aNameCal != "NoCalib") return Std_Cal_From_File( mOutputDirectory+aNameCal );

   return 0;
}

class cCple
{
     public :
         cCple           (const Pt2dr&  aP1, CamStenope * aCam1,const Pt2dr&  aP2,CamStenope * aCam2,bool CorDist) :
            mP1          (aP1),
            mP2          (aP2),
        mQ1          (CorDist ? (aCam1->F2toPtDirRayonL3(mP1)) : aP1),
        mQ2          (CorDist ? (aCam2->F2toPtDirRayonL3(mP2)) : aP2),
            mZ           (0),
            mSPente      (0),
            mOnePenteOut (false),
            mOK          (true)
         {
     }

         Pt2dr mP1;  // Points reels
         Pt2dr mP2;
         Pt2dr mQ1;  // Points photogram
         Pt2dr mQ2;
         double mZ;
         double mSPente;
         bool   mOnePenteOut;
         bool  mOK;

         double Pente(const cCple & aCple) const
         {
             return ElAbs(mZ-aCple.mZ) /euclid(mP1,aCple.mP1);
         }
};


#include "algo_geom/rvois.h"

class cPtOfCple
{
    public :
       cPtOfCple(bool isP1) : mIsP1(isP1) {}
       Pt2dr operator ()(cCple * aCple) {return mIsP1 ? aCple->mP1 : aCple->mP2;}
       bool mIsP1;
};




class cActSetCpleNotOK
{
     public :
       void  operator ()(cCple * aCpl1,cCple * aCpl2)
       {
          aCpl1->mOK=false;
          aCpl2->mOK=false;
       }
};


void FiltrageDup
     (
         std::vector<cCple *> & aVPC,
         double aDist,
         bool   isP1
     )
{
   if ( ( aDist<0 ) || ( aVPC.size()==0 ) ) return;
   cPtOfCple aPOC(isP1);
   cActSetCpleNotOK anAct;
   cCple ** aV0 = &(aVPC[0]);
   rvoisins_sortx(aV0,aV0+aVPC.size(),aDist,aPOC,anAct);
}

std::list<cCple> FiltrageDup
                 (
                    std::list<cCple> & aLC,
                    double aDist
                 )
{
   std::vector<cCple *> aVPC;
   for (std::list<cCple>::iterator itC=aLC.begin(); itC!=aLC.end();itC++)
   {
      aVPC.push_back(&(*itC));
   }
   FiltrageDup(aVPC,aDist,true);
   FiltrageDup(aVPC,aDist,false);

   std::list<cCple> aRes;
   for (int aK=0 ; aK<(int)aVPC.size();aK++)
   {
      if (aVPC[aK]->mOK)
         aRes.push_back(*aVPC[aK]);
   }


   return aRes;
}





ElPackHomologue ToLPt(const std::list<cCple> &aLC,bool isQ,int aNbIn)
{

   int aNbTot = (int)aLC.size();
   if ((aNbIn <0)  || (aNbIn>aNbTot))
       aNbIn = aNbTot;


   ElPackHomologue  aRes;
   for (std::list<cCple>::const_iterator itC=aLC.begin();itC!=aLC.end();itC++)
   {
      double aProbaIn = double(aNbIn) / double(aNbTot);
      if (NRrandom3() < aProbaIn)
      {
         aRes.Cple_Add
         (
             ElCplePtsHomologues
         (
               isQ ? itC->mQ1 : itC->mP1,
               isQ ? itC->mQ2 : itC->mP2
         )
         );
         aNbIn--;
      }
      aNbTot--;
   }

   return aRes;
}


std::list<cCple>  cAppliPastis::FiltrageHomogr(const std::list<cCple> & aLC,double aSeuil)
{
   if (mFiltreOnlyDupl)
      return aLC;

   ElPackHomologue aPackPhgr = ToLPt(aLC,true,1000);
   mCurHom = cElHomographie(aPackPhgr,false);
   std::list<cCple> aRes;
   int aNbHS = 0;

   for (std::list<cCple>::const_iterator itC=aLC.begin();itC!=aLC.end();itC++)
   {
       double aDist = euclid(Homogr1to2(itC->mP1),itC->mP2);
       if (aDist > aSeuil)
       {
          // std::cout << euclid(Homogr1to2(itC->mP1),itC->mP2) << "\n";
          aNbHS++;
       }
       else
       {
          aRes.push_back(*itC);
       }
   }

   std::cout << "Nb Hors Seuil Homographie = " << aNbHS << "\n";

   return aRes;
}

/*
 * Nb Hors Seuil Homographie = 11
 * SZ = 14985
 * ERR = 0.273899
 * DOUBLON = 1767
 * SZ = 14985
 * FFFFF 6228.54 6228.54
 *
*/


class cCmpPCpleOnSP
{
    public :
      bool operator () (cCple * aC1,cCple * aC2) const
      {
           return aC1->mSPente < aC2->mSPente;
      }
};


bool cAppliPastis::ValideGlobH(const std::list<cCple> & aLC)
{
   if (mSeuilHGLOB <0) return true;

   int aNB = (int)aLC.size();
   if (aNB <= mNbMinValidGlobH)
      return false;
   if (aNB > mNbMaxValidGlobH)
      return true;

   ElPackHomologue aPackPhgr = ToLPt(aLC,false,1000);
   cElHomographie aH12 = cElHomographie(aPackPhgr,false);
   cElHomographie aH21 = aH12.Inverse();

   double aEcart = 0;
   for
   (
          ElPackHomologue::const_iterator itP=aPackPhgr.begin();
          itP!=aPackPhgr.end();
          itP++
   )
   {
         aEcart +=   euclid(itP->P2(),aH12.Direct(itP->P1()))
                   + euclid(itP->P1(),aH21.Direct(itP->P2()));
   }
   StatElPackH aStat(aPackPhgr);
   aEcart /= aStat.SomD1() + aStat.SomD2();
   aEcart *= aNB/ (aNB-4.0);

   std::cout << "ECART HGLOB " << aEcart << "\n";
   return aEcart<mSeuilHGLOB;
}

std::list<cCple>  cAppliPastis::FiltrageRot(std::list<cCple>  aLC,double aSeuil)
{
   if (mFiltreOnlyDupl || mFiltreOnlyHom)
      return aLC;
   if (int(aLC.size()) < mNbMinPtsRot)
      return aLC;

   ElPackHomologue aPackPhgr = ToLPt(aLC,true,1000);
   double aD;
   ElRotation3D aR = aPackPhgr.MepRelGen(1.0,false,aD);

   Cam2()->SetOrientation(aR);

   // On filtre selon la pax transverse et on rajoute
   // dans le qdt, on calcule le Z
   Box2dr aBox(Pt2dr(-5,-5),Pt2dr(mSzIm1)+Pt2dr(5,5));
   cPtOfCple aGetP1(true);
   ElQT<cCple * ,Pt2dr,cPtOfCple> aQT(aGetP1,aBox,20,20);
   std::list<cCple> aNewLC;
   std::vector<cCple* > aVC;
   for
   (
       std::list<cCple>::iterator itC=aLC.begin();
       itC != aLC.end();
       itC++
   )
   {
        double aDist;
        Pt3dr aPTer = Cam1()->PseudoInter(itC->mP1,*Cam2(),itC->mP2,&aDist);

        Pt3dr aDirR1 = vunit(Cam1()->F2toDirRayonR3(itC->mP1));
        double aProf1 = scal(aDirR1,aPTer-Cam1()->VraiOpticalCenter());
        Pt3dr aPTer1A = Cam1()->VraiOpticalCenter() + aDirR1 * (aProf1*0.99);
        Pt3dr aPTer1B = Cam1()->VraiOpticalCenter() + aDirR1 * (aProf1*1.01);


        Pt2dr aQa = Cam2()->R3toF2(aPTer1A);
        Pt2dr aQb = Cam2()->R3toF2(aPTer1B);
        Pt2dr aDirEpi2 = vunit(aQb-aQa);

        Pt2dr aPr1 = Cam1()->R3toF2(aPTer);
        Pt2dr aPr2 = Cam2()->R3toF2(aPTer);
        Pt2dr aPH2 = Homogr1to2(itC->mP1);
        double aZ = scal(aDirEpi2,itC->mP2-aPH2);
        aDist =  euclid(itC->mP1,aPr1) + euclid(itC->mP2,aPr2);

        if (aDist<mSeuilDistEpip)
        {
            itC->mZ = aZ;
            aNewLC.push_back(*itC);
            aQT.insert(&aNewLC.back());
            aVC.push_back(&aNewLC.back());
        }
   }

   std::cout << "Apres Epip  " << aNewLC.size() << "\n";

   int aNbPPV = 10;
   double aDistPPV = sqrt((aNbPPV/2.0)*mSzIm1.x*mSzIm1.y/(PI*aNewLC.size()));
   int aKT = (int)aNewLC.size();
   double aMaxPente = 0;
   for
   (
       std::list<cCple>::iterator itC=aNewLC.begin();
       itC != aNewLC.end();
       itC++
   )
   {

      std::list<cCple * > aLTes = aQT.KPPVois(itC->mP1,aNbPPV,aDistPPV);
      if ((! aLTes.empty())  && (&(*itC)==aLTes.front()))
         aLTes.pop_front();

      double aSPente = 0;
      for (std::list<cCple * >::iterator itV=aLTes.begin();itV!=aLTes.end();itV++)
      {
           // double aPente = (aZ1-aZ2)/euclid( itC->mP1,(*itV)->mP1);
           double aPente = itC->Pente(**itV);
           aSPente  += aPente;
           if (aPente>mSeuilPente)
              itC->mOnePenteOut = true;
           ElSetMax(aMaxPente,aPente);
      }
      aSPente /= ElMax(1,int(aLTes.size()));
      itC->mSPente = aSPente;

      aKT--;
   }

   cCmpPCpleOnSP aCmp;
   std::sort(aVC.begin(),aVC.end(),aCmp);

   aLC.clear();
   for ( int aK = (int)(aVC.size() - 1); aK>=0; aK--)
   {
      if (aVC[aK]->mOnePenteOut)
      {
         std::list<cCple * > aLTes = aQT.KPPVois(aVC[aK]->mP1,aNbPPV,aDistPPV);
         if ((! aLTes.empty())  && (aVC[aK]==aLTes.front()))
            aLTes.pop_front();
         for (std::list<cCple * >::iterator itV=aLTes.begin();itV!=aLTes.end();itV++)
         {
           if ((*itV)->mOK)
           {
              double aPente = aVC[aK]->Pente(**itV);
              if (aPente>mSeuilPente)
                 aVC[aK]->mOK = false;
           }
         }
      }
      if (aVC[aK]->mOK)
      {
         aLC.push_back(*aVC[aK]);
      }
      else
      {
         // std::cout << aVC[aK]->mSPente << "\n";
         // getchar();
      }
   }

   // std::cout << "Pente Max " << aMaxPente << "\n";

   return aLC;
}

void cAppliPastis::GenerateXML(std::pair<cCompileCAPI,cCompileCAPI> & aPair)
{


   if (ByMKf() && (!mOnlyXML))
   {
      std::string aCom = ComForRelance() + " OnlyXML=1";
      System(mNameHomXML.c_str(),aCom);
      GPAO().TaskOfName(mNameHomXML).AddDep(mNameAPM);
      GPAO().TaskOfName("all").AddDep(mNameHomXML);

      return;
   }

   if (FileStrictPlusRecent(mNameHomXML,mNameAPM))
   {
      std::cout << "Rien a faire pour " << mNameHomXML << " " << mNameAPM << endl;
      return;
   }

   ELISE_fp aFTxt(mNameAPM.c_str(),ELISE_fp::READ);
   string aBuf; // char aBuf[200]; TEST_OVERFLOW

   cTplValGesInit<std::string> aNoStr;

    mCamera1=0;
    mCamera2=0;
    // Cam1();
    // Cam2();
   // mCam1 = CamOfIm(CurF1());
   // mCam2 = CamOfIm(CurF2());

   // if

   bool End= false;

   int NbIn = 0;
   int NbOut = 0;
   std::list<cCple> aLCple;
   while (! End)
   {
       if (aFTxt.fgets(aBuf,End)) // if (aFTxt.fgets(aBuf,200,End)) TEST_OVERFLOW
       {
            bool DoIt = true;
        if ( mModeBin==eModeAutopano)
        {
            DoIt = (aBuf[0]=='c');
        }
            if (DoIt)
        {
               char A[20], B[20], C[20], D[20];
               char x[20], y[20], X[20], Y[20];
           int aOfset=0;

           switch (mModeBin)
           {
                   case eModeAutopano :
                        sscanf(aBuf.c_str(),"%s %s %s %s %s %s %s %s",A,B,C,x,y,X,Y,D); //sscanf(aBuf,"%s %s %s %s %s %s %s %s",A,B,C,x,y,X,Y,D); TEST_OVERFLOW
            aOfset=1;
                   break;

                   case eModeLeBrisPP :
                        sscanf(aBuf.c_str(),"%s %s %s %s",x,y,X,Y); // sscanf(aBuf,"%s %s %s %s",x,y,X,Y); TEST_OVERFLOW
                   break;
               }

               Pt2dr aP1(atof(x+aOfset),atof(y+aOfset));
               Pt2dr aP2(atof(X+aOfset),atof(Y+aOfset));

           aP1 = aPair.first.Rectif2Init(aP1);
           aP2 = aPair.second.Rectif2Init(aP2);

               if ((OKP1(aP1)) && (OKP2(aP2)))
               {
 // std::cout << mCam1->IsInZoneUtile(aP1) << " " << mCam2->IsInZoneUtile(aP2) << "\n";
                  NbIn++;
                  bool CorDist = (!mFiltreOnlyDupl);
              aLCple.push_back(cCple(aP1,Cam1(CorDist),aP2,Cam2(CorDist),CorDist));
               }
               else
               {
                  NbOut++;
               }
        }
        }
   }

   std::cout << "Cple Init = " << aLCple.size() << "IN " << NbIn << " OUT " << NbOut<< "\n";

   aLCple = FiltrageDup(aLCple,mSeuilDup);


   std::cout << "Apres Rm Dup,  " << aLCple.size() << "\n";

   if (!mSsRes)
   {
      bool OKGlob = ValideGlobH(aLCple);
      if (! OKGlob)
      {
          std::string aD,aN;
          SplitDirAndFile(aD,aN,mNameHomXML);
          mNameHomXML = aD + "Failed_" + aN;
      }
      std::cout << "OK GLOB " <<  OKGlob << " " << mNameHomXML << "\n";
   }

   double aSFH = mSeuilFHom;
   if (aSFH==NOSFH)
   {
       Tiff_Im aI1 = Tiff_Im::StdConvGen(mOutputDirectory+CurF1(),1,false);
       aSFH =  0.1 * euclid(aI1.sz());
   }
   if (aSFH > 0)
   {
      aLCple = FiltrageHomogr(aLCple,aSFH);
   }
   std::cout << "Apres Hom  " << aLCple.size() << "\n";

   aLCple = FiltrageRot(aLCple,1.0);

   std::cout << "Apres Rot  " << aLCple.size() << "\n";

   if (int(aLCple.size())>=mNbMinPtsExp)
   {
         ToLPt(aLCple,false,-1).StdPutInFile(mNameHomXML);
/*
      if (mExTxt)
      {
         ToLPt(aLCple,false,-1).StdPutInFile(mNameHomXML);
      }
      else if (mExpBin)
      {
         ELISE_fp aFP(mNameHomXML.c_str(),ELISE_fp::WRITE);
         ToLPt(aLCple,false,-1).write(aFP);
      }
      else
      {
         cElXMLFileIn aFileXML(mNameHomXML);
         aFileXML.PutPackHom(ToLPt(aLCple,false,-1));
      }
*/
   }


   // std::cout << "FFFFF " <<  mCam1->Focale() << " " << mCam2->Focale() << "\n";

}


void AdaptSzXif(Pt2di & aSz,const std::string & aFileName)
{
    cMetaDataPhoto aMtD  = cMetaDataPhoto::CreateExiv2(aFileName);

    const Pt2di aSzXif = aMtD.XifSzIm(true);
    if  (aSzXif.x >0)
    {
       aSz = Inf(aSz,aSzXif);
    }
}


void cAppliPastis::Exec()
{
    string filename1 = mOutputDirectory+CurF1();
    string filename2 = mOutputDirectory+CurF2();
    if ( isUsingSeparateDirectories() )
    {
        if ( !ELISE_fp::exist_file( filename1 ) ) filename1 = DirChantier()+CurF1();
        if ( !ELISE_fp::exist_file( filename2 ) ) filename2 = DirChantier()+CurF2();
    }

    mSzIm1 = Tiff_Im::StdConvGen(filename1,1,false).sz();
    mSzIm2 = Tiff_Im::StdConvGen(filename2,1,false).sz();
    AdaptSzXif(mSzIm1,filename1);
    AdaptSzXif(mSzIm2,filename2);
/*
if (MPD_MM())
{
    // cMetaDataPhoto aMtD1  = cMetaDataPhoto::CreateExiv2(const std::string &);
    std::cout << "zzzHHhhhhhhhhhhhhhhhhhhhhh " << mSzIm1   << " " << CurF1() << "  " << mSzIm2 << "\n";
    getchar();
}
*/

    ExecSz(mSzPastis,false);
}

void cAppliPastis::ExecSz(double aSzMaxApp,bool)
{
  if  (mModeBin== eModeAutopano)
      aSzMaxApp= -1;

  std::pair<cCompileCAPI,cCompileCAPI> aPair= ICNM()->APrioriAppar
                                               (CurF1(),CurF2(),mKeyGeom1,mKeyGeom2,aSzMaxApp,mIsSFS);

  if (mModeBin==eModeLeBrisPP)
  {
      std::string aDir,aN1,aN2;
      SplitDirAndFile(aDir,aN1,NameKey(aPair.first.NameRectif()));
      SplitDirAndFile(aDir,aN2,NameKey(aPair.second.NameRectif()));

      ELISE_fp::MkDirSvp( mOutputDirectory+"Pastis"+ELISE_CAR_DIR+"LBPp-Match-"+StdPrefix(aN1)+ELISE_CAR_DIR );

      mNameAPM = mOutputDirectory + ICNM()->Assoc1To2(mSiftImplem+"-Pastis-Hom-Txt",aN1,aN2,true);
  }
  else if (mModeBin==eModeAutopano)
      mNameAPM = mOutputDirectory + ICNM()->Assoc1To3(mSiftImplem+"-Pastis-Hom-Txt",CurF1(),CurF2(),ToString(mSzPastis),true);

   if (mNKS!="")
   {
      mNameHomXML = mOutputDirectory + ICNM()->Assoc1To2(mNKS,CurF1(),CurF2(),true);
   }
   else
   {
       std::string aKAssoc =   mSsRes                                  ?
                               "Key-Assoc-SsRes-CpleIm2HomolPastisBin" :
                               "Key-Assoc-CpleIm2HomolPastisBin"       ;

      if (mExt!="") aKAssoc = "KeyStd-Assoc-CplIm2HomBin@" + mExt;

      mNameHomXML = mOutputDirectory + ICNM()->Assoc1To2(aKAssoc,CurF1(),CurF2(),true);

      if (mExpBin)
      {
         mNameHomXML = StdPrefix(mNameHomXML) + ".dat";
      }
      if (mExpTxt)
      {
         mNameHomXML = StdPrefix(mNameHomXML) + ".txt";
      }
   }



   if (ModeExe()==eExeDoNothing)
   {
       std::cout << CurF1() << " " << CurF2() << "\n";
   }
   else
   {
       GenerateKey(CurF1(),aPair.first.NameRectif());
       GenerateKey(CurF2(),aPair.second.NameRectif());

       GenerateMatch(aPair.first.NameRectif(),aPair.second.NameRectif());
       GenerateXML(aPair);
   }
}

// a tool string is composed of two string separated by a ':'
// -the first one contains the executable name
// -the second one contains its arguments
// if there's more than one ':', the first is used
// io_tool is the source string and will receive the executable name in case of success
// if no ':' is found, the whole string is the executable name and o_args is set to the null string
// o_args will receive the arguments in case of success
// the function returns if the source argument could be processed
bool process_pastis_tool_string( string &io_tool, string &o_args )
{
    if ( io_tool.length()==0 ) return false;

    size_t pos = io_tool.find( ':' );
    if ( pos==0 ) return false;
    if ( pos==io_tool.length()-1 ){
        io_tool.resize( io_tool.length()-1 );
        pos = string::npos;
    }
    if ( pos==string::npos )
        o_args.clear();
    else{
        o_args  = io_tool.substr( pos+1 );
        io_tool = io_tool.substr( 0, pos );
    }
    return true;
}

cAppliPastis::cAppliPastis(int argc,char ** argv,bool FBD) :
   cAppliBatch(argc,argv,4,2,"Pastis","",FBD),
   mBinDir           (MMBin()),
   mBinDirAux        (MMDir()+"binaire-aux"+ELISE_CAR_DIR),
   mNbMaxMatch       (100),
   mKeyGeom1         ("DefKey"),
   mKeyGeom2         ("DefKey"),
   mCurHom           (cElHomographie::Id()),
   mFiltreOnlyDupl   (1),
   mFiltreOnlyHom    (0),
   mNbMinValidGlobH  (4),
   mNbMaxValidGlobH  (200000),
   mSeuilHGLOB       (-1.0),
   mDetectingTool    ( TheStrSiftPP ),
   mMatchingTool     ( TheStrAnnPP ),
   mIgnoreMin        (false),
   mIgnoreMax        (false),
   mIgnoreUnknown    (false),
   mAnnClosenessRatio(SIFT_ANN_DEFAULT_CLOSENESS_RATIO),
   mIsSFS(false)
{
    mOutputDirectory = ( isUsingSeparateDirectories()?MMOutputDirectory():DirChantier() );
    std::string aKG12="";
    if (!NivPurgeIsInit())
       SetNivPurge(eNoPurge);

   if (!NivExeIsInit())
      SetNivExe(eExeDoIfFileDontExist);

   mSiftImplem = "eModeLeBrisPP";
   mKCal = "Key-Assoc-CalibOfIm";

   mSeuilFHom = NOSFH;  // Si pas defini : 10% de la diag
   mSeuilDistEpip=2;
   mSeuilPente = 0.7;
   mNbMinPtsExp = -1;
   mNbMinPtsRot = 20;
   mExpBin = 1;
   mExpTxt = 0;
   mSeuilDup = 1;
   mOnlyXML =0;
   mForceByDico=true;

   mSsRes = 0;

    ElInitArgMain
    (
           ARGC(),ARGV(),
           LArgMain() << EAM(mSzPastis) ,
           LArgMain() << EAM(mNbMaxMatch,"NbMM",true)
                      << EAM(aKG12,"KG12",true)
                      << EAM(mKeyGeom1,"KG1",true)
                      << EAM(mKeyGeom2,"KG2",true)
                      << EAM(mSiftImplem,"mSiftImplem",true)
                      << EAM(mKCal,"KCal",true)
                      << EAM(mSeuilFHom,"SFH",true)
                      << EAM(mSeuilDistEpip,"DistEpip",true)
                      << EAM(mSeuilPente,"SeuilPente",true)
                      << EAM(mExpBin,"ExportBinaire",true)
                      << EAM(mExpTxt,"ExpTxt",true)
                      << EAM(mSeuilDup,"SeuilDup",true)
                      << EAM(mNbMinPtsExp,"NbMinPtsExp",true)
                      << EAM(mForceByDico,"ForceByDico",true)
                      << EAM(mOnlyXML,"OnlyXML",true)
                      << EAM(mFiltreOnlyDupl,"FiltreOnlyDupl",true)
                      << EAM(mFiltreOnlyHom,"FiltreOnlyHom",true)
                      << EAM(mSsRes,"SsRes",true)
                      << EAM(mExt,"Ext",true)
                      << EAM(mNKS,"NKS",true)

                      << EAM(mDetectingTool,PASTIS_DETECT_ARGUMENT_NAME.c_str(),true)
                      << EAM(mMatchingTool,PASTIS_MATCH_ARGUMENT_NAME.c_str(),true)

                      << EAM(mIgnoreMax,PASTIS_IGNORE_MAX_NAME.c_str(),true)
                      << EAM(mIgnoreMin,PASTIS_IGNORE_MIN_NAME.c_str(),true)
                      << EAM(mIgnoreUnknown,PASTIS_IGNORE_UNKNOWN_NAME.c_str(),true)
                      << EAM(mAnnClosenessRatio,"ratio",true)
                      << EAM(mIsSFS,"isSFS",true)


    );

    if ( !process_pastis_tool_string( mDetectingTool, mDetectingArguments ) ){
        cerr << "Pastis: ERROR: specified string for the detecting tool is invalid (format is : tool[:arguments] )" << endl;
        ElEXIT( EXIT_FAILURE ,"Pastis:syntax error");
    }
    if ( !process_pastis_tool_string( mMatchingTool, mMatchingArguments ) ){
        cerr << "Pastis: ERROR: specified string for the matching tool is invalid (format is : tool[:arguments] )" << endl;
        ElEXIT( EXIT_FAILURE,"Pastis: match error" );
    }

    if ( mIgnoreMax ) mMatchingArguments += " -ignoreMax";
    if ( mIgnoreMin ) mMatchingArguments += " -ignoreMin";
    if ( mIgnoreUnknown ) mMatchingArguments += " -ignoreUnknown";

    if (mExpTxt) mExpBin = 0;

    if (aKG12!="")
    {
        mKeyGeom1 = aKG12;
        mKeyGeom2 = aKG12;
    }

    if (mFiltreOnlyDupl || mFiltreOnlyHom)
    {
        mSeuilDistEpip=1e15;
        mSeuilPente = 1e15;
        if (mFiltreOnlyDupl)
        {
           mSeuilFHom = 1e15;
          if (mNbMinPtsExp <0)
               mNbMinPtsExp = 0;
           mNbMinPtsRot = 0;
        }
    }
    if (mNbMinPtsExp <0)
       mNbMinPtsExp = 20;

    mModeBin = Str2eModeBinSift(mSiftImplem);
}






   //===========================================

int Pastis_main(int argc,char ** argv)
{
    std::vector<char *> aNewArgv;
    bool FBD=false;
    for (int aK=0 ; aK<argc ; aK++)
    {
         // if (std::string(argv[aK]==std::string("ForceByDico=1")))
         if (strcmp(argv[aK],"ForceByDico=1")==0)
         {
              FBD=true;
         }
         else
         {
             aNewArgv.push_back(argv[aK]);
         }
    }

    argv = & (aNewArgv[0]);
    argc = (int)aNewArgv.size();

    MMD_InitArgcArgv(argc,argv);

    cAppliPastis aAP(argc,argv,FBD);

    aAP.DoAll();
    aAP.Banniere();

    return 0;
}







/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit francais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les memes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,  a l'utilisation,  a la modification et/ou au
developpement et a la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe a
manipuler et qui le reserve donc a des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a charger  et  tester  l'adequation  du
logiciel a leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a l'utiliser et l'exploiter dans les memes conditions de securite.

Le fait que vous puissiez acceder a cet en-tete signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/

