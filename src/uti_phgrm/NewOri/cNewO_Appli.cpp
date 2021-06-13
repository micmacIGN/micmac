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

#include "NewOri.h"


/**********************************************************/
/*                                                        */
/*            cCommonMartiniAppli                         */
/*                                                        */
/**********************************************************/

LArgMain &     cCommonMartiniAppli::ArgCMA()
{
   return *mArg;
}

cCommonMartiniAppli::cCommonMartiniAppli() :
    mNameOriCalib      (""),
    mPrefHom           (""),
    mExtName           (""),
    mExpTxt            (false),
    mInOri             (""),
    mOriOut            (""),
    mOriGPS            (""),
    mOriCheck          (""),
    mDebug             (false),
    mAcceptUnSym       (true),
    mQuick             (true),
    mShow              (false),
    mTStdNbMaxTriplet   (20),
    mTQuickNbMaxTriplet(8),
    mTNbMinTriplet     (5),
    mArg               (new LArgMain),
    mPostInit          (false),
    mNM                (0),
    mNameNOMode        (TheStdModeNewOri)
{

      (*mArg) 
              << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
              << EAM(mPrefHom,"SH",true,"Prefix Homologue , Def=\"\"")  // SH par homogeneite avec autre commandes 
              << EAM(mExtName,"ExtName",true,"User's added Prefix , Def=\"\"")  // SH par homogeneite avec autre commandes 
              << EAM(mExpTxt,"ExpTxt",true,"Homol in text format? , Def=\"false\"")  
              << EAM(mNameNOMode,"ModeNO",true,"Mode Def=Std (TTK StdNoTTK OnlyHomogr)")  
              << EAM(mInOri,"InOri",true,"Existing orientation if any")  
              << EAM(mOriOut,"OriOut",true,"Output orientation dir")  
              << EAM(mOriGPS,"OriGPS",true,"Orientation where find gps data when exists")  
              << EAM(mOriCheck,"OriCheck",true,"Reference Orientation  to check results")
              << EAM(mDebug,"Debug",true,"Debug ....")  
              << EAM(mAcceptUnSym,"AUS",true,"Accept non symetric homologous point;")  
              << EAM(mQuick,"Quick",true,"If true (default) do less test")  
              << EAM(mTStdNbMaxTriplet,"StdNbPtTrip",true,"Max num of triplets per edge (Std mode)")  
              << EAM(mTQuickNbMaxTriplet,"QNbPtTrip",true,"Max num of triplets per edge (Quick mode), Def=8")  
              << EAM(mTNbMinTriplet,"NbTrip",true,"Min num of points to calculate a triplet")  
              << EAM(mShow,"Show",true,"If true (non default) print (a lot of) messages")  ;

}

void cCommonMartiniAppli::PostInit() const
{
   if (mPostInit) return;
   mPostInit = true;

   mModeNO = ToTypeNO(mNameNOMode);
}


cNewO_NameManager *  cCommonMartiniAppli::NM(const std::string & aDir) const
{
   if (mNM==0) 
      mNM =  new cNewO_NameManager(mExtName,mPrefHom,mQuick,aDir,mNameOriCalib,mExpTxt ? "txt" : "dat" ,mOriOut);
   return mNM;
}

eTypeModeNO  cCommonMartiniAppli::ModeNO() const
{
   PostInit();
   return mModeNO;
}

bool  cCommonMartiniAppli::GpsIsInit()
{
   return EAMIsInit(&mOriGPS);
}

bool  cCommonMartiniAppli::CheckIsInit()
{
   return EAMIsInit(&mOriCheck);
}

Pt3dr cCommonMartiniAppli::GpsVal(cNewO_OneIm * anIm)
{
   return anIm->NM().ICNM()->StdCamStenOfNames(anIm->Name(),mOriGPS)->VraiOpticalCenter();
}

CamStenope * cCommonMartiniAppli::CamCheck(cNewO_OneIm * anIm)
{
   return anIm->NM().ICNM()->StdCamStenOfNames(anIm->Name(),mOriCheck);
}


std::string    cCommonMartiniAppli::ComParam()
{
   std::string aCom;
   if (EAMIsInit(&mNameOriCalib))  aCom += aCom + " OriCalib=" + mNameOriCalib;
   if (EAMIsInit(&mPrefHom))       aCom += " SH="        + mPrefHom;
   if (EAMIsInit(&mExtName))       aCom += " ExtName="   + mExtName;
   if (EAMIsInit(&mNameNOMode))    aCom += " ModeNO="    + mNameNOMode;
   if (EAMIsInit(&mInOri))         aCom += " InOri="     + mInOri;
   if (EAMIsInit(&mOriOut))        aCom += " OriOut="    + mOriOut;
   if (EAMIsInit(&mAcceptUnSym))   aCom += " AUS="       + ToString(mAcceptUnSym);
   if (EAMIsInit(&mQuick))         aCom += " Quick="     + ToString(mQuick);
   if (EAMIsInit(&mShow))          aCom += " Show="      + ToString(mShow);
   if (EAMIsInit(&mOriGPS))        aCom += " OriGPS="    + mOriGPS;
   if (EAMIsInit(&mOriCheck))      aCom += " OriCheck="    + mOriCheck;
   if (EAMIsInit(&mDebug))         aCom += " Debug="     + ToString(mDebug);
   if (EAMIsInit(&mTStdNbMaxTriplet)) aCom += " StdNbPtTrip="     + ToString(mTStdNbMaxTriplet);
   if (EAMIsInit(&mTQuickNbMaxTriplet)) aCom += " QNbPtTrip="     + ToString(mTQuickNbMaxTriplet);
   if (EAMIsInit(&mTNbMinTriplet)) aCom += " NbTrip="     + ToString(mTNbMinTriplet);
   // MPD corrige oubli !!!
   if (EAMIsInit(&mExpTxt)) aCom += " ExpTxt="     + ToString(mExpTxt);


   return aCom;
}


     //==============================================================
     //==============================================================
     //==============================================================

const std::string TheStdModeNewOri = "Std";

eTypeModeNO ToTypeNO(const std::string & aStr)
{
   return Str2eTypeModeNO(std::string("eModeNO_")+aStr);
}

/**********************************************************/
/*                                                        */
/*            cAppli_Martini                              */
/*                                                        */
/**********************************************************/

class cAppli_Martini : public cCommonMartiniAppli
{
      public :
          cAppli_Martini(int argc,char ** argv,bool Quick);
          void DoAll();
          void Banniere(bool Quick);
      private :

          void StdCom(const std::string & aCom,const std::string & aPost="");
          std::string mPat;
          bool        mExe;
          bool        mQuick;

          ElTimer     aChrono;
};

void cAppli_Martini::StdCom(const std::string & aCom,const std::string & aPost)
{
    std::string  aFullCom = MM3dBinFile_quotes( "TestLib ") + aCom + " "   + QUOTE(mPat);
    aFullCom = aFullCom + aPost;
    aFullCom = aFullCom + ComParam();



    if (mExe)
       System(aFullCom);
    else
       std::cout << "COM= " << aFullCom << "\n";

    std::cout << " DONE " << aCom << " in time " << aChrono.uval() << "\n";
}

void cAppli_Martini::Banniere(bool Quick)
{
    if (Quick)
    {
        std::cout <<  "\n";
        std::cout <<  " *********************************************\n";
        std::cout <<  " *     MART-ingale d'                        *\n";
        std::cout <<  " *     INI-tialisation                       *\n";
        std::cout <<  " *********************************************\n\n";
    }
    else
    {
        std::cout <<  "\n";
        std::cout <<  " *********************************************\n";
        std::cout <<  " *     MARTIN                                *\n";
        std::cout <<  " *     Gale d'                               *\n";
        std::cout <<  " *     IN-itialisation (stronger version)    *\n"; 
        std::cout <<  " *********************************************\n\n";
    }

}

void cAppli_Martini::DoAll()
{
     // 1-  Calcul de toute les orientations relatives entre paires d'images
     // NO_AllOri2Im =>  cNewO_CpleIm.cpp => TestAllNewOriImage_main
     // mm3d TestLib  NO_AllOri2Im "IMGP70.*JPG" OriCalib=AllRel Quick=1 PrefHom=
     // 
     // Appelle  TestLib NO_Ori2Im   => TestNewOriImage_main
 
     StdCom("NO_AllOri2Im");

     if (ModeNO()==eModeNO_OnlyHomogr)
     {
         return ;
     }
     
     // Homologues flottants
     // StdCom("NO_AllHomFloat"); => Supprime, pris en compte dans NO_AllOri2Im

     // 2-  Generation des fichier de points homologues  triple hom  (flottants)
     //  NO_AllImTriplet  => cNewO_PointsTriples.cpp  => CPP_GenAllImP3
     // lance en parallele pour chaque image NO_OneImTriplet
     // NO_OneImTriplet   => cNewO_PointsTriples.cpp  => CPP_GenOneImP3
     //   CPP_GenOneImP3 cree un objet de la classe cAppli_GenPTripleOneImage
     //   et appelle GenerateTriplets
     StdCom("NO_AllImTriplet");

     // 3-  Selection   des triplet
     //  NO_GenTripl =>  cNewO_OldGenTriplets.cpp   => GenTriplet_main
     StdCom("NO_GenTripl"," Show=false");

     // 4-Optimisation des triplets
     // NO_AllImOptTrip  =>  cNewO_OptimTriplet.cpp  => CPP_AllOptimTriplet_main
     // TestLib NO_OneImOptTrip  =>  cNewO_OptimTriplet.cpp  => CPP_OptimTriplet_main => cAppliOptimTriplet
     StdCom("NO_AllImOptTrip");


     // Solution initiale (et probablement definitive)
     // CPP_NewSolGolInit_main  dans cNewO_SolGlobInit.cpp
     StdCom("NO_SolInit3");
}





cAppli_Martini::cAppli_Martini(int argc,char ** argv,bool Quick) :
    mExe     (true)
/*
    mPrefHom (""),
    mExtName     (""),
    mNameModeNO  (TheStdModeNewOri),
    mInOri       ("")
*/
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Image Pat", eSAM_IsPatFile),
        LArgMain() 
                   << EAM(mExe,"Exe",true,"Execute commands, def=true (if false, only print)")
                   << ArgCMA()
   );

   if (!EAMIsInit(&mQuick)) mQuick = Quick;


  // Force la creation des auto cal
    cElemAppliSetFile anEASF(mPat);
    StdCorrecNameOrient(mNameOriCalib,anEASF.mDir);
    if (EAMIsInit(&mInOri))
       StdCorrecNameOrient(mInOri,anEASF.mDir);

    if (EAMIsInit(&mOriGPS))
       StdCorrecNameOrient(mOriGPS,anEASF.mDir);

    if (EAMIsInit(&mOriCheck))
       StdCorrecNameOrient(mOriCheck,anEASF.mDir);


    cNewO_NameManager aNM(mExtName,mPrefHom,mQuick,anEASF.mDir,mNameOriCalib,mExpTxt ? "txt" : "dat",mOriOut);
    const cInterfChantierNameManipulateur::tSet * aVIm = anEASF.SetIm();
    for (int aK=0 ; aK<int(aVIm->size()) ; aK++)
    {
          cNewO_OneIm (aNM,(*aVIm)[aK]);
    }
}


int CPP_Gene_Martini_main(int argc,char ** argv,bool Quick)
{
   MMD_InitArgcArgv(argc,argv);
   cAppli_Martini anAppli(argc,argv,Quick);
   if (MMVisualMode) return EXIT_SUCCESS;
   anAppli.DoAll();
   anAppli.Banniere(Quick);
   return EXIT_SUCCESS;
}

int CPP_Martini_main(int argc,char ** argv)
{
    return CPP_Gene_Martini_main(argc,argv,true);
}


int CPP_MartiniGin_main(int argc,char ** argv)
{
    return CPP_Gene_Martini_main(argc,argv,false);
}


/**************************************************************/


class cAppliTestMartini
{
      public :
          void OneTest(int aKIter);
          cAppliTestMartini(int argc,char ** argv) ;
      private :
          std::string mPat;
          std::string mNameOriCalib;
          std::string mExtHom;
          int         mK0;
          int         mKIter;
          double      mDist;
          double      mVGFact;
          double      mProbaSel;
};


void cAppliTestMartini::OneTest(int aKIter) 
{
   mKIter = aKIter;
   mDist  = 2000 * ElSquare(NRrandom3());
   mVGFact = 0.5 + 2 * NRrandom3();

   double aExpProba = 2.0;
   mProbaSel =  ElMax(0.0,ElMin(1.0,NRrandom3()));
   
   if (mProbaSel < 0.5)
       mProbaSel = pow(mProbaSel,aExpProba);
   else
       mProbaSel = 1.0- pow(1.0-mProbaSel,aExpProba);
   

   std::string aComRat =    MMBinFile(MM3DStr) + " Ratafia " 
                        + mPat 
                        + " Out=" + mExtHom
                        + " DistPMul=" + ToString(mDist)
                        + " MVG=" + ToString(mVGFact)
                        + " OriCalib=" + mNameOriCalib
                        + " ProbaSel=" + ToString(mProbaSel) ;

   std::cout << "RAAT " << aComRat << "\n";

   std::string aComMartini =    MMBinFile(MM3DStr) 
                                    + " Martini " 
                                    +  mPat
                                    + " ExtName=TM" 
                                    + " SH=" + mExtHom
                                    + " OriCalib=" + mNameOriCalib;
   std::string aDirPurge = "NewOriTmpTM"+mExtHom+mNameOriCalib + "Quick/";
   
   if (aKIter>= mK0)
   {
       System(aComRat);
       System(aComMartini);

       ELISE_fp::PurgeDirRecursif(aDirPurge);
   }

   std::cout << aKIter << " Purge=[" << aDirPurge << "]\n";

}

cAppliTestMartini::cAppliTestMartini(int argc,char ** argv) :
    mNameOriCalib     (""),
    mExtHom ("TestMartini"),
    mK0     (0)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mPat,"Image Pat", eSAM_IsPatFile),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mK0,"K0",true,"K fisrt iter executed")
   );
}

int TestMartini_Main(int argc,char ** argv)
{
    cAppliTestMartini anAppli(argc,argv);
    for (int aK=0; true; aK++)
    {
       anAppli.OneTest(aK);
    }

    return EXIT_SUCCESS;
}







/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est regi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusee par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilite au code source et des droits de copie,
de modification et de redistribution accordes par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitee.  Pour les mêmes raisons,
seule une responsabilite restreinte pese sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concedants successifs.

A cet egard  l'attention de l'utilisateur est attiree sur les risques
associes au chargement,    l'utilisation,    la modification et/ou au
developpement et  la reproduction du logiciel par l'utilisateur etant
donne sa specificite de logiciel libre, qui peut le rendre complexe 
manipuler et qui le reserve donc   des developpeurs et des professionnels
avertis possedant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invites a  charger  et  tester  l'adequation  du
logiciel a  leurs besoins dans des conditions permettant d'assurer la
securite de leurs systemes et ou de leurs donnees et, plus generalement,
a  l'utiliser et l'exploiter dans les mêmes conditions de securite.

Le fait que vous puissiez acceder a  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepte les
termes.
Footer-MicMac-eLiSe-25/06/2007*/


