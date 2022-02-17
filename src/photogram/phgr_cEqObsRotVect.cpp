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

/*
    Etapes pour generer le code :

     - 1 - ecrire le code cote ELISE ! (a base de FoncNum)
     - 2 -  GenCode
     - 3 - rajouter #include "../../src/GC_photogram/cEqObsRotVect_CodGen.cpp"
           dans un des codes phgr_or_code_genX.cpp

      -4 -  rajaouetr

                 #include "../../src/GC_photogram/cEqObsRotVect_CodGen.h"

                 AddEntry("cEqObsRotVect_CodGen",cEqObsRotVect_CodGen::Alloc);   
         
            dans phgr_or_code_gen00.cpp
*/



/*********************************************************/
/*                                                       */
/*        cEqObsRotVect                                  */
/*                                                       */
/*********************************************************/


/* Role de  mLInterv;

   De maniere generale, une equation d'observation (ici cEqObsRotVect) , 
   relie par, une ou plusieurs contraintes (ici 3), un ou  plusieurs objets
   (ici un seul)  qui ont chacun un ensemble different d'inconnues.


   Ces inconnues sont reperees par des numeros l'ensemble des numeros
   d'inconnues est connexe. La class cIncIntervale represente un tel
   ensemble connexe.


   La classe qui fait le calcul, impose qu'en interne, les inconnues
   soit un ensemble connexe numerote entre 0 et N.
   Au momemnt d' utiliser un objet de cette classe,  il faut donc renumeroter
   ces inconnues dans un ensemble [0,N], pour le calcul,
   cette renumerotation doit pouvoir etre differente pour chacun des instance
   de cEqObsRotVect;

   Soit par exemple , une equation qui utilie deux orientation et une calibration.
   
   Pour des instances differente de cette equation on aura par exemple 

            EqG       |        Eq1      Eq2
                      |
    OrA    [0 6]      |      [40 46]    [53 59]
    OrB    [6 12]     |      [22 28]    [62 68]
    Cal    [12 20]    |      [0 8]      [0 8] exemple "realiste" , la calibration est unique
                                              (mais ca n'a rien d'obligatoire)

    Avec  :
          EqG  : celle cree lors de la generation de code
          Eq1, Eq2 : deux  instance creees pour utiliser le code genere

    Pour indiquer comme "maper" les nuretotation Eq1 ou Eq2  sur la
    numerotation de generation du code EqG, on utilise un mecanisme par
    identifiant de nom.   Chaque objet possede  un mLInterv (de type
    cIncListInterv)  et on caclule une correspondance de numero entre
    les EqG.mLInterv et Eq1(2).mLInterv .

      (OrA,0,6)  (OrB,6,12)... et  (OrA,40,46) .. 

    L'objet correspondant a EqG est memorise dans le code genere
    (il est cree dans le constructeur de l'objet par une serie 
     de AddIntRef(..)

    Pour chaque objet equation, et chaque element d'inconu,  on a qq ch comme :
        RotCalc->IncInterv().SetName("Orient");
        mLInterv.AddInterv(mRotCalc->IncInterv());
    la premiere ligne donne un identifiant qui sert a etablir les
    correspondance lors de la phase de generation et la phase d'utilisation

    la deuxieme ligne rajoute cette correspondance

    En sortie, on aura qq ch comme :

        mFoncEqResidu->SetMappingCur(mLInterv);

    Qui transforme la structure en une LUT de correspondance permettant
    de faire le calcul de maniere immediate.
*/



cEqObsRotVect::cEqObsRotVect
(
       cSetEqFormelles &   aSet,
       cRotationFormelle * aRot,
       bool                Code2Gen
)  :
   mSet       (aSet),
   mRotCalc   ( aRot ? 
                aRot :
                aSet.NewRotation(eRotCOptFige,ElRotation3D(Pt3dr(0,0,0),0,0,0))
              ),
   mRot2Destr (aRot ? 0 : mRotCalc),
   mN1        ("P1"),
   mN2        ("P2"),
   mResidu    (mRotCalc->VectC2M(mN1.PtF())-mN2.PtF()),
   mNameType  ("cEqObsRotVect_CodGen")
{

    ELISE_ASSERT
    (
        (&mSet==mRotCalc->Set()),
        "Set incoherence in cEqObsRotVect::cEqObsRotVect"
    );

   // En mode utilisation,  assurele lien les numeros d'inconnu de mRotCalc et 
   // la numerotation interne; en mode generation de code cree l'intervale de ref
     mRotCalc->IncInterv().SetName("Orient");
     mLInterv.AddInterv(mRotCalc->IncInterv());

    // Cette etape va creer  un objet de type "cEqObsRotVect_CodGen",
    // tant que le code n'a pas ete genere et linke, il n'y a aucune
    // raison pour que cet objet puisse etre cree, la valeur 0 sera alors
    // retournee
    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);


    if (Code2Gen)
    {
        GenCode();
        // En phase de generation il n'y a pas grand chose de plus a faire
        return;
    }

    // En theorie le mecanisme permet de fonctionner meme sans generation
    // de code par un mode degrade "interprete" ou toute les evaluation sont
    // faite directement a partir des Fonc_Num ;  cette fonctionnalite n'est
    // pas vraiment maintenue (par exemple elle ne gere par les contraintes
    // multiple);  on conserve l'architecture de code  "au cas ou "mais 
    // on en interdit l'usage

    if (mFoncEqResidu==0)
    {
       ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");      
       mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,Fonc_Num(0));
    }

    // Cree dans l'objet la lut vers les numero d'inconnues
    // reduits (ici [0,3[)
    mFoncEqResidu->SetMappingCur(mLInterv,&mSet);


    // Le code genere fait reference au valeur de P1 et P2;
    // pour pouvoir repidement faire l'initialisation on memorise
    // leur adresse (on envoie un nom, on recupere une adresse)
    // ceci est encapsule dans cP3d_Etat_PhgrF qui gere les variables
    // d'etat triple correspondant des points 3D
    mN1.InitAdr(*mFoncEqResidu);
    mN2.InitAdr(*mFoncEqResidu);


    // Il est necessaire que  mSet connaisse l'ensemble de ses "foncteur"
    // pour  :
    //    1- assurer la mise a jour des inconnues dans le cas d'un processus iteratif
    //    2- assurer la destruction

    mSet.AddFonct(mFoncEqResidu);
}

void cEqObsRotVect::GenCode()
{
    // Un objet de type equation peux gerer plusieurs equation;
    // il faut passer par un vecteur
    std::vector<Fonc_Num> aV;
    aV.push_back(mResidu.x);
    aV.push_back(mResidu.y);
    aV.push_back(mResidu.z);

    cElCompileFN::DoEverything
    (
        "CodeGenere/photogram/",  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aV,  // expressions formelles 
        mLInterv  // intervalle de reference
    );
}


void  cEqObsRotVect::AddObservation
      (
           Pt3dr aDir1,
           Pt3dr aDir2,
           double aPds,
           bool WithD2
      )
{
    mN1.SetEtat(aDir1/euclid(aDir1));
    mN2.SetEtat(aDir2/euclid(aDir2));

    mSet.AddEqFonctToSys(mFoncEqResidu,aPds,WithD2);
}

void  cEqObsRotVect::AddObservation
      (
           Pt2dr aDir1,
           Pt2dr aDir2,
           double aPds,
           bool WithD2
       )
{
    AddObservation(PZ1(aDir1),PZ1(aDir2),aPds,WithD2);
}





cEqObsRotVect::~cEqObsRotVect()
{
    // Pas de delete  : mFoncEqResidu , c'est gere
    // par mSet qui le connait grace a mSet.AddFonct(mFoncEqResidu)

    // delete mRot2Destr; pas de deletes, aussi connu par mSet
}



cRotationFormelle & cEqObsRotVect::RotF()
{
   return *mRotCalc;
}

/*********************************************************/
/*                                                       */
/*        cEqCalibCroisee                                */
/*                                                       */
/*********************************************************/

void cEqCalibCroisee::GenCode()
{
    // Un objet de type equation peux gerer plusieurs equation;
    // il faut passer par un vecteur
    std::vector<Fonc_Num> aV;
    aV.push_back(mResidu.x);
    aV.push_back(mResidu.y);
    aV.push_back(mResidu.z);

    cElCompileFN::DoEverything
    (
        "CodeGenere/photogram/",  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aV,  // expressions formelles 
        mLInterv  // intervalle de reference
    );
}


static Pt3d<Fonc_Num>   AdZ0(Pt2d<Fonc_Num>  aP)
{
   return Pt3d<Fonc_Num>(aP.x,aP.y,0);
}

cEqCalibCroisee::cEqCalibCroisee
(
       bool                      SensC2M,
       cParamIntrinsequeFormel & aPIF,
       cRotationFormelle *       aRot,
       bool                      Code2Gen
)  :
   mSet       (*aPIF.Set()),
   mPIF       (aPIF),
   mRotCalc   ( aRot ? 
                aRot :
                mSet.NewRotation(eRotCOptFige,ElRotation3D(Pt3dr(0,0,0),0,0,0))
              ),
   mRot2Destr (aRot ? 0 : mRotCalc),
   mP1        ("PIm1"),
   mN2        ("DirEsp2"),
   mResidu    (   SensC2M ?
                  (    PointNorm1(mPIF.Cam2DirRayMonde(mP1.PtF(),0))
                       - mRotCalc->VectM2C(mN2.PtF())
                  )  :
		  AdZ0(mP1.PtF() -mPIF.DirRayMonde2Cam(mRotCalc->VectM2C(mN2.PtF()),0))
              ),
   mNameType  ("cEqCalibCroisee_" +  mPIF.NameType() +"_CodGen" + std::string(SensC2M ? "C2M" : "M2C"))
{

    // Test de coherence, si rotation vient de l'exterieur
    ELISE_ASSERT
    (
        (&mSet==mRotCalc->Set()),
        "Set incoherence in cEqObsRotVect::cEqObsRotVect"
    );

   // En mode utilisation,  assurele lien les numeros d'inconnu de mRotCalc et 
   // la numerotation interne; en mode generation de code cree l'intervale de ref
     mRotCalc->IncInterv().SetName("Orient");
     mPIF.IncInterv().SetName("Calib");
     mLInterv.AddInterv(mRotCalc->IncInterv());
     mLInterv.AddInterv(mPIF.IncInterv());

    // Cette etape va creer  un objet de type "cEqCalibCroisee_Calib_CodeGen",
    // tant que le code n'a pas ete genere et linke, il n'y a aucune
    // raison pour que cet objet puisse etre cree, la valeur 0 sera alors
    // retournee
    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);


    if (Code2Gen)
    {
        GenCode();
        // En phase de generation il n'y a pas grand chose de plus a faire
        return;
    }

    // En theorie le mecanisme permet de fonctionner meme sans generation
    // de code par un mode degrade "interprete" ou toute les evaluation sont
    // faite directement a partir des Fonc_Num ;  cette fonctionnalite n'est
    // pas vraiment maintenue (par exemple elle ne gere par les contraintes
    // multiple);  on conserve l'architecture de code  "au cas ou "mais 
    // on en interdit l'usage

    if (mFoncEqResidu==0)
    {
       std::cout << "Name = " << mNameType << "\n";
       ELISE_ASSERT(false,"Can Get Code Comp for cEqCalibCroisee");      
       mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,Fonc_Num(0));
    }

    // Cree dans l'objet la lut vers les numero d'inconnues
    mFoncEqResidu->SetMappingCur(mLInterv,&mSet);


    // Le code genere fait reference au valeur de P1 et P2;
    // pour pouvoir repidement faire l'initialisation on memorise
    // leur adresse (on envoie un nom, on recupere une adresse)
    // ceci est encapsule dans cP3d_Etat_PhgrF qui gere les variables
    // d'etat triple correspondant des points 3D
    mP1.InitAdr(*mFoncEqResidu);
    mN2.InitAdr(*mFoncEqResidu);

    aPIF.InitStateOfFoncteur(mFoncEqResidu,0);

    // Il est necessaire que  mSet connaisse l'ensemble de ses "foncteur"
    // pour  :
    //    1- assurer la mise a jour des inconnues dans le cas d'un processus iteratif
    //    2- assurer la destruction

    mSet.AddFonct(mFoncEqResidu);
}

const std::vector<REAL> &  cEqCalibCroisee::AddObservation
      (
           Pt2dr aPIm1,
           Pt3dr aDir2,
           double aPds,
           bool WithD2
      )
{
    mP1.SetEtat(aPIm1);
    mN2.SetEtat(aDir2/euclid(aDir2));

    return mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,WithD2,NullPCVU);
}


cRotationFormelle & cEqCalibCroisee::RotF()
{
   return *mRotCalc;
}

cEqCalibCroisee::~cEqCalibCroisee()
{
}



/*********************************************************/
/*                                                       */
/*        cEqDirecteDistorsion                           */
/*                                                       */
/*********************************************************/

class cEDD_UsageBayer : public ElDistortion22_Gen
{
     public :
        Pt2dr Direct(Pt2dr aP) const 
	{
	     return mTr0 + mDist.Direct(aP) + (aP-mPP) * mF;
	}
        // bool OwnInverse(Pt2dr &) const ;

        void  Diff(ElMatrix<REAL> & aMat,Pt2dr aP) const
        {
           DiffByDiffFinies(aMat,aP,1e-3);
        }
	Pt2dr GuessInv(const Pt2dr & aP) const {return aP;}
	virtual ~cEDD_UsageBayer() {}

	cEDD_UsageBayer (Pt2dr aTr0,CamStenope * aCam) :
            mTr0 (aTr0),
	    mF   (aCam->Focale()),
	    mPP  (aCam->PP()),
	    mDist (aCam->Dist())
	{
	}

        Pt2dr  mTr0;
	double mF;
	Pt2dr  mPP;
	ElDistortion22_Gen  & mDist;
};



static std::string Usage2Str(cNameSpaceEqF::eTypeEqDisDirecre   Usage)
{
   switch(Usage)
   {
       case  cNameSpaceEqF::eTEDD_Reformat :
             return "Reformat";
       break;

       case cNameSpaceEqF::eTEDD_Bayer :
            return "Bayer";
       break;

       case cNameSpaceEqF::eTEDD_Interp :
            return "Interp";
       break;

       default:
       break;
   }

   ELISE_ASSERT(false,"Usage2Str");
   return "";
}



static Pt2d<Fonc_Num>  P1_EDD
                       ( 
                          cParamIntrinsequeFormel & aPIF,
                          Pt2d<Fonc_Num> aP2 , 
                          cNameSpaceEqF::eTypeEqDisDirecre   Usage 
                       )
{
   if ( Usage== cNameSpaceEqF::eTEDD_Reformat)
      return aPIF.DistorC2M(aP2);

   if (Usage == cNameSpaceEqF::eTEDD_Bayer)
      return aPIF.DistorC2M(aP2) + (aP2-aPIF.FPP()).mul(aPIF.FFoc());
   
   // eTEDD_Interp
   Pt3d<Fonc_Num> aP3 = aPIF.Cam2DirRayMonde(aP2,0);
   return Pt2d<Fonc_Num>(aP3.x,aP3.y);
}


ElDistortion22_Gen * cEqDirecteDistorsion::Dist(Pt2dr aTr0)
{
   if (mUsage == cNameSpaceEqF::eTEDD_Bayer)
      return new cEDD_UsageBayer(aTr0,mPIF.CurPIF());

   if (mUsage == cNameSpaceEqF::eTEDD_Interp)
      return new cDistStdFromCam(*mPIF.CurPIF());
      

   ELISE_ASSERT(false,"cEqDirecteDistorsion::Dist");

   return 0;
}




void cEqDirecteDistorsion::GenCode()
{
    // Un objet de type equation peux gerer plusieurs equation;
    // il faut passer par un vecteur
    std::vector<Fonc_Num> aV;
    aV.push_back(mResidu.x);
    aV.push_back(mResidu.y);

    cElCompileFN::DoEverything
    (
        "CodeGenere/photogram/",  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aV,  // expressions formelles 
        mLInterv  // intervalle de reference
    );
}

cEqDirecteDistorsion:: cEqDirecteDistorsion
(  
     cParamIntrinsequeFormel & aPIF,
     eTypeEqDisDirecre         Usage,
     bool                Code2Gen
)  :
   mUsage     (Usage),
   mSet       (*aPIF.Set()),
   mPIF       (aPIF),
   mP1        ("PIm1"),
   mP2        ("PIm2"),
   mResidu    (P1_EDD(mPIF,mP1.PtF(),Usage) - mP2.PtF()),
   mNameType  (    "cEqDirectDist" 
                 +  mPIF.NameType() 
		 +  (Usage2Str(Usage))
		 +"_CodGen"
              )
{
     mPIF.IncInterv().SetName("Calib");
     mLInterv.AddInterv(mPIF.IncInterv());

     mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);

    if (Code2Gen)
    {
        GenCode();
        return;
    }


    if (mFoncEqResidu==0)
    {
       std::cout << "Name = " << mNameType << "\n";
       ELISE_ASSERT(false,"Can Get Code Comp for cEqCalibCroisee");      
       mFoncEqResidu = cElCompiledFonc::DynamicAlloc(mLInterv,Fonc_Num(0));
    }

    mFoncEqResidu->SetMappingCur(mLInterv,&mSet);


    mP1.InitAdr(*mFoncEqResidu);
    mP2.InitAdr(*mFoncEqResidu);

    aPIF.InitStateOfFoncteur(mFoncEqResidu,0);
    mSet.AddFonct(mFoncEqResidu);
}

const std::vector<REAL> &     
cEqDirecteDistorsion::AddObservation
(
      Pt2dr aPIm1,
      Pt2dr aPIm2,
      double aPds,
      bool WithD2
)
{
    mP1.SetEtat(aPIm1);
    mP2.SetEtat(aPIm2);

    return mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,WithD2,NullPCVU);
}

cEqDirecteDistorsion::~cEqDirecteDistorsion()
{
}

/*********************************************************/
/*                                                       */
/*        cSetEqFormelles                                */
/*                                                       */
/*********************************************************/


cEqPlanInconnuFormel * cSetEqFormelles::NewEqPlanIF
                       (
		            cTFI_Triangle * aTri,
			    bool Code2Gen
                       )
{
   cEqPlanInconnuFormel * aRes  = new cEqPlanInconnuFormel(aTri,Code2Gen);
   AddObj2Kill(aRes);
   return aRes;
}

cTFI_AttrSom * cSetEqFormelles::AttrP3(const Pt3dr &aP)
{
   std::vector<double> aV;
   aV.push_back(aP.z);

   return new cTFI_AttrSom(*this,Pt2dr(aP.x,aP.y),aV);
}


cEqPlanInconnuFormel * cSetEqFormelles::NewEqPlanIF
                       (
                           const Pt3dr &aP0,
                           const Pt3dr &aP1,
                           const Pt3dr &aP2
                       )
{
   return NewEqPlanIF
          (
	       cTFI_Triangle::NewOne
	       (
	            *AttrP3(aP0),
	            *AttrP3(aP1),
	            *AttrP3(aP2)
	       ),
	       false
	  );

}

cEqObsRotVect * cSetEqFormelles::NewEqObsRotVect
                (
                       cRotationFormelle * aRot, 
                       bool Code2Gen 
                )
{
   cEqObsRotVect * aRes  = new cEqObsRotVect(*this,aRot,Code2Gen);
   //pour les cElemEqFormelle qui alloue des variables, par pour une equation
   // aRes->CloseEEF(); 
   AddObj2Kill(aRes);
   return aRes;

}

cEqCalibCroisee * cSetEqFormelles::NewEqCalibCroisee
                  (
		       bool  C2M,
                       cParamIntrinsequeFormel & aPIF,
                       cRotationFormelle * aRot, 
                       bool Code2Gen 
                  )
{
   cEqCalibCroisee * aRes  = new cEqCalibCroisee(C2M,aPIF,aRot,Code2Gen);
   //pour les cElemEqFormelle qui alloue des variables, par pour une equation
   // aRes->CloseEEF(); 
   AddObj2Kill(aRes);
   return aRes;
}

cEqDirecteDistorsion * cSetEqFormelles::NewEqDirecteDistorsion
                  (
                       cParamIntrinsequeFormel & aPIF,
                       eTypeEqDisDirecre   Usage,
                       bool Code2Gen 
                  )
{
   cEqDirecteDistorsion * aRes  = new cEqDirecteDistorsion(aPIF,Usage,Code2Gen);
   //pour les cElemEqFormelle qui alloue des variables, par pour une equation
   // aRes->CloseEEF(); 
   AddObj2Kill(aRes);
   return aRes;
}

class cChangeCamFormat
{
    public :
      cChangeCamFormat
      (
	   CamStenope    * aCamIn,
           bool C2M,
           double Resol,
           Pt2dr  Origine
      );
      void DoAll();
      Pt2dr  ToNewCoord(Pt2dr aP);

      void SetDRad(bool CDistPPLie,int aDegRadOut);

      CamStenope *  mCamInput;
      bool          mC2M;
      double        mResol;
      Pt2dr         mOrigine;
      double        mFInit;
      Pt2dr         mPPInit;
      Pt2di         mSz;
      double        mRay;
      cSetEqFormelles         mSet;
      CamStenope *              mCamInit;
      cParamIFDistRadiale *     mPIF_DR;
      cParamIntrinsequeFormel * mPIF;
      cEqCalibCroisee       *  mEq;

      bool                     mCDistPPLie;
      int                      mDegRadOut;
};

void cChangeCamFormat::SetDRad(bool CDistPPLie,int aDegRadOut)
{
   mCDistPPLie = CDistPPLie;
   mDegRadOut  = aDegRadOut;

   ElDistRadiale_PolynImpair aDist =  ElDistRadiale_PolynImpair::DistId(mRay,mPPInit,5);
   cCamStenopeDistRadPol * aCamDRInit = new cCamStenopeDistRadPol(mC2M,mFInit,mPPInit,aDist,mCamInput->ParamAF());

   mCamInit = aCamDRInit;

   mPIF_DR  = mSet.NewIntrDistRad (mC2M,aCamDRInit,0);
   mPIF_DR->SetFocFree(false);
   mPIF_DR->SetLibertePPAndCDist(false,false);

   mPIF = mPIF_DR;
}


void cChangeCamFormat::DoAll()
{
   mEq =  mSet.NewEqCalibCroisee(mC2M,*mPIF);
   mSet.SetClosed();

   for (int aKet=0 ; aKet< 30 ; aKet++)
   {
       if (mPIF_DR)
       {
           if (aKet== 1)
	      mPIF_DR->SetFocFree(true);

           if (aKet >= 3)
	   {
	      int aDegre =  ElMin(mDegRadOut,1+ (aKet-3)/3);
	      mPIF_DR->SetDRFDegreFige(aDegre);
           }

           if (aKet >= 10)
	       mPIF_DR->SetCDistPPLie();

           if ((aKet >= 15) && (! mCDistPPLie))
               mPIF_DR->SetLibertePPAndCDist(true,true);
       }
       else 
       {
          ELISE_ASSERT(false,"Unknown Cam Modele");
       }
       int aNBP = 20;
       mSet.AddContrainte(mPIF->StdContraintes(),true);
       mSet.AddContrainte(mEq->RotF().StdContraintes(),true);

       for (int aKx=0 ; aKx<= aNBP ; aKx++)
       {
            double aPdsX = aKx / double(aNBP);
            for (int aKy=0 ; aKy<= aNBP ; aKy++)
            {
                double aPdsY = aKy / double(aNBP);
	        Pt2dr aP0  (mSz.x*aPdsX,mSz.y*aPdsY);
		Pt3dr aDir = mCamInput->F2toDirRayonL3(aP0);
		Pt2dr aPNew = ToNewCoord(aP0);
		mEq->AddObservation(aPNew,aDir);
            }
       }

        mSet.SolveResetUpdate();
   }
}

Pt2dr  cChangeCamFormat::ToNewCoord(Pt2dr aP)
{
   return (aP-mOrigine)*mResol;
}

cChangeCamFormat::cChangeCamFormat
(
      CamStenope *  aCamIn,
      bool C2M,
      double  aResol,
      Pt2dr  anOrigine
)  :
   mCamInput(aCamIn),
   mC2M     (C2M),
   mResol   (aResol),
   mOrigine (anOrigine),
   mFInit   (mCamInput->Focale() * aResol),
   mPPInit  (ToNewCoord(mCamInput->PP())),
   mSz      (mCamInput->Sz()),
   mRay     ((euclid(mSz)/2.0) * aResol),
   mSet     (cNameSpaceEqF::eSysPlein,1000),
   mPIF_DR  (0),
   mPIF     (0)
{
}



cCamStenopeDistRadPol *  CamStenope::Change2Format_DRP
                       (
	                    bool C2M,
                            int  aDegreOut,
                            bool CDistPPLie,
                            double aResol,
                            Pt2dr  anOrigine
                       )
{
    cChangeCamFormat aCF(this,C2M,aResol,anOrigine);
    aCF.SetDRad(CDistPPLie,aDegreOut);
    aCF.DoAll();

    return aCF.mPIF_DR->CurPIFPolRad();
}

/*********************************************************/
/*                                                       */
/*        cSurfInconnueFormelle                          */
/*                                                       */
/*********************************************************/

cSurfInconnueFormelle::cSurfInconnueFormelle(cSetEqFormelles & aSet) :
  mSet    (aSet),
  mEqP3I  (mSet.Pt3dIncTmp())
{
}


/*********************************************************/
/*                                                       */
/*        cEqPlanInconnuFormel                           */
/*                                                       */
/*********************************************************/

cEqPlanInconnuFormel::cEqPlanInconnuFormel
(
      cTFI_Triangle * aTri,
      bool            Code2DGen
)  :
   cSurfInconnueFormelle(aTri->Set()),
   mTri        (aTri),
   mPlanCur    (mTri->CalcPlancCurValAsZ()),
   mNameType   ("cCodeGenEqPlanInconnuFormel")
{
  cMatr_Etat_PhgrF    aMatEtat("CoefBar",3,3);

  ELISE_ASSERT(aTri->Dim()==1,"Bad dim in cEqPlanInconnuFormel");

  mLInterv.AddInterv(aTri->IntervA1());
  mLInterv.AddInterv(aTri->IntervA2());
  mLInterv.AddInterv(aTri->IntervA3());
  mLInterv.AddInterv(mEqP3I->IncInterv());


  mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
  if (Code2DGen)
  {
      GenCode(aMatEtat);
      return;
  }

  if (mFoncEqResidu==0)
  {
       ELISE_ASSERT(false,"Can Get Code Comp for cCameraFormelle::cEqAppui");      
  }

  mFoncEqResidu->SetMappingCur(mLInterv,&mSet);
  aMatEtat.InitAdr(*mFoncEqResidu);
  aMatEtat.SetEtat ( mTri->TriGeom().MatCoeffBarry());

  mSet.AddFonct(mFoncEqResidu);
}


double  cEqPlanInconnuFormel::DoResiduPInc(double aPds)
{
  const std::vector<REAL> & aVals =
                      (aPds > 0)                                             ?
                       mSet.VAddEqFonctToSys(mFoncEqResidu,aPds,false,NullPCVU)   :
                       mSet.VResiduSigne(mFoncEqResidu)                  ;
   return aVals[0];
}


void cEqPlanInconnuFormel::GenCode(const cMatr_Etat_PhgrF & aME)
{
   Pt3d<Fonc_Num> aPInc = mEqP3I->PF();
   Pt3d<Fonc_Num> aPBary = aME.Mat() * Pt3d<Fonc_Num>(aPInc.x,aPInc.y,1.0);

   Pt3d<Fonc_Num> aPZ
                  (
                     mTri->S1().ValsIncAsScal(),
                     mTri->S2().ValsIncAsScal(),
                     mTri->S3().ValsIncAsScal()
		  );

    std::vector<Fonc_Num> aVEcart;
    aVEcart.push_back(aPInc.z - scal(aPZ,aPBary));

    cElCompileFN::DoEverything
    (
        "CodeGenere/photogram/",  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aVEcart,  // expressions formelles 
        mLInterv  // intervalle de reference
  );
}

Pt3dr cEqPlanInconnuFormel::InterSurfCur(const ElSeg3D & aSeg)  const
{
   return mPlanCur.Inter(aSeg);
}

cEqPlanInconnuFormel::~cEqPlanInconnuFormel()
{
}

void cEqPlanInconnuFormel::Update_0F2D()
{
   mPlanCur = mTri->CalcPlancCurValAsZ();
}


cIncListInterv & cEqPlanInconnuFormel::IntervSomInc()
{
   return mLInterv;
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
