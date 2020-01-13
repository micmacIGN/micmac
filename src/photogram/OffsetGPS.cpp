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


class cEqOffsetGPS;

/* L'equation est

      C + R * B =  GPS   + Err  (Eq1)

   Avec :

       C = centre de la camera   - inconnue
       R = matrice rotation      - inconnue
       B = base                  - inconnue
       GPS = mesure GPS          - obervation
       Err = erreur (non geree ici, modelisee dans la ponderation);
    
*/


/*****************************************************************/
/*                                                               */
/*               cBaseGPS                                        */
/*                                                               */
/*****************************************************************/


cBaseGPS::cBaseGPS  (cSetEqFormelles & aSet,const Pt3dr & aV0) :
     cElemEqFormelle (aSet,false),
     mV0             (aV0),
     mBaseInc        (mSet.Alloc().NewPt3("cBaseGPS",mV0))
{
   //aRes->CloseEEF();
   //AddObj2Kill(aRes);

}

Pt3d<Fonc_Num> cBaseGPS::BaseInc() {return mBaseInc;}
const Pt3dr &  cBaseGPS::ValueBase() const {return mV0;}

/*****************************************************************/
/*                                                               */
/*               cEqOffsetGPS                                    */
/*                                                               */
/*****************************************************************/

extern bool AllowUnsortedVarIn_SetMappingCur;



cEqOffsetGPS::cEqOffsetGPS(cRotationFormelle & aRF,cBaseGPS & aBase,bool doGenCode) :
    mSet  (aRF.Set()),
    mRot  (&aRF),
    mBase (&aBase),
    mGPS  ("GPS"),
    mNameType ("cEqObsBaseGPS" + std::string(aRF.IsGL() ? "_GL" : "")),
    mResidu   (mRot->C2M(mBase->BaseInc())- mGPS.PtF()),
    mFoncEqResidu (0)
{
/*
    ELISE_ASSERT
    (
       (! aRF.IsGL()),
       "cEqOffsetGPS to complete in Gimbal Lock Mode"
    );
*/


    AllowUnsortedVarIn_SetMappingCur = true;
    ELISE_ASSERT
    (
         mRot->Set()==mBase->Set(),
         "cEqOffsetGPS Rotation & Base do no belong to same set of unknown"
    );

     mRot->IncInterv().SetName("Orient");
     mBase->IncInterv().SetName("Base");

     mLInterv.AddInterv(mRot->IncInterv());
     mLInterv.AddInterv(mBase->IncInterv());

     if (doGenCode)
     {
         GenCode();
         return;
     }

     mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
     ELISE_ASSERT(mFoncEqResidu!=0,"Cannot allocate cEqObsBaseGPS");
     mFoncEqResidu->SetMappingCur(mLInterv,mSet);
     //  GL 
     mGPS.InitAdr(*mFoncEqResidu);
     mSet->AddFonct(mFoncEqResidu);
}


void cEqOffsetGPS::GenCode()
{
   // Un objet de type equation peux gerer plusieurs equation;
    // il faut passer par un vecteur
    std::vector<Fonc_Num> aV;
    aV.push_back(mResidu.x);
    aV.push_back(mResidu.y);
    aV.push_back(mResidu.z);

    cElCompileFN::DoEverything
    (
        DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
        mNameType,  // donne les noms de fichier .cpp et .h ainsi que les nom de classe
        aV,  // expressions formelles 
        mLInterv  // intervalle de reference
    );
}


Pt3dr  cEqOffsetGPS::AddObs(const Pt3dr & aGPS,const Pt3dr & aPds)
{
     mGPS.SetEtat(aGPS);
     if (mRot->IsGL())
     {
         ELISE_ASSERT(false,"cEqOffsetGPS::AddObs GL to complete");
        //     mMatriceGL      (isGL ? new cMatr_Etat_PhgrF("GL",3,3) : 0),
        //   mMatriceGL->SetEtat(mRot.MGL());

     }

     std::vector<double> aVPds;
     aVPds.push_back(aPds.x);
     aVPds.push_back(aPds.y);
     aVPds.push_back(aPds.z);

     const std::vector<REAL> & aResidu =  mSet->VAddEqFonctToSys(mFoncEqResidu,aVPds,false,NullPCVU);

     return Pt3dr(aResidu.at(0),aResidu.at(1),aResidu.at(2));
}

Pt3dr  cEqOffsetGPS::Residu(const Pt3dr & aGPS)
{
     mGPS.SetEtat(aGPS);
     const std::vector<REAL> & aResidu =  mSet->VResiduSigne(mFoncEqResidu);
     return Pt3dr(aResidu.at(0),aResidu.at(1),aResidu.at(2));
}




cRotationFormelle * cEqOffsetGPS::RF()
{
   return mRot;
}
cBaseGPS * cEqOffsetGPS::Base()
{
   return mBase;
}


/*****************************************************************/
/*                                                               */
/*               cSetEqFormelles                                 */
/*                                                               */
/*****************************************************************/

cBaseGPS *  cSetEqFormelles::NewBaseGPS(const Pt3dr & aV0)
{
    AssertUnClosed();

    cBaseGPS *aRes = new cBaseGPS(*this,aV0);

    aRes->CloseEEF(); // OBLIG : indique a la base cElemEqFormelle que toutes les inconnue ont ete allouees
                      // et que la base peut allouer un certain nombre d'objets utilitaire (genre Fctr de rappel)

    AddObj2Kill(aRes);
    return aRes;
}

cEqOffsetGPS * cSetEqFormelles::NewEqOffsetGPS(cRotationFormelle & aRF,cBaseGPS  &aBase,bool Code2Gen)
{
   ELISE_ASSERT(this==aRF.Set(),"NewEqOffsetGPS multiple set");
   cEqOffsetGPS * aRes = new cEqOffsetGPS(aRF,aBase,Code2Gen);

   AddObj2Kill(aRes);
   return aRes;
}

cEqOffsetGPS * cSetEqFormelles::NewEqOffsetGPS(cCameraFormelle & aCam,cBaseGPS  &aBase)
{
    return NewEqOffsetGPS(aCam.RF(),aBase);
}

//               cEqOffsetGPS * NewEqOffsetGPS(cCameraFormelle & aRF,cBaseGPS  &aBase);


void GenerateCodeEqOffsetGPS(bool aGL)
{
     cSetEqFormelles aSet;

     ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
     cRotationFormelle * aRF = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
     aRF->SetGL(aGL,ElRotation3D::Id);

     cBaseGPS * aBase = aSet.NewBaseGPS(Pt3dr(0,0,0));
     aSet.NewEqOffsetGPS(*aRF,*aBase,true);
}

void GenerateCodeEqOffsetGPS()
{
    GenerateCodeEqOffsetGPS(false);
    GenerateCodeEqOffsetGPS(true);
}


/*****************************************************/
/*                                                   */
/*       cEqRelativeGPS                              */
/*                                                   */
/*****************************************************/

const std::string cEqRelativeGPS::mNameType = "cImplEqRelativeGPS";

cEqRelativeGPS::cEqRelativeGPS
(
     cRotationFormelle & aR1,
     cRotationFormelle & aR2,
     bool CodeGen
) :
   mSet   (aR1.Set()),
   mR1    (&aR1),
   mR2    (&aR2),
   mDif21 ("Dif21")

{
    ELISE_ASSERT(mSet==(mR2->Set()),"Different unknown in cEqRelativeGPS");

    mR1->IncInterv().SetName("Ori1");
    mR2->IncInterv().SetName("Ori2");
   
    mLInterv.AddInterv(mR1->IncInterv());
    mLInterv.AddInterv(mR2->IncInterv());

    if (CodeGen)
    {
         Pt3d<Fonc_Num>  aResidu = mR2->COpt() - mR1->COpt() - mDif21.PtF();
         std::vector<Fonc_Num> aV;
         aV.push_back(aResidu.x);
         aV.push_back(aResidu.y);
         aV.push_back(aResidu.z);

         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL,  // Directory ou est localise le code genere
             mNameType,  // donne les noms de .cpp et .h  de classe
             aV,  // expressions formelles 
             mLInterv  // intervalle de reference
         );
         return;
    }

    mFoncEqResidu = cElCompiledFonc::AllocFromName(mNameType);
    ELISE_ASSERT(mFoncEqResidu!=0,"Cannot allocate cEqObsBaseGPS");
    mFoncEqResidu->SetMappingCur(mLInterv,mSet);
     //  GL 
    mDif21.InitAdr(*mFoncEqResidu);
    mSet->AddFonct(mFoncEqResidu);
}

cEqRelativeGPS * cSetEqFormelles::NewEqRelativeGPS
                 (
                     cRotationFormelle & aR1, 
                     cRotationFormelle & aR2
                 )
{
   ELISE_ASSERT(this==aR1.Set(),"NewEqRelativeGPS  multiple set");
   cEqRelativeGPS * aRes = new cEqRelativeGPS(aR1,aR2,false);

   AddObj2Kill(aRes);
   return aRes;
}

Pt3dr    cEqRelativeGPS::AddObs(const Pt3dr & aDif12,const Pt3dr & aPds)
{

     std::vector<double> aVPds;
     aVPds.push_back(aPds.x);
     aVPds.push_back(aPds.y);
     aVPds.push_back(aPds.z);

     const std::vector<REAL> & aResidu =  mSet->VAddEqFonctToSys(mFoncEqResidu,aVPds,false,NullPCVU);
     return Pt3dr(aResidu.at(0),aResidu.at(1),aResidu.at(2));
}

Pt3dr  cEqRelativeGPS::Residu(const Pt3dr & aDif12)
{
     mDif21.SetEtat(aDif12);
     const std::vector<REAL> & aResidu =  mSet->VResiduSigne(mFoncEqResidu);
     return Pt3dr(aResidu.at(0),aResidu.at(1),aResidu.at(2));
}




void GenerateCodeEqRelativeGPS()
{
     cSetEqFormelles aSet;
     ElRotation3D aRot(Pt3dr(0,0,0),0,0,0);
     cRotationFormelle * aRF1 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);
     cRotationFormelle * aRF2 = aSet.NewRotation (cNameSpaceEqF::eRotLibre,aRot);

     new cEqRelativeGPS(*aRF1,*aRF2,true);
}

// cEqOffsetGPS::cEqOffsetGPS(cRotationFormelle & aRF,Pt3d<Fonc_Num>  &aBase)


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
