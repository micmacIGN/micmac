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

const  std::string &  cAllocNameFromInt::NameKth(int aNB)
{

    for (INT aK=(INT) mNAMES.size() ; aK<= aNB ; aK++)
        mNAMES.push_back(mRac+ToString(aK));
    return mNAMES[aNB];
}

cAllocNameFromInt::cAllocNameFromInt(const std::string & aRac) : 
     mRac(aRac) 
{
}

/**************************************************************/
/*                                                            */
/*                  cSetVar                                   */
/*                                                            */
/**************************************************************/

const std::string cElCompiledFonc::NameFoncSetVar("Var");


cElCompiledFonc * cElCompiledFonc::FoncSetVar(cSetEqFormelles * aSet,INT Ind,bool GenCode)
{
   Fonc_Num fEc = FX -cVarSpec(0,NameFoncSetVar);
   cIncListInterv IntInit(false,"toto",0,1);

   std::string aName("cSetVar");
   cElCompiledFonc * aRes = AllocFromName(aName);
   if (GenCode)
   {
         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL,
             aName,
             fEc,
             IntInit
         );
         return aRes;
   }
   if (aRes == 0)
   {
         ELISE_ASSERT(false,"Recompile : New Functeur : FoncSetVar");
         aRes  =  DynamicAlloc(IntInit,fEc);
   }
   cIncListInterv IntRef(false,"toto",Ind,Ind+1);
   aRes->SetMappingCur(IntRef,aSet);
   return aRes;
}

double * cElCompiledFonc::FoncSetVarAdr()
{
     return RequireAdrVarLocFromString(NameFoncSetVar);
}

/**************************************************************/
/*                                                            */
/*                  cSetValsEq                                */
/*                                                            */
/**************************************************************/

// Foncteur pour 
//   Xind1 - XInd2 = Cste
//   Par defaut Cste = 0
//   On peut fixer la valeur de Cste  par FoncSetVarAdr()

cElCompiledFonc * cElCompiledFonc::FoncSetValsEq
                  (
                       cSetEqFormelles * aSet,
		       INT Ind1,
		       INT Ind2,
		       bool GenCode
		  )
{
   Fonc_Num fEc = FX -FY -cVarSpec(0,NameFoncSetVar);
   cIncListInterv IntInit(false,"V1",0,1);
   IntInit.AddInterv(cIncIntervale("V2",1,2,false));

   std::string aName("cSetValsEq");
   cElCompiledFonc * aRes = AllocFromName(aName);
   if (GenCode)
   {
         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL,
             aName,
             fEc,
             IntInit
         );
         return aRes;
   }
   if (aRes == 0)
   {
         ELISE_ASSERT(false,"Recompile : New Functeur : FoncSetVar");
         aRes  =  DynamicAlloc(IntInit,fEc);
   }
   cIncListInterv IntRef(false,"V1",Ind1,Ind1+1);
   IntRef.AddInterv(cIncIntervale("V2",Ind2,Ind2+1,false));
   aRes->SetMappingCur(IntRef,aSet);
   *(aRes->FoncSetVarAdr()) = 0;
   return aRes;
}


/**************************************************************/
/*                                                            */
/*                   Affine                                   */
/*                                                            */
/**************************************************************/

static cAllocNameFromInt aNameAffine(cElCompiledFonc::NameFoncSetVar);


cElCompiledFonc * cElCompiledFonc::FoncRappelAffine(cSetEqFormelles * aSet,INT Ind0,INT NbInd)
{
    Fonc_Num fEc = -cVarSpec(0,NameFoncSetVar);
    for (INT aK=0 ; aK<NbInd ; aK++)
        fEc = fEc +  kth_coord(aK)*cVarSpec(0,aNameAffine.NameKth(aK));
   cIncListInterv IntInit(false,"toto",0,NbInd);
   cElCompiledFonc * aRes  =  DynamicAlloc(IntInit,fEc);
   cIncListInterv IntRef(false,"toto",Ind0,Ind0+NbInd);
   aRes->SetMappingCur(IntRef,aSet);
   return aRes;
}

double * cElCompiledFonc::FoncAffAdrCste()
{
     return RequireAdrVarLocFromString(NameFoncSetVar);
}
double * cElCompiledFonc::FoncAffAdrKth(INT aK)
{
     return RequireAdrVarLocFromString(aNameAffine.NameKth(aK));
}

/**************************************************************/
/*                                                            */
/*                   cSetNormEuclid                           */
/*                                                            */
/**************************************************************/

static cAllocNameFromInt aNameClSetNormEucl("cSetNormEuclid");


cElCompiledFonc * cElCompiledFonc::FoncFixeNormEucl(cSetEqFormelles * aSet,INT Ind0,INT NbInd,REAL Val,bool GenCode)
{
    Fonc_Num fEc = -cVarSpec(0,NameFoncSetVar);
    for (INT aK=0 ; aK<NbInd ; aK++)
        fEc = fEc +  Square(kth_coord(aK));
   cIncListInterv IntInit(false,"toto",0,NbInd);
   std::string aNameClass = aNameClSetNormEucl.NameKth(NbInd);

   cElCompiledFonc * aRes  =  AllocFromName(aNameClass);

   if (GenCode)
   {
         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL, //  "src/GC_photogram/",
             aNameClass,
             fEc,
             IntInit
         );
	 return aRes;
   }
   if (aRes == 0)
   {
         ELISE_ASSERT(false,"Recompile : New Functeur : FoncSetVar");

       aRes = DynamicAlloc(IntInit,fEc);
   }

   cIncListInterv IntRef(false,"toto",Ind0,Ind0+NbInd);
   aRes->SetMappingCur(IntRef,aSet);
   *(aRes->RequireAdrVarLocFromString(NameFoncSetVar)) = ElSquare(Val);
   return aRes;
}

/**************************************************************/
/*                                                            */
/*                   cSetNormEuclidVect                       */
/*                                                            */
/**************************************************************/

static cAllocNameFromInt aNameClSetNormEuclVect("cSetNormEuclidVect");
static cAllocNameFromInt aNameClSetScal("cSetScal");



cElCompiledFonc * cElCompiledFonc::FoncFixedNormScal
                   (cSetEqFormelles * aSet,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool Code2Gen,
                    cAllocNameFromInt & aNameAlloc,bool ModeNorm)
{
   Fonc_Num fEc = -cVarSpec(0,NameFoncSetVar);
   for (INT aK=0 ; aK<NbInd ; aK++)
   {
        if (ModeNorm) 
           fEc = fEc +  Square(kth_coord(aK)-kth_coord(aK+NbInd));
        else
           fEc = fEc +  kth_coord(aK)*kth_coord(aK+NbInd);
   }
   cIncListInterv IntInit(false,"Pt1",0,NbInd);
   IntInit.AddInterv(cIncIntervale("Pt2",NbInd,2*NbInd,false));
   std::string aNameClass = aNameAlloc.NameKth(NbInd);

   cElCompiledFonc * aRes  =  AllocFromName(aNameClass);

   if (Code2Gen)
   {
         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL,
             aNameClass,
             fEc,
             IntInit
         );
	 return aRes;
   }
   if (aRes == 0)
   {
         ELISE_ASSERT(false,"Recompile : New Functeur : FoncSetVar");

       aRes = DynamicAlloc(IntInit,fEc);
   }

   cIncListInterv IntRef(false,"Pt1",Ind0,Ind0+NbInd);
   IntRef.AddInterv(cIncIntervale("Pt2",Ind1,Ind1+NbInd,false));
   aRes->SetMappingCur(IntRef,aSet);
   *(aRes->RequireAdrVarLocFromString(NameFoncSetVar)) = ElSquare(Val);
   return aRes;
}

void cElCompiledFonc::SetNormValFtcrFixedNormEuclid(REAL Val)
{
    *(RequireAdrVarLocFromString(NameFoncSetVar)) = ElSquare(Val);
}


cElCompiledFonc * cElCompiledFonc::FoncFixeNormEuclVect
                   (cSetEqFormelles * aSet,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool Code2Gen)
{
    return FoncFixedNormScal(aSet,Ind0,Ind1,NbInd,Val,Code2Gen,
                             aNameClSetNormEuclVect,true);
}


cElCompiledFonc * cElCompiledFonc::FoncFixedScal
                   (cSetEqFormelles * aSet,INT Ind0,INT Ind1,INT NbInd,REAL Val,bool Code2Gen)
{
    return FoncFixedNormScal(aSet,Ind0,Ind1,NbInd,Val,Code2Gen,
                             aNameClSetScal,false);
}



/**************************************************************/
/*                                                            */
/*                      Regularisation                        */
/*                                                            */
/**************************************************************/

cElCompiledFonc * cElCompiledFonc::GenFoncVarsInd
                   (cSetEqFormelles * aSet,const std::string & aName,INT aNbVar,std::vector<Fonc_Num> aFonc,bool Code2Gen)
{
    cIncListInterv aListInt;
    for (INT aK=0 ; aK<aNbVar ; aK++)
    {
         aListInt.AddInterv(cIncIntervale(std::string("toto")+ToString(aK),aK,aK+1,false));
    }

   if (Code2Gen)
   {
         cElCompileFN::DoEverything
         (
             DIRECTORY_GENCODE_FORMEL,
             aName,
             aFonc,
             aListInt
         );
	 return 0;
   }
   cElCompiledFonc * aRes  =  AllocFromName(aName);
   if (aRes == 0)
   {
         ELISE_ASSERT(false,"Recompile : New Functeur : FoncSetVar");
   }
   aRes->SetMappingCur(aListInt,aSet);
   return aRes;
}

//    2
//    1 0

static std::string aNameRegulD1("cRegD1");
cElCompiledFonc * cElCompiledFonc::RegulD1(cSetEqFormelles * aSet,bool Code2Gen)
{
	std::vector<Fonc_Num> VF;
	VF.push_back(kth_coord(1)-kth_coord(0));
	VF.push_back(kth_coord(1)-kth_coord(2));
	return GenFoncVarsInd(aSet,aNameRegulD1,3,VF,Code2Gen);
}

//    0 1 2
//    3 4 5
//    6 7 8

static std::string aNameRegulD2("cRegD2");
cElCompiledFonc * cElCompiledFonc::RegulD2(cSetEqFormelles * aSet,bool Code2Gen)
{
	std::vector<Fonc_Num> VF;
	VF.push_back(kth_coord(3)+kth_coord(5)-2*kth_coord(4));
	VF.push_back(kth_coord(1)+kth_coord(7)-2*kth_coord(4));
	VF.push_back(kth_coord(6)+kth_coord(2)-2*kth_coord(4));
	VF.push_back(kth_coord(0)+kth_coord(8)-2*kth_coord(4));

	return GenFoncVarsInd(aSet,aNameRegulD2,9,VF,Code2Gen);
}



/******************************/

Fonc_Num CorrelSquare          
         (
             Fonc_Num aF1, 
             Fonc_Num aF2,
             Fonc_Num aPond,
             INT  aSzW,
             REAL anEpsilon
         )
{
   Symb_FNum aS1(aF1);
   Symb_FNum aS2(aF2);
   Symb_FNum aSP(aPond);
   Symb_FNum aS6
             (
                 rect_som
                 (
                    Virgule
                    (
                         aSP,
                         aSP*aS1,
                         aSP*aS2,
                         aSP*Square(aS1),
                         aSP*Square(aS2),
                         aSP*aS1*aS2
                    ),
                    aSzW
                 )
             );

    Symb_FNum aSom0  = aS6.kth_proj(0);
    Symb_FNum aMoy1  = aS6.kth_proj(1) / aSom0;
    Symb_FNum aMoy2  = aS6.kth_proj(2) / aSom0;
    Symb_FNum aEc1   = aS6.kth_proj(3) / aSom0 - Square(aMoy1);
    Symb_FNum aEc2   = aS6.kth_proj(4) / aSom0 - Square(aMoy2);
    Symb_FNum aCov12 = aS6.kth_proj(5) / aSom0 - aMoy1 * aMoy2;

    return aCov12 / sqrt(Max(anEpsilon,aEc1*aEc2));
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
