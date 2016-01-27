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



     /*****************************************/
     /*                                       */
     /*   cIncEnsembleCamera::NROptF1vND      */
     /*                                       */
     /*****************************************/


Im1D_REAL8  cIncEnsembleCamera::CurVals()
{
   Im1D_REAL8 aRes(NbVal());

   for (INT aK=0 ; aK<NbVal() ; aK++)
       aRes.data()[aK] =  mAlloc.GetVar(aK);
   return  aRes;
}

INT cIncEnsembleCamera::NbVal()
{
   return  mAlloc.CurInc();
}

void cIncEnsembleCamera::SetPtCur(const double * aPt)
{
     for (INT aK=0 ; aK < NbVal(); aK++)
     {
         mAlloc.SetVar(aPt[aK],aK);
     }

     for (tContFcteur::iterator itF=mLFoncteurs.begin(); itF!=mLFoncteurs.end() ; itF++)
     {
         (*itF)->SetCoordCur(mAlloc.ValsVar());
     }
}

void cIncEnsembleCamera::SetLambda(REAL aLambda)
{
    Im1D_REAL8 anIm(NbVal());
    ELISE_COPY(anIm.all_pts(),mImState0.in()+mImDir.in()*aLambda,anIm.out());
    SetPtCur(anIm.data());

}

REAL cIncEnsembleCamera::ScoreCurGen(bool WithFaux,bool CumulDeriv)
{
    SetOpt();
    REAL mMoyResidu = 0.0;
    REAL  mNbPts = 0.0;

    for 
    (
             std::list<cIncSetLiaison *>::iterator anIt = mListSl->begin(); 
             anIt != mListSl->end();
             anIt ++
    )
    {
         cIncCpleCamera * aCple = (*anIt)->Cple();
         ElPackHomologue & aPack = (*anIt)->Pack();
         for
         (
                ElPackHomologue::iterator itHoms = aPack.begin();
                itHoms != aPack.end();
                itHoms++
         )
         {
            Pt2dr aP1 = itHoms->P1();
            Pt2dr aP2 = itHoms->P2();
            REAL aPds =  itHoms->Pds();

            REAL Residu = CumulDeriv ?
                          aCple->DevL1AddEqCoPlan(aP1,aP2,aPds,*mSys):
                          aCple->ValEqCoPlan(aP1,aP2);

            mMoyResidu += ElSquare(Residu )* aPds;
            mNbPts += aPds;
         }
    }

    if (WithFaux)
    {
       for (INT aK = 0 ; aK<INT(mFoncsAux.size()) ; aK++)
       {
            cFonctrPond aFP = mFoncsAux[aK];
            
            if (CumulDeriv)
               StdAddEq(aFP.mFctr,aFP.mPds);
            else
               aFP.mFctr->ComputeValAndSetIVC();
            mMoyResidu += ElSquare(aFP.mFctr->Val(0)) * aFP.mPds;
            mNbPts +=  aFP.mPds;
       }
    }


    return mMoyResidu / mNbPts;
}

void cIncEnsembleCamera::OneItereDevL1(bool WithFaux)
{
    ResetEquation();
    ScoreCurGen(WithFaux,true);
    // ItereLineaire();
    OptLineaireOnDirL2(mListSl,mFoncsAux);
}

REAL cIncEnsembleCamera::ScoreCur(bool WithFaux)
{
    return ScoreCurGen(WithFaux,false);
}




REAL cIncEnsembleCamera::ScoreLambda(REAL aLambda)
{
    SetLambda(aLambda);

    REAL aRes =  ScoreCur(true);
    return aRes;
}

void cIncEnsembleCamera::SetImState0()
{
    mImState0.Resize(NbVal());
    for (INT aK=0 ; aK < NbVal(); aK++)
         mImState0.data()[aK] = mAlloc.GetVar(aK);
}

void cIncEnsembleCamera::SetCurDir(const double * aDir)
{
    mImDir.Resize(NbVal());
    for (INT aK=0 ; aK < NbVal(); aK++)
    {
         mImDir.data()[aK] = aDir[aK];
    }
}

class cIEC_OptimCurDir : public NROptF1vND
{
    public :

       cIEC_OptimCurDir(cIncEnsembleCamera & aSet) : 
          mSet(aSet) 
       {
       }
       REAL Optim(REAL aV0,REAL aV1);

       cIncEnsembleCamera & mSet;
    
       REAL NRF1v(REAL aLamda) 
       {
            REAL aRes =  mSet.ScoreLambda(aLamda);
            return aRes;
       }
};

REAL cIEC_OptimCurDir::Optim(REAL aV0,REAL aV1)
{
    REAL aX = aV0;
    REAL bX = aV1;
    REAL cX; 
    REAL aY,bY,cY;

    mnbrack(&aX,&bX,&cX,&aY,&bY,&cY);

    REAL XMin;
    golden(aX,bX,cX,1e-5,&XMin);

    return XMin;
}


void cIncEnsembleCamera::OptLineaireOnDirL2
     (
           std::list<cIncSetLiaison *> *        aListSL,
           const std::vector<cFonctrPond> &     aFoncAux
           // bool                                 CalcMatr
     )
{
    mListSl = aListSL;
    mFoncsAux = aFoncAux;

    SetOpt();
/*
    if (CalcMatr)
    {
        ResetEquation();
        ScoreCurGen(true,true);
    }
*/

    Im1D_REAL8  anIm =  mSys->GSSR_Solve((bool *)0);


    SetImState0();
    SetCurDir(anIm.data());

    cIEC_OptimCurDir anOpti(*this);
    REAL aLambda = anOpti.Optim(0,1);

    SetLambda(aLambda);
}


/*
void cIncEnsembleCamera::OptLineaireOnDirL2
     (
           std::list<cIncSetLiaison *> *        aListSL,
           const std::vector<cFonctrPond> &     aFoncAux
     )
{
    OptLineaireOnDirL2Gen(aListSL,aFoncAux,false);
}

void cIncEnsembleCamera::MatrAndOptLineaireOnDirL2
     (
           std::list<cIncSetLiaison *> *        aListSL,
           const std::vector<cFonctrPond> &     aFoncAux
     )
{
    OptLineaireOnDirL2Gen(aListSL,aFoncAux,true);
}
*/





class  cIEC_OptimPowel : public FoncNVarND<REAL>
{
      public :

          cIEC_OptimPowel(cIncEnsembleCamera & aSet);

      private :
         REAL ValFNV(const REAL *  v);
         cIncEnsembleCamera & mSet;
};


cIEC_OptimPowel::cIEC_OptimPowel(cIncEnsembleCamera & aSet) :
   FoncNVarND<REAL>(aSet.NbVal()),
   mSet (aSet)
{
}


REAL cIEC_OptimPowel::ValFNV(const REAL *  v)
{

   mSet.SetPtCur(v);
   REAL aRes = mSet.ScoreCur(true);

   
   return aRes;
}


void cIncEnsembleCamera::OptimPowel
     (
          std::list<cIncSetLiaison *> *  aListSL,
          const std::vector<cFonctrPond> &     aFoncAux,
          REAL tol,
          INT ItMax
     )
{
    mListSl = aListSL;
    mFoncsAux = aFoncAux;
    SetOpt();
    Im1D_REAL8 I = CurVals();

    cIEC_OptimPowel aOpt(*this);
    aOpt.powel(I.data(),tol,ItMax);
    SetPtCur(I.data());
}


void cIncEnsembleCamera::OptimJacobi
     (
          std::list<cIncSetLiaison *> *  aListSL,
          const std::vector<cFonctrPond> &     aFoncAux
     )
{
    mListSl = aListSL;
    mFoncsAux = aFoncAux;
    SetOpt();
    ELISE_ASSERT(mL2Opt,"Need L2 Sys for OptimJacobi");

    mSysL2->GetMatr(mMatrL2,mMatrtB);
    jacobi_diag(mMatrL2,mMatrValP,mMatrVecP);

    mtBVecP = mMatrtB * mMatrVecP;

cElRanGen aR;
Im1D_REAL8 P0 = CurVals();
Im1D_REAL8 mImBest  = CurVals();


REAL aScInit = ScoreCur(false);
REAL aScMin = aScInit;
REAL aScAmMin = aScInit;
for (INT aTest =0 ; aTest < 100 ; aTest ++)
{
    SetPtCur(P0.data());
    Im1D_REAL8 aDir( NbVal(),0.0);
    

    for (INT aK = 0 ; aK < NbVal() ; aK++)
    {
        ELISE_ASSERT(mMatrValP(aK,aK) != 0,"Jcobi ");
        REAL Val = (mtBVecP(aK,0) /  ElAbs(mMatrValP(aK,aK)))  ;

	REAL aRan = aR.cNRrandom3()  ;

        if (aRan < 0.25)
        {
             Val = 0;
        }
        else if (aRan < 0.5)
	;
	/*
        {
             Val = Val; 
        }
	*/
        else
            Val *=  3 * aR.cNRrandom3() -1;
     

        for (INT aY =0 ; aY< NbVal() ; aY++)
            aDir.data()[aY] += mMatrVecP(aK,aY) * Val;

    }

    ELISE_COPY(aDir.all_pts(),P0.in() + aDir.in(),aDir.out());
    SetPtCur(aDir.data());
    REAL aSc = ScoreCur(false);

if (aSc < aScMin)
{
    ElSetMin(aScMin,aSc);

    for (INT aK= 0 ; aK< 10 ; aK++)
        OneItereDevL1( aK < 3);

    REAL aSc = ScoreCur(false);
    if (aSc < aScAmMin)
    {
         aScAmMin = aSc;
         ELISE_COPY(mImBest.all_pts(),CurVals().in(),mImBest.out());
    }
   
}
    cout  <<  aScMin << " " << aScInit  << " " << aSc <<  " " << aScAmMin << "\n";
}
    SetPtCur(mImBest.data());
}

/*
void cIncEnsembleCamera::OptimJacobi
     (
          std::list<cIncSetLiaison *> *  aListSL,
          const std::vector<cFonctrPond> &     aFoncAux
     )
{
    mListSl = aListSL;
    mFoncsAux = aFoncAux;
    SetOpt();
    ELISE_ASSERT(mL2Opt,"Need L2 Sys for OptimJacobi");

    mSysL2->GetMatr(mMatrL2,mMatrtB);
    jacobi_diag(mMatrL2,mMatrValP,mMatrVecP);

    mtBVecP = mMatrtB * mMatrVecP;

    Im1D_REAL8 aDir( NbVal());

    for (INT aK = 0 ; aK < NbVal() ; aK++)
    {
        SetImState0();
        INT aD = aK;
        for (INT aY =0 ; aY< NbVal() ; aY++)
            aDir.data()[aY] = mMatrVecP(aD,aY);
       SetCurDir(aDir.data());

       cIEC_OptimCurDir anOpti(*this);

       REAL VP = mMatrValP(aD,aD);
       REAL b = mtBVecP(aD,0);

       REAL Val = 0;
       if (VP >0)
           Val = b/VP;
       else if (VP<0)
           Val = -b/(2*VP);
       else
           Val = 0;

       if (Val ==0)
          Val = 1e-4;

       if (VP >0)
       {
          REAL aLambda = anOpti.Optim(0,Val);
          SetLambda(aLambda);
       }
    }
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
Footer-MicMac-eLiSe-25/06/2007*/
