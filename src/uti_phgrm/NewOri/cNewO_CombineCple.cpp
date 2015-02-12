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


/************************ cCdtCombTiep ****************/


cCdtCombTiep::cCdtCombTiep(tMerge * aM)  :
    mMerge (aM),
    mP1    (mMerge->GetVal(0)),
    mDMin  (1e10),
    mTaken (false),
    mPdsOccup (0.0)
{
}


/************************ cNewO_CombineCple ****************/



static const int NbDecoup0PIS2 = 5;
static const int NbPow2 = 4;
static const int NbTieP = 100;
static const int NbMaxInit = 3000;
static const int NbCple = 500;



double  cNewO_CombineCple::CalculCostCur()
{
    // ELISE_ASSERT(false,"HHHHHHHHHHHHHHhh");
// [-64,-48,0][-1.25664,-0.942478,0]

    // return  ElSquare(mCurTeta.x-0.2) + ElSquare(mCurTeta.y+0.5) +  ElSquare(mCurTeta.z);
    return  ElSquare(mCurTeta.x + 1.25 ) + ElSquare(mCurTeta.y+ 0.94) +  ElSquare(mCurTeta.z);
}

double cNewO_CombineCple::GetCost(const Pt3di  & aP) 
{
     int anInd = PInt2Ind(aP);
     std::map<int,double>::const_iterator iT = mMapCost.find(anInd);

     if (iT != mMapCost.end())
        return iT->second;
    
     SetCurRot(aP);
     double aCost = CalculCostCur();
     mMapCost[anInd] = aCost;

     return aCost;
}

void cNewO_CombineCple::SetCurRot(const Pt3di & aP)
{
    mCurInd = aP;
    mCurTeta= PInt2Tetas(aP);
 // std::cout << mCurInd << mCurTeta << "\n"; getchar();
    mCurRot = ElMatrix<double>::Rotation(mCurTeta.z,mCurTeta.y,mCurTeta.x);
}

Pt3dr cNewO_CombineCple::PInt2Tetas(const Pt3di  & aP) const
{
    return Pt3dr
           (
                K2Teta(aP.x),
                K2Teta(aP.y),
                K2Teta(aP.z)
           );
}

double cNewO_CombineCple::K2Teta(int aK) const
{
     return (2 * PI * aK)/ mNbStepTeta ;
}

int    cNewO_CombineCple::PInt2Ind(const Pt3di  & aP) const
{
    return      mod(aP.x,mNbStepTeta) * ElSquare(mNbStepTeta) 
             +  mod(aP.y,mNbStepTeta) *  mNbStepTeta
             +  mod(aP.z,mNbStepTeta);
}


Pt2dr cNewO_CombineCple::ToW(const Pt2dr & aP) const
{
    return (aP-mP0W) *mScaleW;
}


cNewO_CombineCple::cNewO_CombineCple(const  cFixedMergeStruct<2,Pt2dr>  &  aMap) :
    mCurStep     (1<<NbPow2),
    mNbStepTeta  (4 * NbDecoup0PIS2 * mCurStep),
    mCurRot      (3,3),
    mW           (0)
{


    // 1- Preselrection purement aleatoire d'un nombre raisonnable depoints
    const std::list<tMerge *> & aLM  = aMap.ListMerged();
    RMat_Inertie aMat;
    // Pt2dr aP0
    cRandNParmiQ aSelec(NbMaxInit,aLM.size());
    for (std::list<tMerge *>::const_iterator itM=aLM.begin() ; itM!=aLM.end() ; itM++)
    {
         if (aSelec.GetNext())
         {
            mVAllCdt.push_back(cCdtCombTiep(*itM));
            Pt2dr aP1 = (*itM)->GetVal(0);
            aMat.add_pt_en_place(aP1.x,aP1.y);
         }
    }
    aMat = aMat.normalize();
    int aNbSomTot = int(mVAllCdt.size());

    double aSurfType  =  sqrt (aMat.s11()* aMat.s22() - ElSquare(aMat.s12()));
    double aDistType = sqrt(aSurfType/aNbSomTot);

    double aSzW = 800;
    if (1)
    {
         mP0W = aMap.ValInf(0);
         Pt2dr aP1 = aMap.ValSup(0);
         Pt2dr aSz = aP1-mP0W;
         mP0W = mP0W - aSz * 0.1;
         aP1 = aP1 + aSz * 0.1;
         aSz = aP1-mP0W;

         mScaleW  = aSzW /ElMax(aSz.x,aSz.y) ;
         mW = Video_Win::PtrWStd(round_ni(aSz*mScaleW));
    }

    // Calcul d'une fonction de deponderation  
    for (int aKS1 = 0 ; aKS1 <aNbSomTot ; aKS1++)
    {
        for (int aKS2 = aKS1 ; aKS2 <aNbSomTot ; aKS2++)
        {
           // sqrt pour attenuer la ponderation
           double aDist = sqrt(dist48( mVAllCdt[aKS1].mP1-mVAllCdt[aKS2].mP1) / 2.0);
           double aPds = 1 / (aDistType+aDist);
           mVAllCdt[aKS1].mPdsOccup += aPds;
           mVAllCdt[aKS2].mPdsOccup += aPds;
        }
        if (mW)
            mW->draw_circle_abs(ToW( mVAllCdt[aKS1].mP1),2.0,mW->pdisc()(P8COL::blue));
    }
    for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
    {
       cCdtCombTiep & aCdt = mVAllCdt[aKSom];
       aCdt.mPdsOccup *= ElSquare(aCdt.mMerge->NbArc());
    }
    



    int aNbSomSel = ElMin(aNbSomTot,NbTieP);

    // Calcul de aNbSomSel points biens repartis
    ElTimer aChrono;
    for (int aKSel=0 ; aKSel<aNbSomSel ; aKSel++)
    {
         // Recherche du cdt le plus loin
         double aMaxDMin = 0;
         cCdtCombTiep * aBest = 0;
         for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
         {
             cCdtCombTiep & aCdt = mVAllCdt[aKSom];
             double aDist = aCdt.mDMin *  aCdt.mPdsOccup;
             if ((!aCdt.mTaken) &&  (aDist > aMaxDMin))
             {
                 aMaxDMin = aDist;
                 aBest = & aCdt;
             }
         }
         ELISE_ASSERT(aBest!=0,"cNewO_CombineCple");
         for (int aKSom = 0 ; aKSom <aNbSomTot ; aKSom++)
         {
             cCdtCombTiep & aCdt = mVAllCdt[aKSom];
             aCdt.mDMin = ElMin(aCdt.mDMin,dist48(aCdt.mP1-aBest->mP1));
         }
         mVCdtSel.push_back(aBest);
        if (mW)
            mW->draw_circle_abs(ToW( aBest->mP1),3.0,mW->pdisc()(P8COL::red));
    }
 

    int aNbA = NbCple;
    while (aNbA >  ((aNbSomSel * (aNbSomSel-1)) /2)) aNbA--;
    
    for (int aKA = 0 ; aKA < aNbA ; aKA++)
    {
        int aKP1 = NRrandom3(aNbSomSel);
        // Pt2dr aDir = Pt2dr::FromPolar(1,aKA);
        for (int  aKSel=0 ; aKSel<aNbSomSel ; aKSel++)
        {
            if (aKSel!=aKP1)
            {
            }
        }
    }


    if (mW) mW->clik_in();


    std::cout << "cNewO_CombineCple::cNewO_CombineCple " << aNbSomTot << "\n";
    Pt3di aP;

    std::list<Pt3di> aLPMin;

    for (aP.x =  -NbDecoup0PIS2 ; aP.x <= NbDecoup0PIS2 ; aP.x ++)
    {
         for (aP.y =  -NbDecoup0PIS2 ; aP.y <= NbDecoup0PIS2 ; aP.y ++)
         {
              for (aP.z =  - (2*NbDecoup0PIS2) ; aP.z < (2*NbDecoup0PIS2) ; aP.z ++)
              {
                    double aVC =  GetCost(aP*mCurStep);
                    bool IsMinLoc =    (aVC < GetCost((aP+Pt3di( 1,0,0)) * mCurStep))
                                    && (aVC < GetCost((aP+Pt3di(-1,0,0)) * mCurStep))
                                    && (aVC < GetCost((aP+Pt3di(0, 1,0)) * mCurStep))
                                    && (aVC < GetCost((aP+Pt3di(0,-1,0)) * mCurStep))
                                    && (aVC < GetCost((aP+Pt3di(0,0, 1)) * mCurStep))
                                    && (aVC < GetCost((aP+Pt3di(0,0,-1)) * mCurStep));

                    for (int aDx=-1 ; (aDx<=1) && IsMinLoc ; aDx++)
                    {
                        for (int aDy=-1 ; (aDy<=1) && IsMinLoc ; aDy++)
                        {
                            for (int aDz=-1 ; (aDz<=1) && IsMinLoc ; aDz++)
                            {
                                 if ((aDx!=0) || (aDy!=0) || (aDz!=0))
                                 {
                                     IsMinLoc = IsMinLoc && (aVC<GetCost( (aP+Pt3di(aDx,aDy,aDz))*mCurStep));
                                 }
                            }
                        }
                    }
                    if (IsMinLoc)
                    {
                        std::cout << " IisssMinn " << aP << "\n";
                        aLPMin.push_back(aP*mCurStep);
                    }
              }
         }
    }

/*
    std::cout << "Sssz " <<  aLPMin.size() << "\n";
    if ( aLPMin.size()>0) std::cout << "PP00 " << *(aLPMin.begin()) << "\n";
*/
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
