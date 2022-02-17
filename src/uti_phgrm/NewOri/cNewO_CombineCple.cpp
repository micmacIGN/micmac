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


/* Portion de code qui n'est plus utilisee actuellement, correspondait a des tentative pour  selectionner des points ...*/

class cCdtCombTiep
{
    public :
        typedef cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> tMerge;
        cCdtCombTiep(tMerge * aM) ;
        Pt3dr NormQ1Q2();

        tMerge * mMerge;
        Pt2dr    mP1;
        double   mDMin;
        bool     mTaken;
        double   mPdsOccup;
        Pt3dr    mQ1;
        Pt3dr    mQ2;
        Pt3dr    mQ2Init;
};
class cNewO_CombineCple
{
    public :
         typedef cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> tMerge;
         cNewO_CombineCple(const   cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> >   & aM,ElRotation3D * aTestSol);

          const cXml_Ori2Im &  Result() const;
    private :
          cXml_Ori2Im  mResult;
          double CostOneArc(const Pt2di &);
          double CostOneBase(const Pt3dr & aBase);

          Pt2dr ToW(const Pt2dr &) const;
          void SetCurRot(const Pt3di & aP);
          void SetCurRot(const  ElMatrix<double> & aP);

          double K2Teta(int aK) const;
          int    PInt2Ind(const Pt3di  & aP) const;
          Pt3dr   PInt2Tetas(const Pt3di  & aP) const;

          double GetCost(const Pt3di  & aP) ;
          double  CalculCostCur();

          int               mCurStep;
          int               mNbStepTeta;
          ElMatrix<double>  mCurRot;
          Pt3di             mCurInd;
          Pt3dr             mCurTeta;
          Pt3dr             mCurBase;

          std::map<int,double>     mMapCost;
          std::vector<cCdtCombTiep> mVAllCdt;
          std::vector<cCdtCombTiep*> mVCdtSel;
          std::list<Pt2di>         mLArcs;

          Video_Win *                mW;
          double                     mScaleW;
          Pt2dr                      mP0W;
         
};





/************************ cCdtCombTiep ****************/


cCdtCombTiep::cCdtCombTiep(tMerge * aM)  :
    mMerge (aM),
    mP1    (mMerge->GetVal(0)),
    mDMin  (1e10),
    mTaken (false),
    mPdsOccup (0.0)
{
}

Pt3dr cCdtCombTiep::NormQ1Q2()
{
   return mQ1 ^ mQ2;
}



/************************ cNewO_CombineCple ****************/

// img_0778.cr2 img_0779.cr2
//  5 => COST 0.00286011 PMIN [0,0,0] NbMinLoc 2
// 10 COST 0.0026756 PMIN [-1.09956,1.41372,-2.19911] NbMinLoc 3
// 20 = > COST 0.00222331 PMIN [-1.25664,1.33518,-2.12058] NbMinLoc 18
// 40 COST 0.00197137 PMIN [-1.21737,1.37445,-2.12058] NbMinLoc 112


// En Mode Somme 
// COST 0.149567 PMIN [-1.02102,1.49226,-2.19911] NbMinLoc 21


// mm3d TestLib TNO img_0778.cr2 img_0777.cr2

// 10 COST 0.0018939 PMIN [0.15708,0,0] NbMinLoc 3
// 20 COST 0.00174119 PMIN [0.15708,0,0] NbMinLoc 23
// 40 COST 0.0012423 PMIN [0.235619,0.0392699,0.0392699] NbMinLoc 181



static const int NbDecoup0PIS2 = 20;
static const int NbPow2 = 4;
static const int NbTieP = 200;
static const int NbMaxInit = 1500;
static const int NbCple = 40;


/*
Pt3dr  cNewO_CombineCple::BaseOneArc(const Pt2di & anArc,bool & Ok)
{
     
    mCurBase = mVCdtSel[anArc.x]->NormQ1Q2() ^ mVCdtSel[anArc.y]->NormQ1Q2();
    double aNormBase = euclid(mCurBase);
    if (aNormBase<1e-9) return 1e9;
}
*/

double cNewO_CombineCple::CostOneArc(const Pt2di & anArc)
{
    return CostOneBase(mVCdtSel[anArc.x]->NormQ1Q2() ^ mVCdtSel[anArc.y]->NormQ1Q2() );
}

double cNewO_CombineCple::CostOneBase(const Pt3dr & aBase)
{
    // mCurBase = mVCdtSel[anArc.x]->NormQ1Q2() ^ mVCdtSel[anArc.y]->NormQ1Q2();
    mCurBase = aBase;
    double aNormBase = euclid(mCurBase);
    if (aNormBase<1e-9) return 1e9;
    mCurBase = mCurBase / aNormBase;

    double aRes = 0;
    double aStepTeta=  ((2*PI*mCurStep) / mNbStepTeta) / 2.0 ;

    // std::cout << "aStepTeta= " << aStepTeta << "\n";


    for (int aK =0 ;  aK < int (mVCdtSel.size()) ; aK++)
    {
         ElSeg3D aS1(Pt3dr(0,0,0),mVCdtSel[aK]->mQ1);
         ElSeg3D aS2(mCurBase,mCurBase+mVCdtSel[aK]->mQ2);

         Pt3dr anI = aS1.PseudoInter(aS2);
         double aDist = aS1.DistDoite(anI);
         double aA1 = ElAbs(aS1.AbscOfProj(anI));
         double aA2 = ElAbs(aS1.AbscOfProj(anI));

         double anEc = aDist/aA1 + aDist/aA2;

          
          aRes +=   (anEc*aStepTeta) / (anEc + aStepTeta);
    }

    return  aRes / mVCdtSel.size();
}
  

double  cNewO_CombineCple::CalculCostCur()
{
    double aCostMin = 1e10;

    for (std::list<Pt2di>::const_iterator itA=mLArcs.begin() ; itA!=mLArcs.end() ; itA++)
    {
          double aCost = CostOneArc(*itA);
          if (aCost < aCostMin)
          {
               aCostMin = aCost;
          }
/*
*/
         aCostMin += aCost;
    }
    return aCostMin;
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
    SetCurRot(ElMatrix<double>::Rotation(mCurTeta.z,mCurTeta.y,mCurTeta.x));
}

void cNewO_CombineCple::SetCurRot(const ElMatrix<double> & aR)
{

 // std::cout << mCurInd << mCurTeta << "\n"; getchar();
    mCurRot = aR;

    for (int aK=0 ; aK<int(mVCdtSel.size()) ; aK++)
    {
          mVCdtSel[aK]->mQ2 = mCurRot * mVCdtSel[aK]->mQ2Init;
    }
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





cNewO_CombineCple::cNewO_CombineCple(const  cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2dr,cCMT_NoVal> >  &  aMap,ElRotation3D * aTestSol) :
    mCurStep     (1<<NbPow2),
    mNbStepTeta  (4 * NbDecoup0PIS2 * mCurStep),
    mCurRot      (3,3),
    mW           (0)
{

    // REDONDANT AVEC FONCTION GLOBALES FAITE APRES ....  PackReduit

    /******************************************************/
    /*                                                    */
    /*   A-        Selection des sommets                  */
    /*                                                    */
    /******************************************************/

    //------------------------------------------------------------------------
    // A- 1- Preselrection purement aleatoire d'un nombre raisonnable depoints
    //------------------------------------------------------------------------

    const std::list<tMerge *> & aLM  = aMap.ListMerged();
    RMat_Inertie aMat;

    {
       cRandNParmiQ aSelec(NbMaxInit, (int)aLM.size());
       for (std::list<tMerge *>::const_iterator itM=aLM.begin() ; itM!=aLM.end() ; itM++)
       {
            if (aSelec.GetNext())
            {
               mVAllCdt.push_back(cCdtCombTiep(*itM));
               Pt2dr aP1 = (*itM)->GetVal(0);
               aMat.add_pt_en_place(aP1.x,aP1.y);
            }
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

    //------------------------------------------------------------------------
    // A-2   Calcul d'une fonction de deponderation  
    //------------------------------------------------------------------------

    for (int aKS1 = 0 ; aKS1 <aNbSomTot ; aKS1++)
    {
        for (int aKS2 = aKS1 ; aKS2 <aNbSomTot ; aKS2++)
        {
           // sqrt pour attenuer la ponderation
           double aDist = sqrt(dist48( mVAllCdt[aKS1].mP1-mVAllCdt[aKS2].mP1) / 2.0);
// aDist=1;
           // double aDist = (dist48( mVAllCdt[aKS1].mP1-mVAllCdt[aKS2].mP1) / 2.0);
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

    //------------------------------------------------------------------------
    // A-3  Calcul de aNbSomSel points biens repartis
    //------------------------------------------------------------------------

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
         aBest->mQ1 = vunit(Pt3dr(aBest->mP1.x,aBest->mP1.y,1.0));
         Pt2dr aP2 = aBest->mMerge->GetVal(1);
         aBest->mQ2Init = vunit(Pt3dr(aP2.x,aP2.y,1.0));

         mVCdtSel.push_back(aBest);
         if (mW)
            mW->draw_circle_abs(ToW( aBest->mP1),3.0,mW->pdisc()(P8COL::red));
    }



    /******************************************************/
    /*                                                    */
    /*  B- Calcul des arcs                                */
    /*                                                    */
    /******************************************************/
 

    // B-1  Au max le nombre d'arc  possible
    int aNbA = NbCple;
    while (aNbA >  ((aNbSomSel * (aNbSomSel-1)) /2)) aNbA--;
    
    int aNbIter = (aNbA-1) / aNbSomSel + 1;
    cRandNParmiQ aSelec(aNbA- (aNbIter-1) * aNbSomSel,aNbSomSel);
    int aNbAMaj = aNbIter * aNbSomSel;


    std::vector<int> aPermut = RandPermut(aNbA);

    // B-2 Recherche des arsc
    int  aKA=0;
    for (int aCptAMaj = 0 ; aCptAMaj < aNbAMaj ; aCptAMaj++)
    { 
        // Tous les sommets sont equi repartis, sauf a la fin on choisit a hasard
        bool aSelK = true;
        if ( (aCptAMaj/aNbSomSel)== (aNbIter-1))  // Si derniere iter, test special
        {
           aSelK = aSelec.GetNext();
        }

        if (aSelK)
        {
            int aKP1 =  (aCptAMaj%aNbSomSel);
            double aTeta = (aPermut[aKA] * 2 * PI) / aNbA;
            Pt2dr aDir = Pt2dr::FromPolar(1.0,aTeta);
            // std::cout << "teta " << aTeta << "\n";
            double aBestSc=-1.0;
            int aBestK=-1;
            for (int aKP2 = 0 ; aKP2 < aNbSomSel ; aKP2++)
            {
                if (aKP2!=aKP1)
                {
                    Pt2dr aV =  (mVCdtSel[aKP2]->mP1- mVCdtSel[aKP1]->mP1) / aDir;
                    Pt2dr aU = vunit(aV);
                    
               // Favorise les llongs arc et homogeneise les directions
                    double aSc = NRrandom3() * euclid(aV) * (1/(1+ElSquare(5.0*aU.y)));
                    if ((aSc>aBestSc) && (aKP2!=aKP1))
                    {
                       aBestSc= aSc;
                       aBestK = aKP2;
                    }
                }
            }
            ELISE_ASSERT((aBestK>=0),"No Best Arc");
            mLArcs.push_back(Pt2di(aKP1,aBestK));
            if (mW)
            {
                mW->draw_seg(ToW( mVCdtSel[aKP1]->mP1),ToW( mVCdtSel[aBestK]->mP1),mW->pdisc()(P8COL::green));
            }
            aKA++;
        }
    }


    /******************************************************/
    /*                                                    */
    /*                                                    */
    /*                                                    */
    /******************************************************/

    if (mW) mW->clik_in();


    if (aTestSol)
    {
       ElRotation3D aR = * aTestSol;

       // Le sens corret a ete retabli (j'espere !!)
       // SetCurRot(aR.Mat());
       // std::cout << "Test Externe : " << CalculCostCur() <<"\n";
       // aR = aR.inv();

       SetCurRot(aR.Mat());
       std::cout << "Test Externe I : " << CalculCostCur() <<"\n";
       std::cout << "CostBase " << CostOneBase(aR.tr()) << "\n";
          // ElRotation3D *
    }

    std::cout << "cNewO_CombineCple::cNewO_CombineCple " << aNbSomTot << "\n";
    Pt3di aP;

    std::list<Pt3di> aLPMin;
    double aCostMin = 1e10;
    Pt3di aPMin(1000,1000,1000);

    for (aP.x =  -NbDecoup0PIS2 ; aP.x <= NbDecoup0PIS2 ; aP.x ++)
    {
         std::cout << "DECx " << aP.x << "\n";
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

                    int aDelta = 2;
                    for (int aDx=-aDelta ; (aDx<=aDelta) && IsMinLoc ; aDx++)
                    {
                        for (int aDy=-aDelta ; (aDy<=aDelta) && IsMinLoc ; aDy++)
                        {
                            for (int aDz=-aDelta ; (aDz<=aDelta) && IsMinLoc ; aDz++)
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
                       std::cout << " IisssMinn " << aP << " " << aVC << "\n";

                       aLPMin.push_back(aP*mCurStep);
                    }
                    if (aVC<aCostMin)
                    {
                       aPMin = aP*mCurStep;
                       aCostMin = aVC;
                    }
              }
         }
    }

    std::cout << "COST " << aCostMin  << " PMIN " << PInt2Tetas(aPMin ) << " NbMinLoc " << aLPMin.size() << "\n";

    Pt3dr aTeta =  PInt2Tetas(aPMin);
    ElMatrix<double> aR  = ElMatrix<double>::Rotation(aTeta.z,aTeta.y,aTeta.x);
    for (int aY=0 ; aY<3 ; aY++)
    {
        for (int aX=0 ; aX<3 ; aX++)
        {
            std::cout  << aR(aX,aY) << " ";
        }
        std::cout << "\n";
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
