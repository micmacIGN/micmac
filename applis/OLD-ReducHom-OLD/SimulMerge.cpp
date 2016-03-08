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

#include "general/all.h"
#include "private/all.h"
#include "ReducHom.h"

// #include "algo_geom/qdt_implem.h"
#include "ext_stl/numeric.h"
#include "ext_stl/cluster.h"


using namespace NS_ReducHoms;

class cNoAttr
{
};

/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/

class cSomSA
{
    public :
      cSomSA(int anX,int anY,double aDens) :
         mPt      (anX,anY),
         mDensite (aDens),
         mMarqued (false)
      {
      }

      Pt2di mPt;
      double mDensite;
      bool   mMarqued;
};
typedef cMergingNode<cSomSA,cNoAttr> tNodTestAero;

class cSimulAero
{
    public :
      cSimulAero(Pt2di aSz,Pt2dr aPercRec,double aDensFixe) :
         mSz        (aSz),
         mPercRec   (aPercRec),
         mDist      ((1-mPercRec.x/100.0),(1-mPercRec.y/100.0)),
         mNbV       (round_down(1/mDist.x),round_down(1/mDist.y))
      {
          for (int anY=0;  anY<mSz.y ; anY++)
          {
              std::vector<cSomSA *> aLine;
              mSoms.push_back(aLine);
              for (int anX=0;  anX<mSz.x ; anX++)
              {
                   cSomSA * aP = new cSomSA(anX,anY,aDensFixe+NRrandom3());
                   mTabS.push_back(aP);
                   mSoms.back().push_back(aP);
              }
          }
          
      }

      void Vois(cSomSA * aP,std::vector<cSomSA *> & aRes);

      void OnNewLeaf(tNodTestAero * aSingle) {}
      void OnNewCandidate(tNodTestAero * aN1) {}
      void OnNewMerge(tNodTestAero * aN1) {}
      double  Gain // <0, veut dire on valide pas le noeud
              (
                     tNodTestAero * aN1,tNodTestAero * aN2,
                     const std::vector<cSomSA*>&,
                     const std::vector<cSomSA*>&,
                     const std::list<std::pair<cSomSA*,cSomSA*> >&,
                     int aNewNum
              );
 

    // private :
       std::vector<std::vector<cSomSA *> > mSoms;
       std::vector<cSomSA *> mTabS;
       Pt2di mSz;
       Pt2dr mPercRec;
       Pt2dr mDist;
       Pt2di mNbV;

   //  MARQUAGE NON RENTABLE
/*
       std::set<cSomSA*> mSetCur;
       void InitSet ( const std::vector<cSomSA*> & aV)
       {
           mSetCur = std::set<cSomSA*>(aV.begin(),aV.end());
       }
       void ClearSet(const std::vector<cSomSA*> & aV)
       {
             mSetCur.clear();
       }
       bool IsInSet(cSomSA * aS)
       {
           return mSetCur.find(aS) != mSetCur.end();
       }
       void InitSet ( const std::vector<cSomSA*> & aV)
       {
           for (int aK=0; aK<int(aV.size()) ; aK++) aV[aK]->mMarqued = true;
       }
       void ClearSet(const std::vector<cSomSA*> & aV)
       {
           for (int aK=0; aK<int(aV.size()) ; aK++) aV[aK]->mMarqued = false;
       }
       bool IsInSet(cSomSA * aS)
       {
           return aS->mMarqued;
       }
*/

};

double  cSimulAero::Gain // <0, veut dire on valide pas le noeud
        (
                     tNodTestAero * aN1,tNodTestAero * aN2,
                     const std::vector<cSomSA*>&,
                     const std::vector<cSomSA*>&,
                     const std::list<std::pair<cSomSA*,cSomSA*> >& aL,
                     int aNewNum
        )
{
   double aRes = 0.0;
   for 
   (
      std::list<std::pair<cSomSA*,cSomSA*> >::const_iterator itP = aL.begin();
      itP!= aL.end();
      itP++
   )
   {
       cSomSA aQ1 = *(itP->first);
       cSomSA aQ2 = *(itP->second);
       Pt2dr aDif(
                     1.0-ElAbs(aQ1.mPt.x-aQ2.mPt.x)*mDist.x,
                     1.0-ElAbs(aQ1.mPt.y-aQ2.mPt.y)*mDist.y
                 );
       if ((aDif.x < 0) && (aDif.y < 0))
       {
           double aS = aDif.x * aDif.y;
           aRes += aS * aQ1.mDensite * aQ2.mDensite;
       }
   }
   if (aRes == 0.0) return -1;

   return aRes / pow(1.3,ElMax(aN1->Depth(),aN2->Depth()));
}

void cSimulAero::Vois(cSomSA * aP,std::vector<cSomSA *> & aRes)
{
    for (int aDx = -mNbV.x;  aDx <= mNbV.x ; aDx++)
    {
        for (int aDy = -mNbV.y;  aDy <= mNbV.y ; aDy++)
        {
            if ((aDx!=0) || (aDy!=0))
            {
               int  anX = round_ni(aP->mPt.x+aDx);
               int  anY = round_ni(aP->mPt.y+aDy);
               if ((anX>=0)&&(anY>=0)&&(anX<mSz.x)&&(anY<mSz.y))
               {
                  aRes.push_back(mSoms[anY][anX]);
               }
            }
        }
    }
}



typedef cMergingNode<int,cNoAttr> tNodTestInt;

class cTestIntParamMerge
{
    public :
       cTestIntParamMerge(int aNbTest,double aProba) :
          mNbTest (aNbTest),
          mProbaA (aProba)
       {
             for (int aK=0 ; aK< aNbTest ; aK++)
             {
                 mVTestValInt.push_back(aK);
             }
             for (int aK=0 ; aK< aNbTest ; aK++)
             {
                mVTestInt.push_back(&(mVTestValInt[aK]));
             }
       }
       void Vois(int* anIm,std::vector<int *> &);

       double  Gain // <0, veut dire on valide pas le noeud
               (
                     tNodTestInt * aN1,tNodTestInt * aN2,
                     const std::vector<int*>&,
                     const std::vector<int*>&,
                     const std::list<std::pair<int*,int*> >&,
                     int aNewNum
               );

       // Typiquement pour creer les Attibuts
       void OnNewLeaf(tNodTestInt * aSingle);
       void OnNewCandidate(tNodTestInt * aN1);
       void OnNewMerge(tNodTestInt * aN1);

       int mNbTest;
       int mNbGain;
       double mProbaA;
       std::vector<int *> mVTestInt;
       std::vector<int>   mVTestValInt;

};
/*

*/

void cTestIntParamMerge::Vois(int* anIm,std::vector<int *> & aV)
{

    for (int aK=0 ; aK< mNbTest ; aK++)
    {
        if ((aK!=*anIm) && (NRrandom3() < mProbaA))
        {
            aV.push_back(mVTestInt[aK]);
        }
    }
}


double  cTestIntParamMerge::Gain // <0, veut dire on valide pas le noeud
               (
                     tNodTestInt * aN1,tNodTestInt * aN2,
                     const std::vector<int*>&,
                     const std::vector<int*>&,
                     const std::list<std::pair<int*,int*> >& aLPair,
                     int aNewNum
               )
{
/*
    std::cout << "Cdt " << aNewNum << "={" << aN1->Num() << "," << aN2->Num() << "}\n";
*/
    mNbGain ++;
    return   (1.0+ElMax(aN1->Depth(),aN2->Depth()));
}

void cTestIntParamMerge::OnNewLeaf(tNodTestInt * aSingle)
{
   // std::cout << "Creat Feuille " << *(aSingle->Val()) << " " << aSingle->Num() << "\n";
}
void  cTestIntParamMerge::OnNewCandidate(tNodTestInt * aN1)
{
}
void  cTestIntParamMerge::OnNewMerge(tNodTestInt * aN1)
{
/*
    std::cout << aN1->Num() << " MERGE " ;
    for (int aK=0 ; aK<int(aN1->NbFils()) ; aK++)
        std::cout << " " << aN1->FilsK(aK)->Num();
    std::cout << "\n";
*/
}


/*
template class cMergingNode<int,cNoAttr>;
template class cAlgoMergingRec<int,cNoAttr,cTestIntParamMerge>;
template class  ElHeap<cMergingNode<int,cNoAttr> *,cCmpMNode<int,cNoAttr> >;
*/


/*-------------------------------------------------------*/

class cAttrLnkIm
{
};

typedef cMergingNode<cImagH,cAttrLnkIm> tNodIm;

class cParamMerge
{
    public :
       void Vois(cImagH* anIm,std::vector<cImagH *> &);

       double  Gain // <0, veut dire on valide pas le noeud
               (
                     tNodIm * aN1,tNodIm * aN2,
                     const std::vector<cImagH*>&,
                     const std::vector<cImagH*>&,
                     const std::list<std::pair<cImagH*,cImagH*> >&,
                     int aNewNum
               );

       // Typiquement pour creer les Attibuts
       void OnNewLeaf(tNodIm * aSingle);
       void OnNewCandidate(tNodIm * aN1);
       void OnNewMerge(tNodIm * aN1);

};


void cParamMerge::Vois(cImagH* anIm,std::vector<cImagH *> & aV)
{
    const tSetLinks & aLnks = anIm->Lnks();
    for (tSetLinks::const_iterator  itL=aLnks.begin(); itL!=aLnks.end(); itL++)
    {
         aV.push_back(itL->second->Dest());
    }
}

std::string NameNode(tNodIm * aN)
{
    cImagH * anI = aN->Val();
    return anI ? anI->Name() : "XXX" ;
}

double  cParamMerge::Gain // <0, veut dire on valide pas le noeud
               (
                     tNodIm * aN1,tNodIm * aN2,
                     const std::vector<cImagH*>&,
                     const std::vector<cImagH*>&,
                     const std::list<std::pair<cImagH*,cImagH*> >& aLPair,
                     int aNewNum
               )
{
    double aRes = 0.0;
    for 
    (
          std::list<std::pair<cImagH*,cImagH*> >::const_iterator itL=aLPair.begin();
          itL!=aLPair.end();
          itL++
    )
    {
        cImagH* aIm1 =  itL->first;
        cImagH* aIm2 =  itL->second;
        cLink2Img *  aLnk12 = aIm1->GetLinkOfImage(aIm2);
        cLink2Img *  aLnk21 = aIm2->GetLinkOfImage(aIm1);

        aRes += aLnk12->NbPts();
        aRes += aLnk21->NbPts();
    }
    int aDepth = 1 + ElMax(aN1->Depth(),aN2->Depth());
    aRes = aRes / pow(1.3,aDepth);

    std::cout <<  aNewNum 
              << "  Candidate " << aN1->Num() << " " << aN2->Num() << " " << aRes
               << " " << NameNode(aN1) << " " << NameNode(aN2) << " " 
              << "\n";
    return aRes;
}

void cParamMerge::OnNewLeaf(tNodIm * aSingle)
{
   std::cout << "Creat Feuille " << aSingle->Val()->Name() << " " << aSingle->Num() << "\n";
}
void  cParamMerge::OnNewCandidate(tNodIm * aN1)
{
}
void  cParamMerge::OnNewMerge(tNodIm * aN1)
{
    std::cout << aN1->Num() << " MERGE " ;
    for (int aK=0 ; aK<int(aN1->NbFils()) ; aK++)
        std::cout << " " << aN1->FilsK(aK)->Num();
    std::cout << "\n";
}



/*
template class cMergingNode<cImagH,cAttrLnkIm>;
template class cAlgoMergingRec<cImagH,cAttrLnkIm,cParamMerge>;
template class  ElHeap<cMergingNode<cImagH,cAttrLnkIm> *,cCmpMNode<cImagH,cAttrLnkIm> >;
*/





/*************************************************/
/*                                               */
/*                  cImagH                       */
/*                                               */
/*************************************************/


bool cImagH::ComputeLnkHom(cLink2Img & aLnk)
{
   // const ElPackHomologue aPack=ElPackHomologue::FromFile(mAppli.Dir()+aLnk.NameH());
   const ElPackHomologue & aPack=    aLnk.Pack() ; //   ElPackHomologue::FromFile(mAppli.Dir()+aLnk.NameH());
   int aNbPts = aPack.size();
   aLnk.NbPts() = aPack.size();

   if (aNbPts < mAppli.MinNbPtH())
      return false;

   bool Ok;
   double aQual;
   cElHomographie  aHom12 = cElHomographie::RobustInit(&aQual,aPack,Ok,NB_RANSAC_H,90.0,1000);
   // mLnk.push_back(cLink2Img(anI2,aNameH));

   if (!Ok)
     return false;

   aLnk.Hom12() = aHom12;
   aLnk.Qual() = aQual;

   mSomQual += ElMin(aQual,mAppli.SeuilQual()) * aNbPts;
   mSomNbPts += aNbPts;

   return true;
}

void cImagH::ComputeLnkHom()
{
    tSetLinks  aNewL;
    for ( tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
    {
        if (ComputeLnkHom(*(itL->second)))
           aNewL[itL->first] = itL->second ;
        else
           delete itL->second;
    }
    mLnks = aNewL;
    aNewL.clear();

    if (mSomNbPts)
    {
       mSomQual /= mSomNbPts;
       double aSeuilQual = mSomQual*mAppli.RatioQualMoy();
       for (tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL++)
       {
           bool Ok = itL->second->Qual() < aSeuilQual;
           if (Ok)
           {
              aNewL[itL->first] = itL->second ;
           }
           else
           {
               delete itL->second;
           }
           if (Ok)
           {
              std::cout
                     << "IMS " << mName << " " << itL->second->Dest()->Name()
                     << " QUAL " << itL->second->Qual() << " NB " << itL->second->NbPts()
                     << (Ok ? " " : "  ******")
                     <<  "\n";
           }
       }
    }
    mLnks = aNewL;

    std::cout << mName << " QMOY " << (mSomNbPts ? mSomQual : 1e10) << "\n\n";
}


void cImagH::VoisinsNonMarques(const std::vector<cImagH*> & aIn,std::vector<cImagH*> & aVois,int aFlagN,int aFlagT )
{
   aVois.clear();

   for (int aKS=0 ; aKS<int(aIn.size()) ; aKS++)
   {
       cImagH * aIK1 = aIn[aKS];
       for (tSetLinks::iterator itL1 = aIK1->mLnks.begin(); itL1 != aIK1->mLnks.end(); itL1++)
       {
            cImagH * aImTest  = itL1->second->Dest();
            if ((! aImTest->Marqued(aFlagN)) && (! aImTest->Marqued(aFlagT)))
            {
                aImTest->SetMarqued(aFlagT);
                aVois.push_back(aImTest);
            }
        }
    }
    for (int aKT=0 ; aKT<int(aVois.size()) ; aKT++)
        aVois[aKT]->SetUnMarqued(aFlagT);
}

void cImagH::VoisinsMarques(std::vector<cLink2Img*> & aVois,int aFlagN)
{
    aVois.clear();
    for ( tSetLinks::iterator itL = mLnks.begin(); itL != mLnks.end(); itL ++)
    {
        cImagH * aI2  = itL->second->Dest();
        if (aI2->Marqued(aFlagN))
            aVois.push_back(itL->second);
     }
}




void  cAppliReduc::QuadrReestimFromVois(std::vector<cImagH*> & aVLocIm,int aFlag)
{

    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
         cImagH * anI =  mIms[aK];
         anI->HF()->SetModeCtrl(anI->Marqued(aFlag) ? cNameSpaceEqF::eHomLibre : cNameSpaceEqF::eHomFigee);
         // anI->HF()->SetModeCtrl( cNameSpaceEqF::eHomFigee);
    }
    aVLocIm[0]->HF()->SetModeCtrl(cNameSpaceEqF::eHomFigee);
    for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
    {
         cImagH * anI = aVLocIm[aK];
         anI->HF()->ReinitHom(anI->Hi2t());
    }
         // anI->HF()->SetModeCtrl( cNameSpaceEqF::eHomFigee);
    

    for (int aFois = 0 ; aFois<5 ; aFois++)
    {
         for (int aK=0 ; aK<int(mIms.size()) ; aK++)
         {
              cImagH * anI =  mIms[aK];
              mSetEq.AddContrainte(anI->HF()->StdContraintes(),true);
         }
         mSetEq.SetPhaseEquation();

         double aSomR=0;
         double aSomP=0;
         for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
         {
             cImagH * anI1 =  aVLocIm[aK];
             const tSetLinks & aLL = anI1->Lnks();
             for (tSetLinks::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
             {
                 cImagH * anI2 = itL->second->Dest();
                 if (anI2->Marqued(aFlag)) 
                 {
                     cElHomographie aH12 = itL->second->Hom12();
                     const std::vector<Pt3dr> &  anEchP1 = itL->second->EchantP1();
                     cEqHomogFormelle * anEqF = itL->second->EqHF() ;
                     for (int aK=0 ; aK<int(anEchP1.size()) ; aK++)
                     {
                         const Pt3dr & aQ3 = anEchP1[aK];
                         double aPds = aQ3.z;
                         Pt2dr aP1 (aQ3.x,aQ3.y);
                         Pt2dr aP2  = aH12.Direct(aP1);
                         Pt2dr aResidu = anEqF->StdAddLiaisonP1P2(aP1,aP2,aPds,false);
                         aSomR+=square_euclid(aResidu) * aPds;
                         aSomP+= aPds;
                     }
                 }
             }
         }
         std::cout << "RES "  <<  sqrt(aSomR/aSomP)  <<  "\n";
         mSetEq.SolveResetUpdate();
    }
    std::cout << "\n";
    for (int aK=0 ; aK<int(aVLocIm.size()) ; aK++)
    {
        cImagH * anI =  aVLocIm[aK];
        anI->Hi2t() =  anI->HF()->HomCur();
    }
}
/*
*/


void cAppliReduc::TestMerge()
{

    int aCpt=0;
    for (;;)
    {
         for (int aNbTest = 100 ; aNbTest <200; aNbTest++)
         {
             ElTimer aChrono;

             cSimulAero  aParam(Pt2di(aNbTest,aNbTest),Pt2dr(80,80),1.0);
             cAlgoMergingRec<cSomSA,cNoAttr,cSimulAero> anAlgo(aParam.mTabS,aParam,1);
             
             std::cout << aCpt << " " << aNbTest  << " T " << aChrono.uval() << "\n";
         }
         aCpt++;
    }


/*
    int aCpt=0;
    for (;;)
    {
         for (int aNbTest = 200 ; aNbTest <1000; aNbTest++)
         {
             double aRP = NRrandom3();
             double aProba = (5.0 / aNbTest) * aRP;
             aProba=1.0;

             cTestIntParamMerge aParam(aNbTest,aProba);

             cAlgoMergingRec<int,cNoAttr,cTestIntParamMerge> anAlgo(aParam.mVTestInt,aParam,1);
             
             std::cout << aCpt << " " << aNbTest << " Prob " << aRP << "\n";
             // getchar();
         }
         aCpt++;
    }
*/

/*
    std::cout << "TEST MERGE \n";
    // cAlgoMergingRec<cImagH,cAttrLnkIm,cParamMerge> anAlgo(mIms,aParam);
    for (int aK=0 ; aK< aNbTest ; aK++)
        aVTestValInt.push_back(aK);
    for (int aK=0 ; aK< aNbTest ; aK++)
    {
       aVTestInt.push_back(&(aVTestValInt[aK]));
    }
    cAlgoMergingRec<int,cAttrLnkIm,cParamMerge> anAlgo(aVTestInt,aParam);
    std::cout << "TEST MERGED \n";
*/


  
}


/*
*/
/*
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
