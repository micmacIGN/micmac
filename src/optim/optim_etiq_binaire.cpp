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
class cOptimLabelBinaire 
{
    public :

        // Les couts sont entre 0 et 1
        cOptimLabelBinaire(Pt2di aSz,double aDefCost,double aRegul);

        static cOptimLabelBinaire * CoxRoy(Pt2di aSz,double aDefCost,double aRegul);
        static cOptimLabelBinaire * ProgDyn(Pt2di aSz,double aDefCost,double aRegul); // 2 Do


        // 0.0 privilégie l'état 0 ; 1.0 privilégie l'état 1 ....
        void SetCost(Pt2di aP,double aCost);

        virtual Im2D_Bits<1> Sol() = 0;
        virtual ~cOptimLabelBinaire();
        
    protected :
        static U_INT1 ToCost(double aCost);

        Pt2di              mSz;
        Im2D_U_INT1        mCost;  // Memorise les couts entre 0 et 1
        TIm2D<U_INT1,INT>  mTCost;  // Memorise les couts entre 0 et 1
        double             mRegul;
        
        
};
*/



/*******************************************************************/
/*                                                                 */
/*                  cCoxRoyOLB                                     */
/*                                                                 */
/*******************************************************************/

class cCoxRoyOLB : public cOptimLabelBinaire 
{
    public :
        cCoxRoyOLB(Pt2di aSz,double aDefCost,double aRegul);
        Im2D_Bits<1> Sol();
};


cCoxRoyOLB::cCoxRoyOLB(Pt2di aSz,double aDefCost,double aRegul) :
   cOptimLabelBinaire(aSz,aDefCost,aRegul)
{
}



Im2D_Bits<1> cCoxRoyOLB::Sol()
{
    Im2D_Bits<1> aRes(mSz.x,mSz.y);

    Im2D_INT2 aIZMin(mSz.x,mSz.y,0);
    Im2D_INT2 aIZMax(mSz.x,mSz.y,3);

    cInterfaceCoxRoyAlgo * aCox = cInterfaceCoxRoyAlgo::NewOne
                                     (
                                         mSz.x,
                                         mSz.y,
                                         aIZMin.data(),
                                         aIZMax.data(),
                                         true,  // Conx8
                                         false  // UChar
                                     );

     double aMul = 20.0;
     Pt2di aP;
     for (aP.x = 0 ;  aP.x < mSz.x ; aP.x++)
     {
         for (aP.y = 0 ;  aP.y < mSz.y ; aP.y++)
         {
                  // La couche 0 a un cout de 0.5
                  // La couche 1 un cout de 1-C (donc cost 0 privilegie 0, comme il se doit)
                  // La couche 2 est juste necessaire au bon fonctionnement ...
                  aCox->SetCostVert(aP.x,aP.y,0,aMul*0.5);
                  double aCost = 1- mTCost.get(aP) /255.0;
                  aCox->SetCostVert(aP.x,aP.y,1,round_ni(aCost*aMul));
                  aCox->SetCostVert(aP.x,aP.y,2,round_ni(aMul*2));
         }
     }

    aCox->SetStdCostRegul(0,aMul*mRegul,0);

    Im2D_INT2 aISol(mSz.x,mSz.y);
    aCox->TopMaxFlowStd(aISol.data());

    ELISE_COPY(aRes.all_pts(),aISol.in()!=0,aRes.out());


    delete aCox;

    return aRes;
}
   


/*******************************************************************/
/*                                                                 */
/*                  cProgDOLB                                      */
/*                                                                 */
/*******************************************************************/

// L'attribut auxilaire qui sera memorise dans les nappes completes
class cNoValPrgD
{
};

// L'attribut temporaire qui sera memorise le temps d'un balayage
class cNoValPrgDTmp
{
     public :
        void InitTmp(const cTplCelNapPrgDyn<cNoValPrgD> &) // S'initialise a partir de la cellue de la napp3d
        {
        }
};


class cProgDOLB : public cOptimLabelBinaire 
{

    public :
     // Ce qui est necessaire pour cProg2DOptimiser<cProgDOLB>
         typedef cNoValPrgDTmp            tArgCelTmp;
         typedef cNoValPrgD            tArgNappe;
         typedef cProgDOLB             tArgGlob;  // Afin de "se connaitre" dans le calcul

         
      // Pas pre-requis mais aide
         typedef  cTplCelNapPrgDyn<tArgNappe>    tCelNap;
         typedef  cTplCelOptProgDyn<tArgCelTmp>  tCelOpt;
      // Call back pre requis dans Optim
         void DoConnexion
              (
                 const Pt2di & aPIn, const Pt2di & aPOut,
                 ePrgSens aSens,int aRab,int aMul,
                 tCelOpt*Input,int aInZMin,int aInZMax,
                 tCelOpt*Ouput,int aOutZMin,int aOutZMax
             );
         void GlobInitDir(cProg2DOptimiser<cProgDOLB> &);

    public :
        cProgDOLB(Pt2di aSz,double aDefCost,double aRegul);
        Im2D_Bits<1> Sol();
    private :
        // les couts sont signe <0 => label 0, >0 => label 1; 
        // Le fait de les "redynamiser" fait que la valeur entre 0 et 1 n'est plus vraiment possible
        Im2D_REAL4 ImCost( Im2D_REAL4 aCostIn,int aNbDir,double aTeta0);
        

        static const double mMul;
        int mICostRegul;

};

const double cProgDOLB::mMul = 1000.0;

cProgDOLB::cProgDOLB(Pt2di aSz,double aDefCost,double aRegul) :
   cOptimLabelBinaire  (aSz,aDefCost,aRegul),
   mICostRegul         (round_ni(aRegul*mMul))
{
}

void cProgDOLB::GlobInitDir(cProg2DOptimiser<cProgDOLB> & aPrg2D)
{
}

void cProgDOLB::DoConnexion
     (
                 const Pt2di & aPIn, const Pt2di & aPOut,
                 ePrgSens aSens,int aRab,int aMul,
                 tCelOpt*Input,int aInZMin,int aInZMax,
                 tCelOpt*Ouput,int aOutZMin,int aOutZMax
     )
{
    for (int aZIn=aInZMin ; aZIn<aInZMax ; aZIn++)
    {
        for (int aZOut=aOutZMin ; aZOut<aOutZMax ; aZOut++)
        {
             Ouput[aZOut].UpdateCostOneArc(Input[aZIn],aSens,mICostRegul*ElAbs(aZIn-aZOut));
        }
    }
}


Im2D_REAL4 cProgDOLB::ImCost(Im2D_REAL4 aCostIn,int aNbDir,double aTeta0)
{


   TIm2D<REAL4,REAL8>  aTCostIn(aCostIn);
   Im2D_INT2           mZMin(mSz.x,mSz.y,0);
   Im2D_INT2           mZMax(mSz.x,mSz.y,2);
   cProg2DOptimiser<cProgDOLB> mOpt(*this,mZMin,mZMax,0,1);

   
   // On transfere les cout dans la nappe ; avec les convention A- qu'un des deux couts est nul
   // B- que les cout <0 signifie que l'on privilegie 0

   tCelNap *** aNaps = mOpt.Nappe().Data();
   Pt2di aP;
   double aSigmaIn=0;
   for (aP.x=0; aP.x<mSz.x; aP.x++)
   {
       for (aP.y=0; aP.y<mSz.y; aP.y++)
       {
           double aCost = aTCostIn.get(aP);
           aSigmaIn += ElAbs(aCost);
           int IndCost = (aCost<0) ; // Si cout negatif, c'est la couche 1 qui est penalisee
           aNaps[aP.y][aP.x][IndCost].SetOwnCost(round_ni(ElAbs(aCost)*mMul));
           aNaps[aP.y][aP.x][1-IndCost].SetOwnCost(0);
       }
   }

   // On lance l'optimisation
   mOpt.SetTeta0(aTeta0);
   mOpt.DoOptim(aNbDir);


   // On transfere les couts, en calculant la dynamique

   Im2D_REAL4 aRes(mSz.x,mSz.y);
   TIm2D<REAL4,REAL8> aTRes(aRes);
   double aSigmaOut=0;
   for (aP.x=0; aP.x<mSz.x; aP.x++)
   {
       for (aP.y=0; aP.y<mSz.y; aP.y++)
       {
           double aCost0 = aNaps[aP.y][aP.x][0].CostAgrege();
           double aCost1 = aNaps[aP.y][aP.x][1].CostAgrege();
           double aDelta = (aCost0 - aCost1) / mMul;
           aTRes.oset(aP,aDelta);
           aSigmaOut += ElAbs(aDelta);
       }
   }
   double aRatio = aSigmaIn/aSigmaOut;

   ELISE_COPY(aRes.all_pts(),aRes.in()*aRatio,aRes.out());

   return aRes;
}

Im2D_Bits<1> cProgDOLB::Sol()
{


    Im2D_REAL4 aImCostIn(mSz.x,mSz.y);
    ELISE_COPY(aImCostIn.all_pts(),(mCost.in()-128.0)/255.0,aImCostIn.out());

    for (int aTimes=0 ; aTimes<1 ; aTimes++)
    {
          aImCostIn = ImCost(aImCostIn,4,0.0);
    }

    Im2D_Bits<1>  aRes(mSz.x,mSz.y);
    ELISE_COPY(aRes.all_pts(),aImCostIn.in()>0,aRes.out());

    return aRes;
}


/*******************************************************************/
/*                                                                 */
/*                  cOptimLabelBinaire                             */
/*                                                                 */
/*******************************************************************/

cOptimLabelBinaire::cOptimLabelBinaire(Pt2di aSz,double aDefCost,double aRegul) :
      mSz    (aSz),
      mCost  (aSz.x,aSz.y,ToCost(aDefCost)) ,
      mTCost (mCost),
      mRegul (aRegul)
{
}

U_INT1 cOptimLabelBinaire::ToCost(double aCost)
{
    return ElMax(0,ElMin(255,round_ni(255*aCost)));
}

void cOptimLabelBinaire::SetCost(Pt2di aP,double aCost)
{
// std::cout << "COOOOst " << aCost    << " " << int << " " << mSz << "\n";
    mTCost.oset(aP,ToCost(aCost));
}

cOptimLabelBinaire::~cOptimLabelBinaire()
{
}

cOptimLabelBinaire * cOptimLabelBinaire::CoxRoy(Pt2di aSz,double aDefCost,double aRegul)
{
    return new cCoxRoyOLB(aSz,aDefCost,aRegul);
}

cOptimLabelBinaire *  cOptimLabelBinaire::ProgDyn(Pt2di aSz,double aDefCost,double aRegul)
{
   return new cProgDOLB(aSz,aDefCost,aRegul);
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
