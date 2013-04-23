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

namespace NS_ParamMICMAC
{

bool IsPTest(const Pt2di & aP) {return aP == Pt2di(40,40);}

Pt2di Px2Point(int * aPx) { return Pt2di(aPx[0],0); }
int CostR2I(double aCost) { return round_ni(aCost*1e4); }


/**************************************************/
/*                                                */
/*       cGBV2_CelOptimProgDyn                         */
/*                                                */
/**************************************************/
typedef unsigned int tCost;

class cGBV2_CelOptimProgDyn
{
public :
    cGBV2_CelOptimProgDyn() :
        mCostFinal (0)
    {
    }
    typedef enum
    {
        eAvant = 0,
        eArriere = 1
    } eSens;
    void SetCostInit(int aCost)
    {
        mCostInit = aCost;
    }
    void SetBeginCumul(eSens aSens)
    {
        mCostCum[aSens] = mCostInit;
    }
    void SetCumulInitial(eSens aSens)
    {
        mCostCum[aSens] = int(1e9) ;
    }
    void UpdateCost
    (
            const cGBV2_CelOptimProgDyn& aCel2,
            eSens aSens,
            int aCostTrans
            )
    {
        ElSetMin
                (
                    mCostCum[aSens],
                    mCostInit + aCostTrans + aCel2.mCostCum[aSens]
                    );
    }

    tCost CostPassageForce() const
    {
        return mCostCum[eAvant] + mCostCum[eArriere] - mCostInit;
    }
    tCost GetCostInit() const
    {
        return mCostInit;
    }
    const tCost & CostFinal() const { return mCostFinal; }
    tCost & CostFinal() { return mCostFinal; }
private :
    cGBV2_CelOptimProgDyn(const cGBV2_CelOptimProgDyn &);
    tCost   mCostCum[2];
    tCost   mCostInit;
    tCost   mCostFinal;
};

typedef  cSmallMatrixOrVar<cGBV2_CelOptimProgDyn>   tCGBV2_tMatrCelPDyn;




/**************************************************/
/*                                                */
/*                     ::                         */
/*                                                */
/**************************************************/


//  COPIE TEL QUEL DEPUIS /home/mpd/ELISE/applis/Anag/SimpleProgDyn.cpp

/*
    Soit Z dans l'intervalle ouvert I1 [aZ1Min,aZ1Max[,
    on recherche dans l'intervalle ouvert I0 [aZ0Min,aZ0Max[,
    un intervalle ferme, non vide, le plus proche possible
    [aZ+aDzMin,aZ+aDzMax].

    De plus ce calcul doit generer des connexion symetrique.

    Ex  :
        I1 = [10,30[
        I0 = [5,20[

        MaxDeltaZ = 2


        Z = 13 ->    Delta = [-2,2]   // aucune contrainte
        Z = 18 ->    Delta = [-2,1]   // Pour que ca reste dans I0
        Z = 25 ->    Delta = [-6,-6]  //  Pour que l'intersection soit non vide avec I0
        Z = 10 ->    Delta = [-5,-1]  // principe de symetrie, dans l'autre sens                                      // les points [5,9] de I0 devront etre connecte a 10

*/

/// brief Calcul le Z min et max.
static inline void ComputeIntervaleDelta
(
        INT & aDzMin,
        INT & aDzMax,
        INT aZ,
        INT MaxDeltaZ,
        INT aZ1Min,
        INT aZ1Max,
        INT aZ0Min,
        INT aZ0Max
        )
{
    aDzMin =   aZ0Min-aZ;
    if (aZ != aZ1Min)
        ElSetMax(aDzMin,-MaxDeltaZ);

    aDzMax = aZ0Max-1-aZ;
    if (aZ != aZ1Max-1)
        ElSetMin(aDzMax,MaxDeltaZ);

    // Si les intervalles sont vides, on relie
    // les bornes des intervalles a tous les points
    if (aDzMin > aDzMax)
    {
        if (aDzMax <0)
            aDzMin = aDzMax;
        else
            aDzMax = aDzMin;
    }
}

/**************************************************/
/*                                                */
/*               cProgDynOptimiseur               */
/*                                                */
/**************************************************/


class cGBV2_TabulCost
{
public :
    void Reset(double aCostL1,double aCostL2)
    {
        mNb=0;
        mTabul = std::vector<int>();
        mCostL1 = aCostL1;
        mCostL2 = aCostL2;
    }
    inline int Cost(int aDx)
    {
        aDx = ElAbs(aDx);
        for (;mNb<=aDx; mNb++)
        {
            mTabul.push_back
                    (
                        CostR2I
                        (
                            mCostL1 * mNb
                            + mCostL2 * ElSquare(mNb)
                            )
                        );
        }
        return mTabul[aDx];
    }

    double           mCostL1;
    double           mCostL2;
    int              mNb;
    std::vector<int> mTabul;
};

class cGBV2_ProgDynOptimiseur : public cSurfaceOptimiseur
{
public :
    cGBV2_ProgDynOptimiseur
    (
        cAppliMICMAC &    mAppli,
        cLoadTer&         mLT,
        const cEquiv1D &        anEqX,
        const cEquiv1D &        anEqY,
            Im2D_INT2  aPxMin,
            Im2D_INT2  aPxMax
            );
    ~cGBV2_ProgDynOptimiseur() {}
    void Local_SetCout(Pt2di aPTer,int *aPX,REAL aCost,int aLabel);
    void Local_SolveOpt(Im2D_U_INT1 aImCor);


    // Im2D_INT2     ImRes() {return mImRes;}

private :

    void BalayageOneDirection(Pt2dr aDir);
    void BalayageOneLine(const std::vector<Pt2di> & aVPt);
    void BalayageOneLineGpu(const std::vector<Pt2di> & aVPt);
    void BalayageOneSens
    (
            const std::vector<Pt2di> & aVPt,
            cGBV2_CelOptimProgDyn::eSens,
            int anIndInit,
            int aDelta,
            int aLimite
            );

    void BalayageOneSensGpu
    (
            const std::vector<Pt2di> & aVPt,
            cGBV2_CelOptimProgDyn::eSens,
            int anIndInit,
            int aDelta,
            int aLimite
            );
    void SolveOneEtape(int aNbDir);

    Im2D_INT2                          mXMin;
    Im2D_INT2                          mXMax;
    Pt2di                              mSz;
    int                                mNbPx;
    Im2D_INT2                          mYMin;
    Im2D_INT2                          mYMax;
    cMatrOfSMV<cGBV2_CelOptimProgDyn>  mMatrCel;
    cLineMapRect                       mLMR;
    cGBV2_TabulCost                    mTabCost[theDimPxMax];
    // int                             mCostActu[theDimPxMax];
    int                                mMaxEc[theDimPxMax];
    eModeAggregProgDyn                 mModeAgr;
    int                                mNbDir;
    double                             mPdsProgr;

    // Im2D_INT2     mImRes;
    // INT2 **       mDataImRes;

};
static cGBV2_CelOptimProgDyn aCelForInit;

cGBV2_ProgDynOptimiseur::cGBV2_ProgDynOptimiseur
(
        cAppliMICMAC&   mAppli,
        cLoadTer&       mLT,
        const cEquiv1D& anEqX,
        const cEquiv1D& anEqY,
        Im2D_INT2       aPxMin,
        Im2D_INT2       aPxMax
) :
    cSurfaceOptimiseur ( mAppli,mLT,1e4,anEqX,anEqY,false,false),
    mXMin       (aPxMin),
    mXMax       (aPxMax),
    mSz         (mXMin.sz()),
    mNbPx       (1),
    mYMin       (mSz.x,mSz.y,0),
    mYMax       (mSz.x,mSz.y,1),
    mMatrCel    (
                    Box2di(Pt2di(0,0),mSz),
                    mXMin.data(),
                    mYMin.data(),
                    mXMax.data(),
                    mYMax.data(),
                    aCelForInit),
    mLMR        (mSz)
{
}

void cGBV2_ProgDynOptimiseur::Local_SetCout(Pt2di aPTer,int *aPX,REAL aCost,int aLabel)
{
    mMatrCel[aPTer][Px2Point(aPX)].SetCostInit(CostR2I(aCost));
}

void cGBV2_ProgDynOptimiseur::BalayageOneSens
(
        const std::vector<Pt2di> &   aVPt,     // vecteur de points
        cGBV2_CelOptimProgDyn::eSens aSens,    // sens du parcourt
        int                          anIndInit,// premier point
        int                          aDelta,   // delta incremenation de progression
        int                          aLimite   // Limite de progression
        )
{

    // Initialisation des couts sur les premieres valeurs
    {
        // Matrice des cellules
        tCGBV2_tMatrCelPDyn &  aMat0 = mMatrCel[aVPt[anIndInit]];

        // Le rectangle
        const Box2di & aBox0 = aMat0.Box();

        Pt2di aP0;

        std::cout << aBox0.P0() << " " << aBox0.P1() << "\n";
        getchar();

        for (aP0.y = aBox0._p0.y ;  aP0.y<aBox0._p1.y; aP0.y++)

            for (aP0.x = aBox0._p0.x ; aP0.x<aBox0._p1.x;aP0.x++)

                aMat0[aP0].SetBeginCumul(aSens);
    }

    // Propagation
    int anI0 = anIndInit;
    while ((anI0+ aDelta)!= aLimite)
    {
        //
        int anI1 = anI0+aDelta;

        tCGBV2_tMatrCelPDyn &  aMat1 = mMatrCel[aVPt[anI1]];
        const Box2di & aBox1 = aMat1.Box();
        Pt2di aP1;

        // Met un cout infini aux successeurs
        for (aP1.y = aBox1._p0.y ; aP1.y<aBox1._p1.y;aP1.y++)
        {
            for (aP1.x = aBox1._p0.x ; aP1.x<aBox1._p1.x ; aP1.x++)
            {
                aMat1[aP1].SetCumulInitial(aSens);
            }
        }

        // Propagation
        tCGBV2_tMatrCelPDyn &  aMat0 = mMatrCel[aVPt[anI0]];
        const Box2di & aBox0 = aMat0.Box();
        Pt2di aP0;
        for (aP0.y=aBox0._p0.y ; aP0.y<aBox0._p1.y ; aP0.y++)
        {
            int aDyMin,aDyMax;
            // Calcul du delta sur Y
            ComputeIntervaleDelta
                    (
                        aDyMin,
                        aDyMax,
                        aP0.y,
                        mMaxEc[1],
                    aBox0._p0.y,
                    aBox0._p1.y,
                    aBox1._p0.y,
                    aBox1._p1.y
                    );
            for (aP0.x=aBox0._p0.x ;  aP0.x<aBox0._p1.x ; aP0.x++)
            {
                int aDxMin,aDxMax;
                // Calcul du delta sur X
                ComputeIntervaleDelta
                        (
                            aDxMin,
                            aDxMax,
                            aP0.x,
                            mMaxEc[0],
                        aBox0._p0.x,
                        aBox0._p1.x,
                        aBox1._p0.x,
                        aBox1._p1.x
                        );

                // Cellule courante
                cGBV2_CelOptimProgDyn & aCel0 = aMat0[aP0];

                // Parcours des cellules dans l'intervalle des Deltas
                for (int aDy=aDyMin ; aDy<=aDyMax; aDy++)
                {
                    for (int aDx=aDxMin ; aDx<=aDxMax; aDx++)
                    {

                        aMat1[aP0+Pt2di(aDx,aDy)].UpdateCost                       // cellule colonne suivante
                                (
                                    aCel0,                                               // cellule colonne courante
                                    aSens,                                               // Sens de parcours
                                    mTabCost[0].Cost(aDx) + mTabCost[1].Cost(aDy)        // Tabulation des p�nalit�s ou cout de transition
                                );                                                                                 // mCostActu[0]*ElAbs(aDx)+mCostActu[1]*ElAbs(aDy)
                    }
                }
            }
        }
        anI0 = anI1;
    }
}

void cGBV2_ProgDynOptimiseur::BalayageOneSensGpu(const std::vector<Pt2di> &aVPt, cGBV2_CelOptimProgDyn::eSens aSens, int anIndInit, int aDelta, int aLimite)
{
    // Initialisation des couts sur les premieres valeurs -------------------

    // Matrice des cellules
    tCGBV2_tMatrCelPDyn &  aMat0 = mMatrCel[aVPt[anIndInit]];

    // Le rectangle
    const Box2di & aBox0 = aMat0.Box();
    Pt2di aP0;

    for(aP0.x = aBox0._p0.x ; aP0.x<aBox0._p1.x;aP0.x++)
        aMat0[aP0].SetBeginCumul(aSens);

    // Propagation ------------------------------------------------------------
    int anI0 = anIndInit;
    while ((anI0+ aDelta)!= aLimite)
    {
        int anI1 = anI0+aDelta;

        tCGBV2_tMatrCelPDyn& aMat1 = mMatrCel[aVPt[anI1]];
        const Box2di&        aBox1 = aMat1.Box();
        Pt2di aP1;

        // Met un cout infini aux successeurs
        for (aP1.x = aBox1._p0.x ; aP1.x<aBox1._p1.x ; aP1.x++)
            aMat1[aP1].SetCumulInitial(aSens);

        // Propagation
        tCGBV2_tMatrCelPDyn& aMat0 = mMatrCel[aVPt[anI0]];
        const Box2di &       aBox0 = aMat0.Box();
        Pt2di aP0;

        //std::cout << "[" << aBox0._p0.x << "," << aBox0._p1.x << "] ";
        for (aP0.x=aBox0._p0.x ;  aP0.x<aBox0._p1.x ; aP0.x++)
        {
            int aDxMin,aDxMax;
            // Calcul du delta sur X
            ComputeIntervaleDelta(aDxMin,aDxMax,aP0.x, mMaxEc[0],aBox0._p0.x, aBox0._p1.x, aBox1._p0.x,aBox1._p1.x);

            // Cellule courante
            cGBV2_CelOptimProgDyn & aCel0 = aMat0[aP0];

            // Parcours des cellules dans l'intervalle des Deltas
            for (int aDx=aDxMin ; aDx<=aDxMax; aDx++)
                aMat1[aP0+Pt2di(aDx,0)].UpdateCost(aCel0, aSens, mTabCost[0].Cost(aDx));

        }

        anI0 = anI1;
    }
}

void cGBV2_ProgDynOptimiseur::BalayageOneLine(const std::vector<Pt2di> & aVPt)
{
    // 1er Parcour dans un sens
    // aVPt         : ensemble des points
    // eAvant       : Sens de parcours
    // 0            : Premier point
    // 1            : delta incrementation
    // aVPt.size()  : limite du parcours
    BalayageOneSens(aVPt,cGBV2_CelOptimProgDyn::eAvant,0,1,(int) aVPt.size());

    // 2eme Parcour dans un sens inverse
    // aVPt         : ensemble des points
    // eArriere     : Sens de parcours
    // aVPt.size()-1: on part du dernier point
    // -1           : delta incrementation invers�
    // -1           : limite du parcours
    BalayageOneSens(aVPt,cGBV2_CelOptimProgDyn::eArriere,(int) (aVPt.size())-1,-1,-1);

    // on parcours la ligne
    for (int aK=0 ; aK<int(aVPt.size()) ; aK++)
    {
        // Matrice des cellules
        tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aVPt[aK]];
        // rectancle
        const Box2di &  aBox = aMat.Box();
        Pt2di aP;

        // Cout infini
        tCost aCoutMin = tCost(1e9);

        //recherche du cout minimum dans le le rectangle
        for (aP.y = aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)        
            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)          
                ElSetMin(aCoutMin,aMat[aP].CostPassageForce());


        for (aP.y = aBox._p0.y ; aP.y<aBox._p1.y; aP.y++)
        {
            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            {
                tCost  aNewCost = aMat[aP].CostPassageForce()-aCoutMin;
                tCost & aCF = aMat[aP].CostFinal();
                if (mModeAgr==ePrgDAgrSomme) // Mode somme
                {
                    printf("a");
                    aCF += aNewCost;
                }
                else if (mModeAgr==ePrgDAgrMax) // Mode max
                {
                    printf("b");
                    ElSetMax(aCF,aNewCost);
                }
                else if (mModeAgr==ePrgDAgrProgressif) // Mode max
                {
                    printf("c");
                    aCF= aNewCost;
                    aMat[aP].SetCostInit
                            (
                                round_ni
                                (
                                    mPdsProgr*aCF
                                    + (1-mPdsProgr)* aMat[aP].GetCostInit()
                                    )
                                );
                }
                else  // Mode reinjection
                {
                    printf("d");
                    aCF = aNewCost;
                    aMat[aP].SetCostInit(aCF);
                }

            }
        }

    }
}

void cGBV2_ProgDynOptimiseur::BalayageOneLineGpu(const std::vector<Pt2di> &aVPt)
{

    BalayageOneSensGpu(aVPt,cGBV2_CelOptimProgDyn::eAvant,0,1,(int) aVPt.size());
    BalayageOneSensGpu(aVPt,cGBV2_CelOptimProgDyn::eArriere,(int) (aVPt.size())-1,-1,-1);

    // on parcours la ligne
    for (int aK=0 ; aK<int(aVPt.size()) ; aK++)
    {
        // Matrice des cellules
        tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aVPt[aK]];
        // rectancle
        const Box2di &  aBox = aMat.Box();
        Pt2di aP;
        // Cout infini
        tCost aCoutMin = tCost(1e9);

        //recherche du cout minimum dans le le rectangle
        for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            ElSetMin(aCoutMin,aMat[aP].CostPassageForce());

        for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
        {
            tCost  aNewCost = aMat[aP].CostPassageForce()-aCoutMin;
            tCost & aCF = aMat[aP].CostFinal();
            aCF += aNewCost;
        }
    }
}

void cGBV2_ProgDynOptimiseur::BalayageOneDirection(Pt2dr aDirR)
{
    Pt2di aDirI = Pt2di(vunit(aDirR) * 20.0);
    mLMR.Init(aDirI,Pt2di(0,0),mSz);

    const std::vector<Pt2di> * aVPt;



#ifdef CUDA_ENABLED

    uint    profondeur = 32;
    uint3   dimStream = make_uint3(profondeur,mSz.x,mSz.y);

    CuHostData3D<uint>      streamCostVolume(dimStream);
    CuHostData3D<short2>    index(make_uint2(mSz.x,mSz.y),1);
    CuHostData3D<uint>      hOutputValue_AV(dimStream);
    CuHostData3D<uint>      hOutputValue_AR(dimStream);

    streamCostVolume.Fill(10123);
    uint line = 40, x = 0;

    while ((aVPt = mLMR.Next()))
    {   
        int  idStream = 0;

        uint Pit        = x * mSz.x * profondeur ;
        uint lenghtLine = int(aVPt->size());

        //printf("lenghtLine = %d\n",lenghtLine);
        for (uint aK = 0 ; aK < lenghtLine; aK++)
        {
            // Matrice des cellules
            tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[(*aVPt)[aK]];
            const Box2di &  aBox = aMat.Box();

            Pt2di aP;

            short2 Z = make_short2(max(-15,aBox._p0.x),min(16,aBox._p1.x));

            //index[make_uint2(aK,x)] = make_short2(aBox._p0.x,aBox._p1.x);
            index[make_uint2(aK,x)] = Z;

            //for (aP.x = aBox._p0.x ; aP.x < aBox._p1.x ; aP.x++)
            for (aP.x = Z.x ; aP.x < Z.y ; aP.x++,idStream++)
                streamCostVolume[Pit + idStream]  = aMat[aP].GetCostInit();            

        }
        x++;
    }

//    index.OutputValues(0,XY,Rect(0,line,mSz.x,line+1),0,make_short2(0,0));

//    streamCostVolume.OutputValues(line);

    OptimisationOneDirection(streamCostVolume,index,make_uint3(mSz.y,mSz.x,profondeur),hOutputValue_AV,hOutputValue_AR);

//    hOutputValue_AV.OutputValues(line);

    mLMR.Init(aDirI,Pt2di(0,0),mSz);
    x = 0;
    while ((aVPt = mLMR.Next()))
    {
        // on parcours la ligne

        int lenghtLine = int(aVPt->size());

        for (int aK=0 ; aK < lenghtLine ; aK++)
        {
            tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[(*aVPt)[aK]];
            const Box2di &  aBox = aMat.Box();
            Pt2di aP;
            tCost aCoutMin = tCost(1e9);

            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            {
                uint3   Pt_AV       = make_uint3( aP.x, aK,                  x);
                uint3   Pt_AR       = make_uint3( aP.x, lenghtLine - aK - 1, x);
                int     costInit    = aMat[aP].GetCostInit();

                ElSetMin(aCoutMin,hOutputValue_AV[Pt_AV] + hOutputValue_AR[Pt_AR] - costInit);
            }

            for (aP.x = aBox._p0.x ; aP.x<aBox._p1.x ; aP.x++)
            {
                uint3   Pt_AV       = make_uint3( aP.x, aK,                  x);
                uint3   Pt_AR       = make_uint3( aP.x, lenghtLine - aK - 1, x);
                int     costInit    = aMat[aP].GetCostInit();
                tCost   aNewCost    = hOutputValue_AV[Pt_AV] + hOutputValue_AR[Pt_AR] - costInit - aCoutMin;
                tCost & aCF         = aMat[aP].CostFinal();
                aCF                 += aNewCost;
            }
        }
        x++;
    }

    streamCostVolume.Dealloc();
    index.Dealloc();
    hOutputValue_AV.Dealloc();
    hOutputValue_AR.Dealloc();

#else
    //printf("Optimisation CPU\n");
     while ((aVPt=mLMR.Next()))
        BalayageOneLineGpu(*aVPt);
#endif
}

void cGBV2_ProgDynOptimiseur::SolveOneEtape(int aNbDir)
{
    mModeAgr = ePrgDAgrSomme;
    mNbDir = aNbDir;


    for (int aKP=0 ; aKP<theDimPxMax ; aKP++)
    {
        // mCostActu[aKP] =0;
        mTabCost[aKP].Reset(0,0);
    }

    // Parcours dans toutes les directions
    for (int aKDir=0 ; aKDir<mNbDir ; aKDir++)
    {
        mPdsProgr = (1.0+aKDir) / mNbDir;
        double aTeta =   (aKDir*PI)/mNbDir;

        Pt2dr aP = Pt2dr::FromPolar(100.0,aTeta);
        // On le met la parce que en toute rigueur ca depend de la
        // direction, mais pour l'instant on ne gere pas cette dependance
        // Tabulation des couts de transition
        for (int aKP=0 ; aKP<mNbPx ; aKP++)
        {
            mTabCost[aKP].Reset
                    (
                        mCostRegul[aKP],
                        mCostRegul_Quad[aKP]
                        );
        }
        // Balayage dans une direction aP
        BalayageOneDirection(aP);
    }

    {
        Pt2di aPTer;
        for (aPTer.y=0 ; aPTer.y<mSz.y ; aPTer.y++)
        {
            for (aPTer.x=0 ; aPTer.x<mSz.x ; aPTer.x++)
            {
                tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aPTer];
                const Box2di &  aBox = aMat.Box();
                Pt2di aPRX;
                for (aPRX.y=aBox._p0.y ;aPRX.y<aBox._p1.y; aPRX.y++)
                {
                    for (aPRX.x=aBox._p0.x ;aPRX.x<aBox._p1.x; aPRX.x++)
                    {
                        tCost & aCF = aMat[aPRX].CostFinal();
                        if (mModeAgr==ePrgDAgrSomme) // Mode somme
                        {
                            aCF /= mNbDir;
                        }
                        aMat[aPRX].SetCostInit(aCF);
                        aCF = 0;
                    }
                }
            }
        }
    }
}



void cGBV2_ProgDynOptimiseur::Local_SolveOpt(Im2D_U_INT1 aImCor)
{

    // double aVPentes[theDimPxMax];
    const cModulationProgDyn &  aModul = mEtape.EtapeMEC().ModulationProgDyn().Val();

    // std::cout << " ZRrg " << mCostRegul[0] << " Pente " <<  aModul.Px1PenteMax().Val() << "\n";

   double aPenteMax = aModul.Px1PenteMax().Val();
   double aRegul    =  mCostRegul[0];
   double aRegul_Quad = 0.0;
    //=================
    double aVPentes[theDimPxMax];

    mCostRegul[0] = aRegul;
    mCostRegul[1] = 0;

    mCostRegul_Quad[0] = aRegul_Quad;
    mCostRegul_Quad[1] = 0 ;

    aVPentes[0] = aPenteMax;
    aVPentes[1] = 10;


    for (int aKP=0 ; aKP<mNbPx ; aKP++)
    {
        double aPente = aVPentes[aKP]; // MODIF  / mEtape.KPx(aKP).Pas();
        mMaxEc[aKP] = ElMax(1,round_ni(aPente));
    }


     
    for 
    (
        std::list<cEtapeProgDyn>::const_iterator itE=aModul.EtapeProgDyn().begin();
        itE!=aModul.EtapeProgDyn().end();
        itE++
    )
    {
        SolveOneEtape(itE->NbDir().Val());
    }

Im2D_INT4 aDupRes(mSz.x,mSz.y);

    {
        Pt2di aPTer;
        for (aPTer.y=0 ; aPTer.y<mSz.y ; aPTer.y++)
        {
            for (aPTer.x=0 ; aPTer.x<mSz.x ; aPTer.x++)
            {
                tCGBV2_tMatrCelPDyn &  aMat = mMatrCel[aPTer];
                const Box2di &  aBox = aMat.Box();
                Pt2di aPRX;
                Pt2di aPRXMin;
                tCost   aCostMin = tCost(1e9);
                for (aPRX.y=aBox._p0.y ;aPRX.y<aBox._p1.y; aPRX.y++)
                {
                    for (aPRX.x=aBox._p0.x ;aPRX.x<aBox._p1.x; aPRX.x++)
                    {
                        tCost aCost = aMat[aPRX].GetCostInit();
                        if (aCost<aCostMin)
                        {
                            aCostMin = aCost;
                            aPRXMin = aPRX;
                        }
                    }
                }
                // MODIF
                mDataImRes[0][aPTer.y][aPTer.x] = aPRXMin.x ;
aDupRes.data()[aPTer.y][aPTer.x] = aPRXMin.x ;


            }
        }

    }


if (0)
{
    
   Video_Win aW = Video_Win::WStd(mSz,5.0);
   ELISE_COPY(aW.all_pts(),aDupRes.in()*10,aW.ocirc());
getchar();
}
}

cSurfaceOptimiseur * cSurfaceOptimiseur::AllocAlgoTestGPU
                     (
                                     cAppliMICMAC &    mAppli,
                                     cLoadTer&         mLT,
                                     const cEquiv1D &        anEqX,
                                     const cEquiv1D &        anEqY
                     )
{
   return new cGBV2_ProgDynOptimiseur
              (
                  mAppli,mLT,
                  anEqX,anEqY,
                  mLT.KthNap(0).mImPxMin,
                  mLT.KthNap(0).mImPxMax
              );
}


/**************************************************/
/*                                                */
/*               cSurfaceOptimiseur               */
/*                                                */
/**************************************************/


}

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
