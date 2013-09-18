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

#include "Apero.h"
#include "im_tpl/algo_filter_exp.h"

namespace NS_ParamApero
{

//  Lorsque l'on veut ponderer des observation ponctuelle dans le plan, 
//  si elles tombe toute au meme endroit, chacune doit avoir un poid proportionnel
// a l'inverse du nombre d'observation;
//  Ici cela est gere avec une certaine incertitude, les observation etant suposee
// etre localisees selon une gaussienne


class cImDePonderH
{
    public :

        cImDePonderH(Pt2di aSz,double aDZ,double aDistAttenRel);  // aDistAttenRel est donnee pur DZ=1
        void Add(cObsLiaisonMultiple *);
        void AddPt(const Pt2dr & aP);

        void Compile();

        // n'a de sens que lorsque c'est un des points ayant ete utilises pour 
        // pondere
        double GetPds(const Pt2dr & aP)
        {
             return  1/mTImP.getproj(ToLocInd(aP));
        }

        Pt2di ToLocInd(const Pt2dr & aP) const {return round_ni(aP/mDZ);}

    private :

        Pt2di                  mSzR1;
        double                 mDZ;
        double                 mDAtt;
        Pt2di                  mSzRed;
        Im2D_REAL4             mImP;
        TIm2D<REAL4,REAL8>     mTImP;
};


cImDePonderH::cImDePonderH
(
    Pt2di aSz,
    double aDZ,
    double aDistAtten
) :
 mSzR1  (aSz),
 mDZ    (aDZ),
 mDAtt  (aDistAtten),
 mSzRed (round_up(Pt2dr(mSzR1)/aDZ)),
 mImP   (mSzRed.x,mSzRed.y,0.0),
 mTImP  (mImP)
{
} 

void cImDePonderH::Compile()
{
    FilterGauss(mImP,mDAtt/mDZ);
}

cImDePonderH * NewImDePonderH(cObsLiaisonMultiple * anOLM,double aDZ,double aDAR)
{
    Pt2di aSz = anOLM->Pose1()->Calib()->SzIm();
    const std::vector<cOnePtsMult *> &  aVPM = anOLM->VPMul();
    int aNbPts = aVPM.size();

    cImDePonderH * aRes = new cImDePonderH(aSz,aDZ,aDAR * std::sqrt((double)(aSz.x*aSz.y)/aNbPts));


   
    for (int aKPt=0 ; aKPt<aNbPts ;aKPt++)
    {
        aRes->AddPt(aVPM[aKPt]->P0());
    }
    aRes->Compile();
    return aRes;
}

void cImDePonderH::AddPt(const Pt2dr & aP)
{
  mTImP.incr(ToLocInd(aP),1.0);
}

   // ==================================================================================
   // ==================================================================================


class cCmpImOnGainHom
{
    public :

       bool operator()(cPoseCam* aPC1,cPoseCam * aPC2)
       {
             return aPC1->MMNbPts()*aPC1->MMGainAng()  > aPC2->MMNbPts()*aPC2->MMGainAng();
       }
};

class cCmpImOnAngle
{
    public :

       bool operator()(cPoseCam* aPC1,cPoseCam * aPC2)
       {
             return aPC1->MMAngle() < aPC2->MMAngle();
       }
};



// A Angle  , O Optimum
//  La formule est faite pour que
//   *  on ait un gain proportionnel a l'angle en 0
//   *  un maximum pour Opt avec une derivee nulle
//   *

double GainAngle(double A,double Opt) 
{
   if (A <= Opt)
      return 2*A*Opt - A*A;

   return (Opt * Opt) / (1+ 2*pow(A/Opt-1,2));
}

void  cAppliApero::ExportImSecMM(const cChoixImMM & aCIM,cPoseCam* aPC0)
{


    cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale (aCIM.IdBdl(),aPC0->Name());
    const std::vector<cOnePtsMult *> &  aVPM = anOLM->VPMul();
    Pt3dr  aPt0 = anOLM->CentreNuage();

    const CamStenope & aCS0 = *(aPC0->CurCam());

    aPC0->MMDir() =  vunit(aCS0.PseudoOpticalCenter()  - aPt0);
    Pt3dr aDir0 =  aPC0->MMDir();
 

     // On remet a zero les gains
    for(int aKP=0 ; aKP<int(mVecPose.size()) ;aKP++)
    {
       cPoseCam* aPC2 = mVecPose[aKP];
       aPC2->MMNbPts() =0;
    }

     // On compte le nombre de points de liaisons
    for (int aKPt=0 ; aKPt<int(aVPM.size()) ;aKPt++)
    {
        cOnePtsMult & aPMul = *(aVPM[aKPt]);
        if (aPMul.MemPds() >0)
        {
           cOneCombinMult * anOCM = aPMul.OCM();
           const std::vector<cPoseCam *> & aVP = anOCM->VP();
           for (int aKPos=1 ; aKPos<int(aVP.size()) ;aKPos++)
           {
               aVP[aKPos]->MMNbPts()++;
           }
        }
    }
    aPC0->MMNbPts() = 0;



ELISE_ASSERT(false,"METTRE AU MOINS NbMaxPresel \n");
ELISE_ASSERT(false,"METTRE AU MOINS NbMaxPresel \n");
ELISE_ASSERT(false,"METTRE AU MOINS NbMaxPresel \n");
ELISE_ASSERT(false,"METTRE AU MOINS NbMaxPresel \n");


    std::vector<cPoseCam*> aVPPres;

    for(int aKP=0 ; aKP<int(mVecPose.size()) ;aKP++)
    {
       cPoseCam* aPC2     = mVecPose[aKP];
       if ((aPC2 != aPC0) && (aPC2->MMNbPts()>aCIM.NbMinPtsHom().Val()))
       {
           aPC2->MMDir() = vunit(aPC2->CurCam()->PseudoOpticalCenter()  - aPt0);
           double anAngle = euclid(aDir0-aPC2->MMDir());
           aPC2->MMAngle() = anAngle;
           aPC2->MMGainAng() =  GainAngle(anAngle,aCIM.TetaOpt().Val());
           aVPPres.push_back(aPC2);
       }
    }


    // On reduit au nombre Max de Presel
    cCmpImOnGainHom aCmpGH;
    std::sort(aVPPres.begin(),aVPPres.end(),aCmpGH);
    while (int(aVPPres.size()) > aCIM.NbMaxPresel().Val())
          aVPPres.pop_back();


    // On supprime les angles trop fort en gardant au NbMin
    cCmpImOnAngle   aCmpAngle;
    std::sort(aVPPres.begin(),aVPPres.end(),aCmpAngle);
    while (
               (int(aVPPres.size()) > aCIM.NbMinPresel().Val()) && 
               (aVPPres.back()->MMAngle() > aCIM.TetaMaxPreSel().Val())
          )
          aVPPres.pop_back();


    cImDePonderH * aImPond = NewImDePonderH (anOLM,10,1.0);

    std::cout << aPC0->Name() << "\n";
    for(int aKP=0 ; aKP<int(aVPPres.size()) ;aKP++)
    {
       aVPPres[aKP]->MMSelected() = false;
       std::cout << "      "   <<  aVPPres[aKP]->Name() << " " << aVPPres[aKP]->MMAngle() << " " << aVPPres[aKP]->MMNbPts() << "\n";
    }
    std::cout << "=================================================\n";




#if(0)
    while (! Ok)
    {
        for(int aKP1=0 ; aKP1<int(mVecPose.size()) ;aKP1++)
        {
            if (! mVecPose[aKPos]->MMSelected())
            {
                // double aG = ;
            }
            //    mVecPose[aKPos]->MMGain() = 0.0;
        }
    }




    bool Ok = false;

    while (! Ok)
    {
        for(int aKPos=0 ; aKPos<int(mVecPose.size()) ;aKPos++)
        {
             if (! mVecPose[aKPos]->MMSelected()) 
                mVecPose[aKPos]->MMGain() = 0.0;
        }

        int aNbNN=0;

        for(int aKPt=0 ; aKPt<int(aVPM.size()) ;aKPt++)
        {
             cOnePtsMult & aPMul = *(aVPM[aKPt]);
             if(aPMul.MemPds() >0)
             {
                  aNbNN ++;
/*
                  cOneCombinMult * anOCM = aPMul->OCM();
                  const std::vector<cPoseCam *> & aVP = anOCM->VP();
                  for(int aKPos=1 ; aKPos<int(mVecPose.size()) ;aKPos++)
                  {
                      cPoseCam * aPC = aVP[aKPos];
                      if (! aPC->MMSelected())
                      {
                      }
                  }
*/
             }
        }

        ELISE_ASSERT(aNbNN>0,"ExportImSecMM : probably no iteration !!");
 
    }
#endif

    delete aImPond;
    std::cout << "OLM " << aVPM.size() << "\n";
}

void cAppliApero::ExportImMM(const cChoixImMM & aCIM)
{
    cSetName *  aSelector = mICNM->KeyOrPatSelector(aCIM.PatternSel());

    for(int aKP=0 ; aKP<int(mVecPose.size()) ;aKP++)
    {
       cPoseCam* aPC = mVecPose[aKP];
       if (aSelector->IsSetIn(aPC->Name()))
       {
           ExportImSecMM(aCIM,aPC);
       }
    }
}



};


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
