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

class cSubset
{
    public :
         cSubset(const std::vector<cPoseCam*> & aVPres,const std::vector<int> & aVInd)
         {
            for (int aK=0 ; aK<int(aVInd.size()) ; aK++)
               mPoses.push_back(aVPres.at(aVInd[aK]));
         }

         double                 mGainPond;
         double                 mCouvAng;
         std::vector<cPoseCam*> mPoses;
};

class cCmpSubset
{
    public :

       bool operator()(const cSubset   & aP1,const cSubset   & aP2)
       {
             return aP1.mGainPond > aP2.mGainPond;
       }
};


// A Angle  , O Optimum
//  La formule est faite pour que
//   *  on ait un gain proportionnel a l'angle en 0
//   *  un maximum pour Opt avec une derivee nulle
//   *

double GainSup(double aRatio)
{
   return  1 / (1+ 2*pow(aRatio-1,3));
}

double GainAngle(double A,double Opt)
{
   A /= Opt;
   if (A <= 1)
      return pow(ElAbs(2*A - A*A),1.0);

   return  1 / (1+ 2*pow(ElAbs(A-1),3));
}


// Un autre critere de qualite du couple est que la base soit _| aux dir de visee
// C'est un critere complementair du B/H tradi , il evite les "stereo" en profondeurs
// qui sont mauvaises et peut generer des epipolaire degenerees


double OrthogBase(cPoseCam* aPC1,cPoseCam* aPC2)
{
    const CamStenope & aCS1 = *(aPC1->CurCam());
    const CamStenope & aCS2 = *(aPC2->CurCam());

    Pt3dr aB12 = aCS2.PseudoOpticalCenter() -aCS1.PseudoOpticalCenter();
    double aD = euclid(aB12);
    if (aD<=0) return 1;

    aB12 = aB12 / aD;

    double aS1 = ElAbs(scal(aB12,aCS1.DirK()));
    double aS2 = ElAbs(scal(aB12,aCS2.DirK()));

     return ElMax(aS1,aS2);
    
}



void  cAppliApero::ExportImSecMM(const cChoixImMM & aCIM,cPoseCam* aPC0)
{

    std::cout << "ExportImSecMM " << aPC0->Name() << "\n";
    cImSecOfMaster aISM;
    aISM.ISOM_AllVois().SetVal(cISOM_AllVois());
    cISOM_AllVois &  aILV = aISM.ISOM_AllVois().Val();

    cObsLiaisonMultiple * anOLM = PackMulOfIndAndNale (aCIM.IdBdl(),aPC0->Name());
    const std::vector<cOnePtsMult *> &  aVPM = anOLM->VPMul();

    //  Point 3D : pt image par moyenne, profondeur par médiane, devrait etre robuste ?
    Pt3dr  aPt0 = anOLM->CentreNuage();

    const CamStenope & aCS0 = *(aPC0->CurCam());

    aPC0->MMDir() =  vunit(aCS0.PseudoOpticalCenter()  - aPt0);
    Pt3dr aDir0 =  aPC0->MMDir();

    Pt3dr aU,aV;
    MakeRONWith1Vect(aDir0,aU,aV);
    ElMatrix<double>  aMat = MakeMatON(aU,aV);
    ElMatrix<double>  aMatI = gaussj(aMat);
 
	
     // On remet a zero le nombre de points
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


	// Pre selection 
    std::vector<cPoseCam*> aVPPres;

    for(int aKP=0 ; aKP<int(mVecPose.size()) ;aKP++)
    {
       cPoseCam* aPC2     = mVecPose[aKP];
       if ((aPC2 != aPC0) && (aPC2->MMNbPts()>aCIM.NbMinPtsHom().Val()))
       {
           aPC2->MMDir() = vunit(aPC2->CurCam()->PseudoOpticalCenter()  - aPt0);
           Pt3dr aDirLoc  =   aMatI * aPC2->MMDir();
           Pt2dr aDir2D (aDirLoc.x,aDirLoc.y);
           aPC2->MMDir2D() =  vunit(aDir2D);
           double anAngle = euclid(aDir0-aPC2->MMDir());
           aPC2->MMAngle() = anAngle;
           aPC2->MMGainAng() =  GainAngle(anAngle,aCIM.TetaOpt().Val());


           if (
                      (anAngle>aCIM.TetaMinPreSel().Val()) 
                   && (anAngle<aCIM.TetaMaxPreSel().Val())
                   && (OrthogBase(aPC0,aPC2) < 0.7)
              )
           {
               aVPPres.push_back(aPC2);
           }
           cISOM_Vois aV;
           aV.Name() = aPC2->Name();
           aV.Nb() = aPC2->MMNbPts();
           aV.Angle() = anAngle;
           aILV.ISOM_Vois().push_back(aV);
       }
    }



    // On reduit au nombre Max de Presel
    cCmpImOnGainHom aCmpGH;
    std::sort(aVPPres.begin(),aVPPres.end(),aCmpGH);
    while (int(aVPPres.size()) > aCIM.NbMaxPresel().Val())
          aVPPres.pop_back();
	
    int aNbIm = aVPPres.size();

	    
    double aTeta0 = aCIM.Teta2Min().Val();
    double aTeta1 = aCIM.Teta2Max().Val();
    int aNbTeta = 30;
		
    for (int aKI=0 ; aKI<aNbIm ; aKI++)
    {
        cPoseCam * aPC = aVPPres[aKI];
        aPC->MMGainTeta().resize(aNbTeta);
        for (int aKTeta = 0 ; aKTeta<aNbTeta ; aKTeta++)
        {
             Pt2dr aDirT = Pt2dr::FromPolar(1.0,(2*PI*aKTeta) / aNbTeta);
             Pt2dr aDirLoc = aDirT / aPC->MMDir2D();
             double aDifTeta = ElAbs(angle(aDirLoc));
             double aGain = 0.0;
             if (aDifTeta < aTeta0)
                aGain = 1.0;
             else if (aDifTeta < aTeta1)
                  aGain = (aTeta1-aDifTeta) / (aTeta1-aTeta0);
             aPC->MMGainTeta()[aKTeta] = aGain;
        }
    }



	
   if (0)
   {
	    std::cout << aPC0->Name() << "\n";
        for(int aKP=0 ; aKP<int(aVPPres.size()) ;aKP++)
        {
           aVPPres[aKP]->MMSelected() = false;
           std::cout << "      "   <<  aVPPres[aKP]->Name() << " " << aVPPres[aKP]->MMAngle() <<  " " << angle(aVPPres[aKP]->MMDir2D()) << " " << aVPPres[aKP]->MMNbPts() << "\n";
        }
        std::cout << "=================================================\n";
    }
   


    aISM.Master() = aPC0->Name();
    // ON TESTE LES SUBSET 
    for (int aCard=1 ; aCard< ElMin(aNbIm+1, 1+aCIM.CardMaxSub().Val()) ; aCard++)
    {
         // On selectionne a cardinal donne, les subset qui couvrent l'ensemble des directions
         std::vector<std::vector<int> > aSubSub;
         GetSubset(aSubSub,aCard,aNbIm);
         
         double aBestK = -1;
         double aBestGain = -1;
         double aBestCov = -1;
         for (unsigned int aKS=0 ; aKS<aSubSub.size() ; aKS++)
         {
               std::vector<int> & aSub = aSubSub[aKS];
               double aSomGain = 0.0;
               double aSomCouv = 0.0;
               for (int aKTeta = 0 ; aKTeta<aNbTeta ; aKTeta++)
               {
                   //Pt2dr aDirT = Pt2dr::FromPolar(1.0,(2*PI*aKTeta) / aNbTeta);
                   double aGainMax = 0;
                   double aCouvMax = 0;
                   for (int aKE=0 ; aKE<aCard ; aKE++)
                   {
                       cPoseCam * aPC = aVPPres.at(aSub.at(aKE));
                       double aGain  = aPC->MMGainTeta()[aKTeta];
                       ElSetMax(aCouvMax,aGain);
                       aGain *=    pow(aPC->MMNbPts(),0.33333)*aPC->MMGainAng() ;
                       ElSetMax(aGainMax,aGain);
					   
                   }
                   aSomCouv += aCouvMax;
                   aSomGain += aGainMax;
               }
               if (aSomGain>aBestGain)
               {
                    aBestGain = aSomGain;
                    aBestCov = aSomCouv / aNbTeta;
                    aBestK = aKS;
               }
         }

         cOneSolImageSec aSol;
         std::vector<int> & aSub = aSubSub.at( (int)aBestK );
         for (int aKE=0 ; aKE<aCard ; aKE++)
         {
             cPoseCam * aPC = aVPPres.at(aSub.at(aKE));
             aSol.Images().push_back(aPC->Name());
         }
         aSol.Coverage() =  aBestCov;
         aSol.Score() =  aBestCov  / pow(aCard,0.2);
         aISM.Sols().push_back(aSol);
    }
    std::string aName = mDC + mICNM->Assoc1To1(aCIM.KeyAssoc(),aPC0->Name(),true);
    MakeFileXML(aISM,aName);

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
