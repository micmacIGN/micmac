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


/*
   Ce fichier est l'interface pour la nouvelle implementation dans
   eLiSe des travaux d'Hassan Jibrini.
*/

#ifndef _HASSAN_ARRANGT_H  // general
#define _HASSAN_ARRANGT_H

#include <algorithm>

#define ElHJAEpsilon 1e-5

           // ******* Classe definie dans ce fichier

class cElHJaSomEmpr;  // Un sommet de l'emprise plani
class cElHJaPlan3D;          // Represente un plan support de facette
class cElHJaPoint;  // Represente un point, intersection de 3 cElHJaPlan3D
class cElHJaDroite; // Represente une droite, intersection de 2 cElHJaPlan3D
class cElHJaFacette;         // Facette d'un plan
class cElHJaArrangt_Visu;    // Permet une visualisation
class cElHJaArrangt;         // Classe contenant tous les objets

                    // Pour representer un modele 3D solution
		    // topologie "a la Trapu"
class cTrapuFace;
class cTrapuBat;
    // Anciennes
class cElHJSom3D;     // Fichier cElHJSol3D.cpp
class cElHJFac3D;     // Fichier cElHJSol3D.cpp
class cElHJSol3D;

typedef std::vector<cElHJaSomEmpr>  tEmprPlani;  // Type Emprise planimetrique
typedef std::list<cElHJaFacette *> tBufFacette;

          // ******* Classe d'implementation pure, defini en
          // ******* interne pour ne pas surcharger
          // *****   la compile (instanciation de template)


#include "graphes/graphe.h"

class cElHJaAttrSomPlani;
class cElHJaAttrArcPlani;


typedef ElGraphe<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tGrPl;
typedef ElSom<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tSomGrPl;
typedef ElArc<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tArcGrPl;
typedef ElSomIterator<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tItSomGrPl;
typedef ElArcIterator<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tItArcGrPl;
typedef ElSubGraphe<cElHJaAttrSomPlani,cElHJaAttrArcPlani> tFullSubGrPl;

typedef tSomGrPl * tSomGrPlPtr;


           // **************************************

class cElHJaSomEmpr
{
	public :
           cElHJaSomEmpr(Pt2dr  aPos,const cElHJaSomEmpr * mPrec);
	   Pt2dr Pos() const;
	   REAL  ACurv() const;
	private :
           Pt2dr mPos;
	   REAL  mACurv;
};

class cElHJaPlan3D
{
     public :
         ~cElHJaPlan3D();
         cElHJaPlan3D
         (
	      cElHJaArrangt &             anArrgt,
              INT                         aNum,
              const cElPlan3D &           aPlan,
              const tEmprPlani &          anEmprGlob,
              const std::vector<Pt2dr>&   anEmpriseSpec,
              Video_Win *                 aW  // Eventuelement 0
         );
	 void SetSegOblig(Seg2d aSeg);
         INT    Num() const;
	 const cElPlan3D & Plan() const;
	 void  SetNbPlansInter(INT);

	 void AddInter(cElHJaDroite &,cElHJaPlan3D & AutrPl);
         cElHJaDroite * DroiteOfInter(const cElHJaPlan3D & anOtherPl) const;
	 void Show(Video_Win,INT CoulRaz,bool ShowDroite,bool ShowInterEmpr);

	 void AddArcEmpriseInGraphe();

         tSomGrPl *  SomGr3Pl(cElHJaPoint *);
         tSomGrPl *  SomGrEmpr(const cElHJaSomEmpr &);
	 tArcGrPl *  NewArcInterieur(tSomGrPl *,tSomGrPl *);
	 bool MakeFacettes(cElHJaArrangt &);
	 Video_Win * W();
	 cElHJaArrangt & Arrgt();

	 void SetStatePlanInterdit();
	 void SetStatePlanWithSegOblig(tBufFacette *);

	 std::vector<std::vector<Pt3dr> > FacesSols(bool WithIndet);

     private :
         tSomGrPl *  AddSom(bool & IsNew,Pt2dr aP,REAL Absc,bool IsEmpr);
	 cElHJaPlan3D(const cElHJaPlan3D &); // Non implemente
	 tSomGrPl * SomNearest(Pt2dr aP,REAL & aDist);


	 cElHJaArrangt &              mArrgt;
         INT                          mNum;
	 cElPlan3D                    mPlan;
	 std::vector<Pt2dr>           mEmpriseSpec;
	 bool                         mHasEmprSpec;
	 Video_Win *                  mW;
	 std::vector<cElHJaDroite *>  mVInters;
         tGrPl *                      mGr;
         std::vector<tSomGrPl *>      mVSomEmpr;
	 std::vector<cElHJaFacette *> mFacettes;
	 std::vector<cElHJaFacette *> mFacOblig;
	 bool                         mHasSegOblig;
	 Seg2d                        mSegOblig;
};

class cElHJaDroite
{
     public :
         cElHJaDroite(const ElSeg3D & aSeg,cElHJaPlan3D & aP1,cElHJaPlan3D & aP2,INT aNbPl);
	 // P1 et P2 pour verfier "temporairement" que l'on ne
	 // s'est pas emmele sur l'ordre des plan, a part ca inutile
         void AddPoint(cElHJaPoint &,cElHJaPlan3D & aP3,
			 tSomGrPl *s1,cElHJaPlan3D * aP1,
			 tSomGrPl *s2,cElHJaPlan3D * aP2);
	 ElSeg3D Droite();

	 // Calcul les intersection de la droite (consideree plani)
	 // avec l'emprise de l'arrangement, memorise les inter
	 // et les "transmet: aux plans support
	 void MakeIntersectionEmprise(const tEmprPlani &);

	 void AddArcsInterieurInGraphe(const std::vector<Pt2dr> & aEmprInit);

	 class cPaireSom
	 {
               public :
                  cPaireSom(tSomGrPl *,tSomGrPl *,const SegComp &);
                  tSomGrPl * mS1;
                  tSomGrPl * mS2;
		  Pt2dr      mPt;
                  REAL       mAbsc;
	 };
     private :
	 void AddPaire(tSomGrPl * aS1,tSomGrPl * aS2);


	 cElHJaDroite(const cElHJaDroite &); // Non implemente
	 ElSeg3D                    mDr;
	 SegComp                    mSegPl;
         std::vector<cElHJaPoint *> mInters;
	 cElHJaPlan3D *             mP1;
	 cElHJaPlan3D *             mP2;
	 std::vector<cPaireSom>     mVPaires;

};


class cElHJaPoint
{
      public :
          cElHJaPoint
          (
                Pt3dr aPt,
                cElHJaPlan3D & aP1,
                cElHJaPlan3D & aP2,
                cElHJaPlan3D & aP3
         );
	  //  retrouve les droites passant par le point
	  //  est fait a posteriori car necessite que tout les droites
	  //  soient construites et associees a leur plan
         void MakeDroites();
         Pt3dr Pt() const;

      private :
         cElHJaPoint(const cElHJaPoint &); // Non Impl
         Pt3dr          mPt;
         cElHJaPlan3D * mP1;
         cElHJaPlan3D * mP2;
         cElHJaPlan3D * mP3;
         tSomGrPl *     mS1;
         tSomGrPl *     mS2;
         tSomGrPl *     mS3;

         cElHJaDroite * mDr23;
         cElHJaDroite * mDr13;
         cElHJaDroite * mDr12;
};


class cElHJaFacette
{
      public :

         cElHJaFacette
         (
             const std::vector<tArcGrPl *> & aCont,
             cElHJaPlan3D * aPlan
         );
	 cElHJaPlan3D * Plan();
	 bool PointInFacette(Pt2dr aP) const;
	 void Show(REAL aDirH,INT aCoul,bool WithBox);
	 void ShowGlob();
	 void ShowCont(INT aCoul,Video_Win);
	 REAL Surf() const;
	 bool IsExterne() ; // La "Fausse" face externe

	 void MakeAdjacences();
	 void MakeRelRecouvrt(cElHJaFacette *);

	 bool IsRecouvrt(bool &EnDessus,cElHJaFacette * ,bool ShowMes);

         void MakeRecouvrt(cElHJaFacette *);

         void  ShowState();
         void  DupState();
         void  PopState();
	 void SetTopState(const FBool &);

	 bool IsSure();
	 bool IsImpossible();
	 bool IsIndeterminee();


         // Toute les fonction de manipulation des facette
         // renvoie true si il y a eu bloquage
         bool SetFacetteSure(tBufFacette *);
         bool SetFacetteImpossible(tBufFacette *);


         // Si la facette est sure, propage l'incompatibilte
         // a toute les voisine
         bool PropageIncompEnDessous(tBufFacette *);
         bool PropageIncompVert(tBufFacette *);
         bool PropageIncompVertGen(tBufFacette *,bool OnlyDessous);

	 bool PropageIcompVois(tBufFacette *);
         static bool PropageVoisRecurs(tBufFacette *);

	 // Si la facette est non refutee et que toute celle qui
	 // l'intersecte sont refute, alors elle devient certaine
	 //  A priori surtout pour mise au point
	 void SetSureIfPossible();
	 const  std::vector<tArcGrPl *>  & Arcs();
         INT    NbThisEnDessous() const;
         void AddDessouSansDessus(std::vector<cElHJaFacette *> &);

         void MakeInertie(Im2D_INT2 aMnt,INT aValForb,cGenSysSurResol *);
         const RMat_Inertie & MatInert() const;

      private :
         inline const FBool  & TopState() const;
         inline FBool & TopState() ;
         bool SetFacetteGen(tBufFacette *,const FBool& NewState,const FBool& ForbidSate);
         cElHJaFacette(const cElHJaFacette &); // Non Impl
	 cElHJaArrangt & Arrgt();
         void  AddRecouvrt(bool ThisIsEnDessus,cElHJaFacette * aF2);



         cElHJaPlan3D * mPlan;

	  std::vector<tArcGrPl *>  mVArcs;
	  std::vector<Pt2dr>       mVPt; // Points pour requetes geom

	  Box2dr                       mBox; // Boite Englob
	 // Pour un arc aCont[K]
	 //   mVFAdjcMPl[K] est la facette adjacente dans le meme plan
	 //   mVFAdjcComp[K] est la facette adjacente dans un autre plan
	 //          mais compatible avec elle;
	 //   mVFAdjcIncomp[K] Incompatible

	 std::vector<cElHJaFacette *> mVFAdjcPl;
	 std::vector<cElHJaFacette *> mVFAdjcComp;
	 std::vector<cElHJaFacette *> mVFAdjcIncomp;

	 // mVFInterVert : Ensemble des facettes ayant une intersection
	 // verticale non nulle avec elle
	 std::vector<cElHJaFacette *> mVFRecouvrt;
	 std::vector<INT>             mThisIsEnDessus;
         INT                          mNbThisIsEnDessous;
	 REAL                         mSurf;

         std::vector<FBool>           mStates;
         RMat_Inertie                 mMatInert;

};

class cElHJaArrangt_Visu
{
	public :
           cElHJaArrangt_Visu(Pt2di aSzG,Pt2di aNbVisu,Pt2di aSzPl);
	   Video_Win * WinOfPl(INT aK);
	   Video_Win   WG();
	   Video_Win   WG2();
	   Video_Win   WG3();

	   void GetPolyg
	        (
		    std::vector<Pt2dr> & aPolyg,
		    std::vector<int> &   aVSelect
		);

	   cElHJaFacette * GetFacette(const std::vector<cElHJaFacette *> &);

	private :
	   Pt2di                    mSzG;
	   Video_Win                mWG;
	   Video_Win                mWG2;
	   Video_Win                mWG3;
	   Pt2di                    mNbVisu;
	   Pt2di                    mSzPl;
           std::vector<Video_Win>   mVWPl;
           std::vector<Video_Win>   mVallW;
};

template <class cIter> cIter NextIter(const cIter & anIt)
{
	cIter aRes = anIt;
	aRes++;
	return aRes;
}

template <class tCont> void DeleteAndClear(tCont & aCont)
{
    for (typename tCont::iterator anIt=aCont.begin() ; anIt!=aCont.end() ; anIt++)
        delete *anIt;
    aCont.clear();
}

template <class Type> Type & VAt(std::vector<Type> & aV,INT aK)
{
    ELISE_ASSERT((aK>=0)&&(aK<int(aV.size())),"Out of Vect VAt");
    return aV[aK];
}
template <class Type> const Type & VAt(const std::vector<Type> & aV,INT aK)
{
    ELISE_ASSERT((aK>=0)&&(aK<int(aV.size())),"Out of Vect VAt");
    return aV[aK];
}

template <class Type>
const Type & VAtDef(const std::vector<Type> & aV,INT aK,const Type & aDef)
{
   if ((aK>=0)&&(aK<int(aV.size())))
      return aV[aK];
   return aDef;
}

template <class TVal,class TCont>
bool BoolFind(const TCont & aCont,const TVal & aVal)
{
   return std::find(aCont.begin(),aCont.end(),aVal) != aCont.end();
}

template <class Type>
int  IndFind(const std::vector<Type> & aCont,const Type & aVal)
{
   typename std::vector<Type>::const_iterator iT = std::find(aCont.begin(),aCont.end(),aVal);
   if (iT== aCont.end())
      return -1;
   return (int)(iT - aCont.begin());
}

template <class T1,class T2>
void AssertEntreeDicoVide(T1 &  aCont,const T2 & aVal,const std::string & aMessage)
{
    if (aCont.find(aVal) != aCont.end())
    {
         std::cout << "  NAME= "<< aVal << "\n";
         std::string aM = "Non unique name for "+aMessage;
         ELISE_ASSERT(false,aM.c_str());
    }

}


template <class TDic> 
typename TDic::mapped_type 
GetEntreeNonVide(TDic & aDic,const std::string& aName,const std::string& aMes)
{
    typename TDic::mapped_type  aV=aDic[aName];
    if (aV==0)
    {
        std::cout << "Entree = " << aName << "  ;; Contexte = " << aMes << "\n";
        ELISE_ASSERT(false,"Pas d'entree trouvee dans le dictionnaire\n");
    }
    return aV;
}





// Je ne comprend pas pourquoi mais BoolFind ne marche pas avec les map

template <class TVal,class TCont>
bool DicBoolFind(const TCont & aCont,const TVal & aVal)
{
   return aCont.find(aVal) != aCont.end();
}


template <class Type> class cVectTr : public std::vector<Type>
{
     public :
        cVectTr() :
           std::vector<Type>(),
           mDec(0)
        {
        }
        void SetDec(int aDec)
        {
             mDec = aDec;
        }
        Type & operator [] (int aK)
        {
             return std::vector<Type>::operator [](aK+mDec);
        }
        const Type & operator [] (int aK)  const
        {
             return std::vector<Type>::operator [](aK+mDec);
        }
     private  :
       int mDec;
};



class cTrapuFace
{
    public :
       friend class cTrapuBat;
    private :
        cTrapuFace(INT aNum);
        void AddSom(INT aNum);

        std::vector<INT>  mSoms;
        INT               mNum;
};


class cTrapuBat
{
    public :
        cTrapuBat();

        void AddFace
             (
                     const std::vector<Pt3dr> &,
                     REAL aEps,
		     INT aNum
             );
        Pt3dr  P0() const;
        Pt3dr  P1() const;
        INT    NbFaces() const;
        std::vector<Pt3dr> PtKiemeFace(INT aK) const;
        std::vector<Pt3dr> & Soms();
        Pt3dr & P0();
        Pt3dr & P1();

        void PutXML(class cElXMLFileIn &);

    private :
        INT  GetNumSom(Pt3dr aP,REAL aEpsilon);

        std::vector<Pt3dr>  mSoms;
        std::vector<cTrapuFace> mFaces;
	Pt3dr                     mP0;
	Pt3dr                     mP1;
        bool                      mFirstPt;
};



class cElImagesOfTrapu
{
       public :
	       // Si Dec = 0, calcule
           cElImagesOfTrapu
           (const cTrapuBat & aBat,INT aRab,bool Sup,Pt2di *Dec);

	   Im2D_INT1 ImLabel();
	   Im2D_REAL4 ImZ();
	   Im2D_U_INT1 ImShade();
           Pt2di Dec() const;
       private :
           Pt2di                     mDec;
	   Im2D_INT1                 mImLabel;
	   Im2D_REAL4                mImZ;
	   Im2D_U_INT1               mImShade;
};




class cElHJaArrangt
{
      public :
              cElHJaArrangt (Pt2di   aNbVisu = Pt2di(-1,-1));

	      void ReInit(const std::vector<Pt2dr> &);
	      cElHJaPlan3D * AddPlan(const cElPlan3D & aPlan,const std::vector<Pt2dr>* anEmprise = 0);
	      void ConstruireAll();


              void TestPolygoneSimple(const std::vector<Pt2dr> &,const std::vector<int> &);
              void TestInteractif();
	      void Show();
              void ShowStateFacette();
	      void AddFacette(cElHJaFacette *);

	      struct  cStatistique
	      {
                      cStatistique(REAL Sure,REAL Impos,REAL Indet);

                      REAL mSurfSure;
                      REAL mSurfImpossible;
                      REAL mSurfIndet;
	      };

	      void AddStatistique(REAL aSurf,const FBool &);

	      void ShowStatistique();
	      REAL SurfResiduelle();
	      bool StdSolFlag(INT aFlag);
              void MakeInertieFacettes(Im2D_INT2 aMnt,INT aValForb);
	      RMat_Inertie MInertCurSol();
              void  PopStateFac();
	      enum eModeEvalZ
	      {
		      eNoCorrecZ,
		      eCorrelEvalZ,
		      eL2EvalZ,
		      eL1EvalZ
	      };

	      cTrapuBat  MakeSolTrapu
                         (
                              Pt2dr aDec,
			      eModeEvalZ,
			      INT  aValZ,
			      Im2D_INT2 aMnt,
                              bool WithIndet  = false
                         );

      protected :
              bool GetTheSolution(bool WithForcageSup,tBufFacette & aBuf);
              void ForcageSup();
              void InitFlag(INT aFlag,tBufFacette & aBuf);

	      void SetAllFacSureIfPossible();
              void  DupStateFac();
	      static const REAL Epsilon;
	      cElHJaArrangt(const cElHJaArrangt &); // Non Implementes

              void TriTopologiqueFacette();


	      typedef std::vector<cElHJaPlan3D *> tContPl;
	      typedef tContPl::iterator           tItPl;
	      typedef std::vector<cElHJaDroite *> tContDr;
	      typedef tContDr::iterator           tItDr;
	      typedef std::vector<cElHJaPoint *>  tContPoint;
	      typedef tContPoint::iterator        tItPoint;
	      typedef std::vector<cElHJaFacette *>  tContFac;
	      typedef tContFac::iterator            tItFac;

	      Video_Win * WinOfPl(INT aK);

	      cElHJaArrangt_Visu *     mVisu;
              tContPl                  mPlans;
              tContDr                  mDroites;
              tContPoint               mPoints;
	      tContFac                 mFacettes;
              tEmprPlani               mEmprPl;  // Emprise planimetrique
              std::vector<Pt2dr>       mEmprVPt2d; // Id prec, pour format PointInPoly
              std::vector<Pt2dr>        mEmptyEmprise; // Utilitaire
	      INT                       mNbPl;
	      std::vector<cStatistique> mStats;
	      REAL                      mSurfEmpr;

	      cGenSysSurResol *         mSysResolv;
	      L2SysSurResol             mL2SysResolv;
	      SystLinSurResolu          mL1SysResolv;
};

template <class T1,class T2> void ConvertContainer(T1 & aV1,const T2 & aV2)
{
    for (typename T2::const_iterator it2=aV2.begin() ; it2!=aV2.end() ; it2++)
    {
        aV1.push_back( (typename T1::value_type)(*it2));
    }
}



#endif // _HASSAN_ARRANGT_H


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
