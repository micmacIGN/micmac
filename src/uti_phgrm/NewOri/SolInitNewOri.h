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

#ifndef _ELISE_SOLINIT_NEW_ORI_H
#define _ELISE_SOLINIT_NEW_ORI_H


class cLinkTripl;
class cNOSolIn_AttrSom;
class cNOSolIn_AttrASym;
class cNOSolIn_AttrArc;
class cAppli_NewSolGolInit;
class cNOSolIn_Triplet;

typedef  ElSom<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tSomNSI;
typedef  ElArc<cNOSolIn_AttrSom,cNOSolIn_AttrArc>         tArcNSI;
typedef  ElSomIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItSNSI;
typedef  ElArcIterator<cNOSolIn_AttrSom,cNOSolIn_AttrArc> tItANSI;
typedef  ElGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>      tGrNSI;
typedef  ElSubGraphe<cNOSolIn_AttrSom,cNOSolIn_AttrArc>   tSubGrNSI;

class cLinkTripl
{
     public :
         cLinkTripl(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3) :
            m3   (aTrip),
            mK1  (aK1),
            mK2  (aK2),
            mK3  (aK3)
         {
         }

         cNOSolIn_Triplet  *  m3;
         U_INT1               mK1;
         U_INT1               mK2;
         U_INT1               mK3;
         tSomNSI *            S1() const;
         tSomNSI *            S2() const;
         tSomNSI *            S3() const;
};



class cNOSolIn_AttrSom
{
     public :
         cNOSolIn_AttrSom(const std::string & aName,cAppli_NewSolGolInit & anAppli);
         cNOSolIn_AttrSom() :
             mCurRot (ElRotation3D::Id),
             mTestRot (ElRotation3D::Id)
         {}

         void AddTriplet(cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
         // std::vector<cNOSolIn_Triplet *> & V3() {return mV3;}
         cNewO_OneIm * Im() {return mIm;}
         void ReInit();
         ElRotation3D & CurRot() {return mCurRot;}
         ElRotation3D & TestRot() {return mTestRot;}
         

     private :
         std::string                      mName;
         cAppli_NewSolGolInit *           mAppli;
         cNewO_OneIm *                    mIm;
         std::vector<cLinkTripl >         mLnk3;
         double                           mCurCostMin;
         ElRotation3D                     mCurRot;
         ElRotation3D                     mTestRot;
};


class cNOSolIn_AttrASym
{
     public :
         void AddTriplet(cNOSolIn_Triplet * aTrip,int aK1,int aK2,int aK3);
         std::vector<cLinkTripl> & Lnk3() {return mLnk3;}
         ElRotation3D &    EstimC2toC1() {return mEstimC2toC1;}
         cNOSolIn_AttrASym();
         void PostInit();
     private :
         std::vector<cLinkTripl> mLnk3;
         ElRotation3D            mEstimC2toC1;

};



class cNOSolIn_AttrArc
{
     public :
           cNOSolIn_AttrArc(cNOSolIn_AttrASym *,bool OrASym);
           cNOSolIn_AttrASym * ASym() {return mASym;}
           ElRotation3D  EstimC2toC1() {return mOrASym ? mASym->EstimC2toC1() :  mASym->EstimC2toC1().inv();}
           bool          IsOrASym() const {return mOrASym;}

     private :
           cNOSolIn_AttrASym * mASym;
           bool                mOrASym;
};


class cNOSolIn_Triplet
{
      public :
          cNOSolIn_Triplet(cAppli_NewSolGolInit *,tSomNSI * aS1,tSomNSI * aS2,tSomNSI *aS3,const cXml_Ori3ImInit &);
          void SetArc(int aK,tArcNSI *);
          tSomNSI * KSom(int aK) const {return mSoms[aK];}
          tArcNSI * KArc(int aK) const {return mArcs[aK];}

          void InitRot3Som();
          const ElRotation3D & RotOfSom(tSomNSI * aS) const
          {
                if (aS==mSoms[0]) return ElRotation3D::Id;
                if (aS==mSoms[1]) return mR2on1;
                if (aS==mSoms[2]) return mR3on1;
                ELISE_ASSERT(false," RotOfSom");
                return ElRotation3D::Id;
          }
          double BOnH() const {return mBOnH;}
          int  Nb3() const {return mNb3;}
          double  Cost() const {return mCost;}

          void CalcCoherFromArcs(bool Test);
          void CheckArcsSom();

           void Show(const std::string & aMes) const ;
           const Pt3dr &  PMed() const {return mPMed;}


      private :
          cAppli_NewSolGolInit * mAppli;
          tSomNSI *     mSoms[3];
          tArcNSI *     mArcs[3];
          ElRotation3D  mR2on1;
          ElRotation3D  mR3on1;
          double        mBOnH;
          int           mNb3;
          Pt3dr         mPMed;
   // Gere les triplets qui vont etre desactives
          bool          mAlive;
          double        mCost;
};


class cCmpPtrTriplOnCost
{
   public :
       bool operator () (const  cNOSolIn_Triplet * aT1,const  cNOSolIn_Triplet * aT2)
       {
           return aT1->Cost() < aT2->Cost() ;
       }
};


class cAppli_NewSolGolInit
{
    public :
        cAppli_NewSolGolInit(int , char **);
        cNewO_NameManager & NM() {return *mNM;}
        double  CoherMed12() const {return  mCoherMed12;}

    private :

        void GenerateOneSol(cNOSolIn_Triplet * aTri);
        void GenerateAllSol();


        void FinishNeighTriplet();

        void TestInitRot(tArcNSI * anArc,const cLinkTripl & aLnk);
        void TestOneTriplet(cNOSolIn_Triplet *);
        void SetNeighTriplet(cNOSolIn_Triplet *);

        void SetCurNeigh3(tSomNSI *);
        void SetCurNeigh2(tSomNSI *);

        void                 CreateArc(tSomNSI *,tSomNSI *,cNOSolIn_Triplet *,int aK0,int aK1,int aK2);
        void   EstimCoherenceMed();
        void   EstimRotsInit();
        void   EstimCoheTriplet();
        void    InitRotOfArc(tArcNSI * anArc,bool Test);




        std::string          mFullPat;
        std::string          mOriCalib;
        cElemAppliSetFile    mEASF;
        cNewO_NameManager  * mNM;
        bool                 mQuick;
        bool                 mTest;
        bool                 mSimul;

        tGrNSI               mGr;
        tSubGrNSI            mSubAll;
        std::map<std::string,tSomNSI *> mMapS;
        cComputecKernelGraph             mCompKG;

// Variables temporaires pour charger un triplet 
        std::vector<tSomNSI *>  mVCur3;  // Tripelt courrant
        std::vector<tSomNSI *>  mVCur2;  // Adjcent au triplet courant
        std::vector<cNOSolIn_Triplet*> mV3;
        int                     mFlag3;
        int                     mFlag2;
        cNOSolIn_Triplet *      mTestTrip;
        cNOSolIn_Triplet *      mTestTrip2;
        tSomNSI *               mTestS1;
        tSomNSI *               mTestS2;
        tArcNSI *               mTestArc;
        int                     mNbSom;
        int                     mNbArc;
        int                     mNbTrip;
        double                  mCoherMedAB;
        double                  mCoherMed12;

};


void AssertArcOriented(tArcNSI *);
ElRotation3D RotationC2toC1(tArcNSI * anArc,cNOSolIn_Triplet * aTri);











#endif // _ELISE_SOLINIT_NEW_ORI_H

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
