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

#include "Extern_NewO.h"

class cNewO_NameManager;

class cArgCreateSPV
{
     public :
         cArgCreateSPV(cNewO_NameManager * aNM,cNewO_OneIm *anI1,cNewO_OneIm * anI2) :
              mNM (aNM),
              mI1 (anI1),
              mI2 (anI2)
         {
         } 

         cNewO_NameManager * mNM;
         cNewO_OneIm *       mI1;
         cNewO_OneIm *       mI2;
};


class cSwappablePairVPts : public cBaseSwappable
{
     public :
   // Pre-requis
         typedef cArgCreateSPV  Swap_tArgCreate;

         void Swap_Delete()
         {
             delete mVP1;
             delete mVP2;
             mVP1=0;
             mVP2=0;
         }
         bool Swap_Loaded() {return mVP1!=0;}
         void Swap_Create(const Swap_tArgCreate & anArg);
         int  Swap_SizeOf();

   //======================

         cSwappablePairVPts() :
            mVP1 (0),
            mVP2 (0),
            mInfP1 ( 1e20, 1e20),
            mSupP1 (-1e20,-1e20)
         {
         }

         std::vector<Pt2df> * mVP1; 
         std::vector<Pt2df> * mVP2; 
         Pt2df                mInfP1;
         Pt2df                mSupP1;
};

int  cSwappablePairVPts::Swap_SizeOf()
{
    return (int)(2 * (sizeof(Pt2df) * mVP1->size() + sizeof(std::vector<Pt2df>)));
}

void cSwappablePairVPts::Swap_Create(const Swap_tArgCreate & anArg)
{
    mVP1 = new std::vector<Pt2df>;
    mVP2 = new std::vector<Pt2df>;
    anArg.mNM->LoadHomFloats(anArg.mI1,anArg.mI2,mVP1,mVP2);

    if (Swap_NbCreate()==0)
    {
       for (int aKP=0 ; aKP<int(mVP1->size()) ; aKP++)
       {
            const Pt2df & aP1 = (*mVP1)[aKP];
            mInfP1 = Inf(mInfP1,aP1);
            mSupP1 = Sup(mSupP1,aP1);
       }
    }
}


// NewOri.h:#define  TNbCaseP1  6  // Nombre de case sur lesquelle on discretise
// Un certain  nombre de grandeur sont calcule en geometrie image avec des
// densite, cette valeur fixe le nombre de cases

// Par efficacite (??) ceci est géré de manière linéaire, la fonction
// ToIndex fait la corresponace entre ces indexes lineaire et les coordonnees images
// initiale.

// mNb => memorise le nombre de point par case
// mDens est une densite, theoriquement dans [0,1], mais comme tout les tableaux sont entier, c'est 
// multiplie par TQuant



static const int  TMaxNbCase = TNbCaseP1 * TNbCaseP1;  
static const int  TMaxGain = 2e9;

class cGTrip_AttrSom;
class cGTrip_AttrASym;
class cGTrip_AttrArc;
class cAppli_GenTriplet;

typedef  ElSom<cGTrip_AttrSom,cGTrip_AttrArc>         tSomGT;
typedef  ElArc<cGTrip_AttrSom,cGTrip_AttrArc>         tArcGT;
typedef  ElSomIterator<cGTrip_AttrSom,cGTrip_AttrArc> tItSGT;
typedef  ElArcIterator<cGTrip_AttrSom,cGTrip_AttrArc> tItAGT;
typedef  ElGraphe<cGTrip_AttrSom,cGTrip_AttrArc>      tGrGT;
typedef  ElSubGraphe<cGTrip_AttrSom,cGTrip_AttrArc>   tSubGrGT;


class cGTrip_AttrSom
{
     public :
         cGTrip_AttrSom(int aNum,const std::string & aNameIm,cAppli_GenTriplet & anAppli) ;

         cGTrip_AttrSom() : mM3(1,1)  {}

         const int & Num() const {return mNum;}
         const std::string & Name() const {return mName;}
         cNewO_OneIm & Im() const {return *mIm;}

         bool InitTriplet(tSomGT*,tArcGT *);
         void InitNb(const std::vector<Pt2df> & aVP1);
         const int &  Nb(int aK) {return mNb[aK];}
         const Pt3dr  & C3() const {return mC3;}
         ElRotation3D R3() const {return ElRotation3D(mC3,mM3,true);}


         void UpdateCost(tSomGT * aSomThis,tSomGT *aSomSel);
         bool & SomTest() {return mTest;}
         int  GainGlob() const {return mGainGlob;}
         int * Dens() {return  mDens;}

     private :

         cAppli_GenTriplet * mAppli;
         int            mNum;
         std::string         mName;
         cNewO_OneIm *       mIm;
         Pt3dr               mC3;
         ElMatrix<double>    mM3;

         int mNb[TMaxNbCase];
         int mDens[TMaxNbCase];
         int mGain[TMaxNbCase];
         int mGainGlob;
         bool mTest;
};

class cGTrip_AttrASym
{
     public :
        cGTrip_AttrASym(const cXml_Ori2Im & aXml) :
             mXML   (aXml),
             mTest  (false)
        {
        }
        const cXml_Ori2Im & Xml() const {return mXML;}
        bool  & ArcTest() {return mTest;}

     private  :
        cGTrip_AttrASym(const cGTrip_AttrASym & aXml) ; // N.I.
        cXml_Ori2Im mXML;

        bool              mTest;
};

class cGTrip_AttrArc
{
    public :
        cGTrip_AttrArc(ElRotation3D aR,cGTrip_AttrASym * anASym,bool ASymIsDir) :
             mRot   (aR),
             mASym  (anASym),
             mASDir (ASymIsDir)
        {
        }
        cGTrip_AttrASym & ASym() {return *mASym;}
        bool IsDirASym() const {return mASDir;}
        const ElRotation3D & Rot() const {return mRot;}

    private  :
        ElRotation3D      mRot;
        cGTrip_AttrASym * mASym;
        bool              mASDir;
};

tSomGT * S1ASym(tArcGT * anArc) {return anArc->attr().IsDirASym() ? &(anArc->s1()) : &(anArc->s2());}
tSomGT * S2ASym(tArcGT * anArc) {return anArc->attr().IsDirASym() ? &(anArc->s2()) : &(anArc->s1());}



class cResTriplet
{
    public :
        cXml_Ori3ImInit  mXml;
};

typedef cTplTriplet<int> cTripletInt;


class cAppli_GenTriplet : public cCommonMartiniAppli
{
    public :
       cAppli_GenTriplet(int argc,char ** argv);
       void  GenTriplet();
       cNewO_NameManager & NM() {return *mNM;}

       int ToIndex(const Pt2df &  aP) const;
       int NbCases() const {return mNbCases;}
       tSomGT * CurS1() {return mCurS1;}
       tSomGT * CurS2() {return mCurS2;}

       int GainBSurH(tSomGT * aS1,tSomGT * aS2);
       int * Pds() {return mPds;}
       bool  CurTestArc() const {return  mCurTestArc;}
       double  MulQuant() const {return  mMulQuant;}


       void TestS3()
       {
            if (mSomTest3)
            {
               int * aD = mSomTest3->attr().Dens();
               std::cout << "------------------------------------ --------mSomTest3 " << aD[0] << "\n";
            }
       }
       bool  AddTriplet(tSomGT & aS1,tSomGT & aS2,tSomGT & aS3);
       double                        mTimeLoadCple;
       double                        mTimeLoadTri;
       int                           mNbLoadCple;
       int                           mNbLoadTri;
       std::string & InOri() {return  mInOri;}

    private :


       void  GenTriplet(tArcGT & anArc);
       void AddSomTmp(tSomGT & aS);

       tSomGT * GetNextSom();

       tGrGT                mGrT;
       tSubGrGT             mSubAll;
       std::string          mFullName;
       cElemAppliSetFile    mEASF;
       cNewO_NameManager *  mNM;
       std::map<std::string,tSomGT *> mMapS;
       std::vector<tSomGT *>          mVecAllSom;

       std::map<cTripletInt,cResTriplet>  mMapTriplets;
       cXml_TopoTriplet                   mTopoTriplets;
       std::map<cCpleString,cListOfName>  mTriOfCple;

       // Voisin de l'arc, hors de l'arc lui meme
       std::vector<tSomGT *>         mVSomVois;
       std::vector<tSomGT *>         mVSomEnCourse;
       std::vector<tSomGT *>         mVSomSelected;
       int                           mFlagVois;
       tArcGT *                      mCurArc;
       bool                          mCurTestArc;
       Pt3dr                         mCurPMed;
       double                        mHautBase;
       tSomGT *                      mCurS1;
       tSomGT *                      mCurS2;
       tSomGT *                      mSomTest3;
       bool                          mDebug;
       Pt2df                         mPInf;
       Pt2df                         mPSup;
       double                        mStepCases;
       Pt2di                         mSzCases;
       int                           mNbCases;
       int                           mPds[TMaxNbCase];
       std::string                   mNameTest1;
       std::string                   mNameTest2;
       std::string                   mNameTest3;
       double                        mMulQuant;
       double                        mTimeMerge;
       double                        mTimeSelec;
       double                        mRamAllowed;
       int                           mKS0;
       bool                          mSelAll;
};

/*********************************************************/
/*                                                       */
/*               cGTrip_AttrSom                          */
/*                                                       */
/*********************************************************/

cGTrip_AttrSom::cGTrip_AttrSom(int aNum,const std::string & aNameIm,cAppli_GenTriplet & anAppli) :
     mAppli  (&anAppli),
     mNum    (aNum),
     mName   (aNameIm),
     mIm     (new cNewO_OneIm(anAppli.NM(),mName)),
     mM3     (3,3),
     mTest   (false)
{
}


void cGTrip_AttrSom::InitNb(const std::vector<Pt2df> & aVP1)
{
    for (int aK=0 ;  aK< TMaxNbCase ; aK++)
        mNb[aK] = 0;

    for (int aK=0 ; aK<int(aVP1.size()) ; aK++)
    {
        mNb[mAppli->ToIndex(aVP1[aK])] ++;
    }
}


extern void  CoordInterSeg(const Pt3dr & aP0,const Pt3dr & aP1,const Pt3dr & aQ0,const Pt3dr & aQ1,bool & Ok,double &p , double & q);



bool cGTrip_AttrSom::InitTriplet(tSomGT * aSom,tArcGT * anA12)
{
      tGrGT & aGr = aSom->gr();
      tArcGT* anA13 = aGr.arc_s1s2(anA12->s1(),*aSom);
      tArcGT* anA23 = aGr.arc_s1s2(anA12->s2(),*aSom);

      static std::vector<Pt2df> aVP1;
      static std::vector<Pt2df> aVP2;
      static std::vector<Pt2df> aVP3;

      ElTimer aChrono;
      bool OK = mAppli->NM().LoadTriplet
                (
                     anA12->s1().attr().mIm,anA12->s2().attr().mIm, mIm,
                     &aVP1,&aVP2,&aVP3
                );
      mAppli->mTimeLoadTri += aChrono.uval();
      mAppli->mNbLoadTri ++;

       if (! OK) return false;



      // std::cout << "initTriplet " << aNb << " " << aNb3 << " " << aNb33 << "\n";

      // Pour qu'il y ait intersection
      // Det(L C1C3 + C12,  R3U3, ,R2U) = 0
      std::vector<double> aVL13;
      std::vector<double> aVL23;
      std::vector<double> aVLInv;
      ElRotation3D aR21 = anA12->attr().Rot();
      ElRotation3D aR31 = anA13->attr().Rot();
      ElRotation3D aR32 = anA23->attr().Rot();
      ElRotation3D aR31Bis =  aR21 * aR32;


      Pt3dr aC1(0,0,0);
      Pt3dr aC2 = aR21.tr();
      Pt3dr aC3 = aR31.tr();
      Pt3dr aV3Bis = aR31Bis.tr() - aC2;

     // C'est refait a chaque fois, pas grave ...
      mAppli->CurS1()->attr().mC3 = aC1;
      mAppli->CurS1()->attr().mM3 = ElMatrix<double>(3,true);
      mAppli->CurS2()->attr().mC3 = aC2;
      mAppli->CurS2()->attr().mM3 = aR21.Mat();


      // Calul du centre
 
      // Methode qui n'utilisent pas les points multiples, OK mais degeneree avec sommet alignes
      //  C2  + L C3, U2 , U3 colineaire
     

      int aStep = ElMin(10,ElMax(1,int(aVP1.size())/50));

      std::vector<double> aVLByInt;

      for (int aKL=0 ; aKL<int(aVP1.size()) ; aKL+=aStep)
      {
          Pt3dr aU31(aR31.ImVect(Pt3dr(aVP3[aKL].x,aVP3[aKL].y,1.0)));
          Pt3dr aU2(aR21.ImVect(Pt3dr(aVP2[aKL].x,aVP2[aKL].y,1.0)));

          Pt3dr aU1(Pt3dr(aVP1[aKL].x,aVP1[aKL].y,1.0));
          Pt3dr aU31Bis(aR31Bis.ImVect(Pt3dr(aVP3[aKL].x,aVP3[aKL].y,1.0)));


          double aDet = Det(aC2,aU31,aU2);
          if (aDet)
              aVL13.push_back(Det(aC3,aU31,aU2) / aDet);

          aDet =  Det(aV3Bis,aU31Bis,aU1);
          if (aDet)
              aVL23.push_back( -Det(aC2,aU31Bis,aU1) /aDet);


          bool OkI;
          Pt3dr aPInt12 = InterSeg(aC1,aC1+aU1,aC2,aC2+aU2,OkI);
          if (OkI)
          {
              // La vrai C3 est a la fois sur la droite C1C3 et sur le rayon partant de l'intersection et porte
              // par U3, on calcule
              double p,q;
              CoordInterSeg(aC1,aC3,aPInt12,aPInt12+aU31,OkI,p,q);
              if (OkI)
              {
                 aVLByInt.push_back(p);
              }

               // Pt3dr aPseudoC3 = InterSeg(aC1,aC3,
               //  L C3 + K U3 = aPInt12
          }
          
      }
	  if (aVL13.empty()) return false;
	  if (aVL23.empty()) return false;
	  if (aVLByInt.empty()) return false;

      Pt3dr aC31 = aC3 / MedianeSup(aVL13);
      Pt3dr aC32 = aC2 + aV3Bis * MedianeSup(aVL23);
      mC3 = (aC31 + aC32) / 2.0;
      Pt3dr aC3I = aC3 *  MedianeSup(aVLByInt);

      // std::cout << "DDDDD " << euclid(mC3-aC3I)  << " " << aC3I << "\n";

     // Finalement on prefere celui par point multiple, qui n'est pas degenere en cas de sommets alignes
      mC3 = aC3I;
      mM3 =  NearestRotation((aR31.Mat() + aR31Bis.Mat())*0.5);

      //  Cas ou In Ori est 
      std::string & aInOri = mAppli->InOri();
      if (EAMIsInit(&aInOri))
      {
          // Les deux premieres viennent de la paire sauvee en xml, donc pas besoin
          bool Ok;
          std::pair<ElRotation3D,ElRotation3D>  aPair = mAppli->NM().OriRelTripletFromExisting
                                                (
                                                    aInOri,
                                                    anA12->s1().attr().mIm->Name(),
                                                    anA12->s2().attr().mIm->Name(),
                                                    aSom->attr().mIm->Name(),
                                                    Ok
                                                );

          if (Ok)
          {
               mM3 = aPair.second.Mat();
               mC3 = aPair.second.tr();
/*
               std::cout << " D2 " << euclid(aC2-aPair.first.tr())
                         << " D3 " << euclid(mC3-aPair.second.tr())
                         << " M2 " << (aR21.Mat()-aPair.first.Mat()).L2()
                         << " M3 " << (mM3-aPair.second.Mat()).L2()
                         << " \n";
*/
          }
      }


     // Calcul gain et densite

      int aGain1 = mAppli->GainBSurH( mAppli->CurS1(),aSom);
      int aGain2 = mAppli->GainBSurH( mAppli->CurS2(),aSom);
      int aGain = ElMin(aGain1,aGain2);


      // Initialise en fonction des points triple de 1 et 2 et 3
      // alors que NbGlob a ete initialise en fonction des points double
      // !!!! [NB23]  Les points triples ne sont pas un sous ensemble des point 
      // double  pt AB et pt BC  => triple ABC sans paire AC 
      InitNb(aVP1);
      int aNbC = mAppli->NbCases();
      int * aNbGlob =  mAppli->CurS1()->attr().mNb;
      int * aPdsGlob = mAppli->Pds();
      mGainGlob = 0;
      for (int aK=0 ; aK< aNbC ; aK++)
      {
           // densite des 3 par rapport au 2
           double aDens = double(mNb[aK]) / double(ElMax(1,aNbGlob[aK]));
           aDens = ElMin(1.0,aDens); // voir NB23 , la densite peut etre > 1
           aDens = (aDens * TAttenDens) / (aDens * TAttenDens +1) ; // ?? redondant avec prec
           mDens[aK] = round_ni( (TQuant * aDens * (TAttenDens+1)) / TAttenDens);

           mGain[aK] = mDens[aK] * TQuant * aGain;
           mGainGlob +=  mGain[aK] * aPdsGlob[aK];  // Pds de la case prop a sqrt du NbGlob
          /*
                Le gain c'est + ou -  Som (Nb/NbGlob * sqrt(NbGblob)) , a revoir ...
          */
      }

      if (mAppli->CurTestArc())
      {
                       //     D * D * Pds
          static bool First = true;
          double aMulQ = mAppli->MulQuant();
          if (First)
             std::cout << " ========= Mul = " << aMulQ << " ========\n";
           First = false;


          double aFlg = double(mGainGlob) / aMulQ;
          std::cout << "Name " << aSom->attr().Name() << " G=" << aFlg  << " " << aGain / double(TQuantBsH)  << "\n";
      }

      return true;
}


void  cGTrip_AttrSom::UpdateCost(tSomGT * aSomThis,tSomGT *aSomSel)
{

      int aNbC = mAppli->NbCases();
      mGainGlob = 0;
      int aNewGain =  mAppli->GainBSurH(aSomThis,aSomSel);
      int * aPdsGlob = mAppli->Pds();
      int * aDens2 = aSomSel->attr().mDens;

      for (int aK=0 ; aK< aNbC ; aK++)
      {
           int aD2 =  aDens2[aK];
           ElSetMin(mGain[aK], mDens[aK] * ( aNewGain * aD2 + TQuant*(TQuant-aD2)));
           mGainGlob +=  mGain[aK] * aPdsGlob[aK];
/*
           mGain[aK] = mDens[aK] * TQuant * aGain;
           mGainGlob +=  mGain[aK] * aPdsGlob[aK];  // Pds de la case prop a sqrt du NbGlob
*/
      }
}

/*********************************************************/
/*                                                       */
/*              cAppli_GenTriplet                        */
/*                                                       */
/*********************************************************/



void cAppli_GenTriplet::AddSomTmp(tSomGT & aS)
{
   if ( aS.flag_kth(mFlagVois))
      return;

   aS.flag_set_kth_true(mFlagVois);
   mVSomVois.push_back(&aS);


   if (aS.attr().InitTriplet(&aS,mCurArc))
      mVSomEnCourse.push_back(&aS) ;
}


int cAppli_GenTriplet::GainBSurH(tSomGT * aS1,tSomGT * aS2)
{
    double aBSH = euclid(aS1->attr().C3()-aS2->attr().C3()) / mHautBase;

    return round_ni(TQuantBsH * aBSH  / (aBSH + TBSurHLim));
}


int cAppli_GenTriplet::ToIndex(const Pt2df &  aP0) const
{
    Pt2di aP =  round_down((aP0-mPInf)/mStepCases);
    aP.x  = ElMax(0,ElMin(mSzCases.x-1,aP.x));
    aP.y  = ElMax(0,ElMin(mSzCases.y-1,aP.y));

    int aRes = aP.x + aP.y * mSzCases.x;
    ELISE_ASSERT(aRes>=0 && aRes<mNbCases,"cAppli_GenTriplet::ToIndex");
    return aRes;
}


tSomGT * cAppli_GenTriplet::GetNextSom()
{
   int aNbMaxTriplet = mQuick ? mTQuickNbMaxTriplet : mTStdNbMaxTriplet;
   if (mVSomEnCourse.empty()) return 0;
   if (int(mVSomSelected.size()) > aNbMaxTriplet) return 0;

   int aGainMax=-1;
   tSomGT * aRes=0;
   int aIndexRes = -1;

//std::cout << "GMMaax " << aGainMax << "\n";
   for (int aK=0 ; aK<int(mVSomEnCourse.size()) ; aK++)
   {
//std::cout << "GGGGlob " << mVSomEnCourse[aK]->attr().GainGlob() << "\n";
        if (mVSomEnCourse[aK]->attr().GainGlob() > aGainMax)
        {
            aRes = mVSomEnCourse[aK];
            aIndexRes = aK;
            aGainMax = aRes->attr().GainGlob();
        }
   }
   ELISE_ASSERT(aRes != 0,"cAppli_GenTriplet::GetNextSom");

   if ((!mSelAll) && (aGainMax < (TGainSeuil * mMulQuant))) return 0;

   mVSomEnCourse.erase(mVSomEnCourse.begin()+aIndexRes);
   mVSomSelected.push_back(aRes);

   for (int aK=0 ; aK<int(mVSomEnCourse.size()) ; aK++)
   {
      tSomGT * aSom = mVSomEnCourse[aK];
      aSom->attr().UpdateCost(aSom,aRes);
   }

   return aRes;
}


void cAppli_GenTriplet::GenTriplet(tArcGT & anArc)
{
    if (!anArc.attr().IsDirASym() ) return;
    mCurArc = & anArc;



    std::vector<Pt2df> aVP1,aVP2;
    ElTimer aChrono;
    mNM->LoadHomFloats(&(anArc.s1().attr().Im()),&(anArc.s2().attr().Im()),&aVP1,&aVP2);
if (0&&MPD_MM())
{
    std::cout << "iiIiii " << aVP1.size() << "\n";
}
    mTimeLoadCple += aChrono.uval();
    mNbLoadCple ++;

    mPInf = Pt2df(1e20,1e20);
    mPSup = Pt2df(-1e20,-1e20);

    for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
    {
        const Pt2df & aP1 = aVP1[aKP];
        mPInf = Inf(mPInf,aP1);
        mPSup = Sup(mPSup,aP1);
    }

    mCurTestArc = anArc.attr().ASym().ArcTest();


    mCurPMed = mCurArc->attr().ASym().Xml().Geom().Val().OrientAff().PMed1();
    mHautBase = euclid(mCurPMed);

    mCurS1  = & (anArc.s1());
    mCurS2  = & (anArc.s2());

    Pt2df aPLarg = mPSup-mPInf;
    mStepCases = ElMax(aPLarg.x,aPLarg.y) / TNbCaseP1;
    mSzCases = Pt2di(round_up(aPLarg.x/mStepCases),round_up(aPLarg.y/mStepCases));
    mSzCases = Inf(mSzCases,Pt2di(TNbCaseP1,TNbCaseP1));
    mNbCases = mSzCases.x * mSzCases.y;
    ELISE_ASSERT(mNbCases<=TMaxNbCase,"cAppli_GenTriplet::GenTriplet");

    mMulQuant  = mNbCases *pow((float)TQuant,3) * TQuantBsH;
    ELISE_ASSERT(mMulQuant<TMaxGain,"Owerflow in cAppli_GenTriplet::GenTriplet");

    // mCurS1->attr().InitNb(mCurArc->attr().VPts1());
    mCurS1->attr().InitNb(aVP1);
    int aNbMax = 0;
    for (int aK=0 ; aK<mNbCases ; aK++)
    {
        ElSetMax(aNbMax,mCurS1->attr().Nb(aK));
    }
    for (int aK=0 ; aK<mNbCases ; aK++)
    {
        double aPds = mCurS1->attr().Nb(aK) / double(aNbMax);
        mPds[aK] = round_ni(TQuant*sqrt(aPds));
    }


    ElTimer aChroMerge;
    for(tItAGT itA=anArc.s1().begin(mSubAll) ; itA.go_on() ; itA++)
    {
       tSomGT & aS3 = (*itA).s2();
       if (mGrT.arc_s1s2(anArc.s2(),aS3))
       {
          if (mCurTestArc) 
          {
             std::cout << "Push " <<  aS3.attr().Name() << "\n";
          }
          AddSomTmp(aS3);
       }
    }
if (0&&MPD_MM())
{
    std::cout << "jjjjjj " << aVP1.size() << "\n";
    getchar();
}
    mTimeMerge += aChroMerge.uval();


    if (mCurTestArc) std::cout << " -*-*-*-*-*-*-*-*-\n";
    ElTimer aChroSel;
    while (tSomGT * aSom = GetNextSom())
    {
        AddTriplet(*aSom,mCurArc->s1(),mCurArc->s2());
        if (mCurTestArc)
           std::cout << " SEL " << aSom->attr().Name()  << " G " << aSom->attr().GainGlob() / mMulQuant << "\n";
    }
    mTimeSelec += aChroSel.uval();


    // Vider les structure temporaires
    for (int aKS=0 ; aKS<int(mVSomVois.size()) ; aKS++)
    {
       mVSomVois[aKS]->flag_set_kth_false(mFlagVois);
    }
    mVSomVois.clear();
    mVSomEnCourse.clear();
    mVSomSelected.clear();
    mCurArc = 0;

    if (mCurTestArc)
    {
        getchar();
    }
}


void cAppli_GenTriplet::GenTriplet()
{
   mTimeLoadCple = 0.0;
   mTimeLoadTri = 0.0;
   mNbLoadCple =0;
   mNbLoadTri  =0;

   ElTimer aTimeGT;
   for (int aKS=mKS0 ; aKS<int(mVecAllSom.size()) ; aKS++)
   {
       if (mShow) 
       {
          std::cout << "KS " << aKS
                    << " ONE SOOMM  GT " << mVecAllSom.size() - aKS 
                    << " T=" <<  aTimeGT.uval()
                    << " TLA=" <<  mTimeLoadCple << " n=" << mNbLoadCple
                    << " TL3=" <<  mTimeLoadTri  << " n=" << mNbLoadTri
                    << " TLA+3=" <<  mTimeLoadTri + mTimeLoadCple
                    << " \n";
       }
       for( tItAGT itA=mVecAllSom[aKS]->begin(mSubAll) ; itA.go_on() ; itA++)
       {
             bool DoArc = (! mDebug) || (aKS==1);
             if (DoArc)
             {
                if (0) 
                {
                     std::cout << "  ARC GT "  << aTimeGT.uval() 
                               << " N1:" << (*itA).s1().attr().Im().Name()
                               << " N1:" << (*itA).s2().attr().Im().Name()
                               << " \n";
                }
                GenTriplet(*itA);
             }
       }
   }
   for (int aK=0 ; aK <2 ; aK++)
   {
       MakeFileXML(mTopoTriplets,mNM->NameTopoTriplet(aK==0));
   }

   cSauvegardeNamedRel aLCple;
   for (std::map<cCpleString,cListOfName>::const_iterator itCpl3=mTriOfCple.begin() ; itCpl3!=mTriOfCple.end() ; itCpl3++)
   {
       const cCpleString & aCple = itCpl3->first;
       cNewO_OneIm * aIm1 = &mMapS[aCple.N1()]->attr().Im();
       cNewO_OneIm * aIm2 = &mMapS[aCple.N2()]->attr().Im();
       for (int aKBin=0; aKBin<2; aKBin++)
       {
          std::string aNameTri = mNM->NameTripletsOfCple(aIm1,aIm2,(aKBin==0));
          MakeFileXML(itCpl3->second,aNameTri);

       }
       aLCple.Cple().push_back(aCple);
   }
   MakeFileXML(aLCple,mNM->NameCpleOfTopoTriplet(true));
   MakeFileXML(aLCple,mNM->NameCpleOfTopoTriplet(false));
  
   if (mShow)
      std::cout  << " Merge " << mTimeMerge << " Selec " << mTimeSelec << " GenTripl " << aTimeGT.uval() << "\n";
    if (0)
    {
        double aSomDiff=0;
        for (std::list<cXml_OneTriplet>::const_iterator it3=mTopoTriplets.Triplets().begin() ; it3!=mTopoTriplets.Triplets().end() ; it3++)
        {
            std::string aBase = "NewOriTmpQuick/";
            std::string aNameTri = aBase+it3->Name1() + "/" + it3->Name2() + "/Triplet-Ori0-" + it3->Name3() + ".dmp";

            cXml_Ori3ImInit aXmlNew = StdGetFromSI(aNameTri,Xml_Ori3ImInit);
            cXml_Ori3ImInit aXmlOld = StdGetFromSI("ORI-"+aNameTri,Xml_Ori3ImInit);

            double aDiff = ElAbs( aXmlNew.ResiduTriplet() - aXmlOld.ResiduTriplet());
            aDiff +=  euclid(aXmlNew.Ori2On1().Centre()-aXmlOld.Ori2On1().Centre());
            aDiff +=  euclid(aXmlNew.Ori3On1().Centre()-aXmlOld.Ori3On1().Centre());
            aDiff +=  euclid(aXmlNew.Ori2On1().Ori().L1()-aXmlOld.Ori2On1().Ori().L1());
            aDiff +=  euclid(aXmlNew.Ori3On1().Ori().L1()-aXmlOld.Ori3On1().Ori().L1());
            ELISE_ASSERT(aDiff<1e-5,"Check in SwapTriplet");
            aSomDiff+= aDiff;
        }
        cXml_TopoTriplet aOld3 = StdGetFromSI("ORI-NewOriTmpQuick/ListeTriplets.xml",Xml_TopoTriplet);
        ELISE_ASSERT(aOld3.Triplets().size() == mTopoTriplets.Triplets().size(),"Chexk Size Triple")

        std::cout << "SOM DIFF= " << aSomDiff << "\n";
    }
}

bool operator < (const tSomGT & aS1,const tSomGT & aS2)
{
    return aS1.attr().Name() < aS2.attr().Name();
}

cXml_Rotation El2Xml(const ElRotation3D & aRot)
{
  cXml_Rotation aRes;
  aRes.Centre() = aRot.tr();
  aRes.Ori() = ExportMatr(aRot.Mat());
  return aRes;
}

ElRotation3D Xml2El(const cXml_Rotation & aXml)
{
  return ElRotation3D(aXml.Centre(),ImportMat(aXml.Ori()),true);
}

ElRotation3D Xml2ElRot(const cXml_O2IRotation & aXml)
{
  return ElRotation3D(aXml.Centre(),ImportMat(aXml.Ori()),true);
}



void AddSegOfRot(std::vector<Pt3dr> & aV1,std::vector<Pt3dr> & aV2,const ElRotation3D & aR,const Pt2df &  aP)
{
   aV1.push_back(aR.ImAff(Pt3dr(0,0,0)));
   aV2.push_back(aR.ImAff(Pt3dr(aP.x,aP.y,1.0)));
}

double Residu(cNewO_OneIm  * anIm , const ElRotation3D & aR,const Pt3dr & aPTer,const Pt2df & aP)
{
    Pt3dr aQ = aR.ImRecAff(aPTer);
    Pt2df aProj (aQ.x/aQ.z,aQ.y/aQ.z);
    double aD = euclid(aProj,aP);
    return aD * anIm->CS()->Focale();
}

bool cAppli_GenTriplet::AddTriplet(tSomGT & aS1Ori,tSomGT & aS2Ori,tSomGT & aS3Ori)
{
   cTplTripletByRef<tSomGT> aTBR(aS1Ori,aS2Ori,aS3Ori);
   const cGTrip_AttrSom & aA1 = aTBR.mV0->attr();
   const cGTrip_AttrSom & aA2 = aTBR.mV1->attr();
   const cGTrip_AttrSom & aA3 = aTBR.mV2->attr();

   ELISE_ASSERT(aA1.Name() < aA2.Name(),"cAppli_GenTriplet::AddTriplet");
   ELISE_ASSERT(aA2.Name() < aA3.Name(),"cAppli_GenTriplet::AddTriplet");


   ElRotation3D aR1Inv = aA1.R3().inv();

   ElRotation3D aR2 = aR1Inv*aA2.R3();
   ElRotation3D aR3 = aR1Inv*aA3.R3();

   double aD = euclid(aR2.tr());
   aR2 = ElRotation3D(aR2.tr()/aD,aR2.Mat(),true);
   aR3 = ElRotation3D(aR3.tr()/aD,aR3.Mat(),true);


   double aResidu=-1;
   int    aNbTriplet=-1;
   if (true)
   {
      ElRotation3D aR1 = aR1Inv*aA1.R3();
      static std::vector<Pt2df> aVP1;
      static std::vector<Pt2df> aVP2;
      static std::vector<Pt2df> aVP3;


       bool OK = NM().LoadTriplet
                (
                     &aA1.Im(),&aA2.Im(), &aA3.Im(),
                     &aVP1,&aVP2,&aVP3
                );
       ELISE_ASSERT(OK,".LoadTriplet");

       std::vector<double> aVRes;
       for (int aK=0 ; aK< int(aVP1.size()) ; aK++)
       {
           std::vector<Pt3dr> aW1;
           std::vector<Pt3dr> aW2;
           AddSegOfRot(aW1,aW2,aR1,aVP1[aK]);
           AddSegOfRot(aW1,aW2,aR2,aVP2[aK]);
           AddSegOfRot(aW1,aW2,aR3,aVP3[aK]);
           bool OkI;
           Pt3dr aI = InterSeg(aW1,aW2,OkI);
           if (OkI)
           {
              double aRes1 = Residu(&aA1.Im(),aR1,aI,aVP1[aK]);
              double aRes2 = Residu(&aA2.Im(),aR2,aI,aVP2[aK]);
              double aRes3 = Residu(&aA3.Im(),aR3,aI,aVP3[aK]);
              aVRes.push_back((aRes1+aRes2+aRes3)/3.0);

           }
       }
       aResidu = MedianeSup(aVRes);
       aNbTriplet = (int)aVP1.size();
   }

   cTripletInt aTr(aA1.Num(),aA2.Num(),aA3.Num());
   bool aNewTriplet = false;
   {
      std::map<cTripletInt,cResTriplet>::iterator  itM = mMapTriplets.find(aTr) ;

      aNewTriplet =  (itM == mMapTriplets.end());
      if (  (!aNewTriplet) && (itM->second.mXml.ResiduTriplet() < aResidu))
      {
         return false;
      }
   }




   cResTriplet aRT;
   aRT.mXml.Ori2On1() = El2Xml(aR2);
   aRT.mXml.Ori3On1() = El2Xml(aR3);
   aRT.mXml.ResiduTriplet() = aResidu;
   aRT.mXml.NbTriplet() = aNbTriplet;

   for (int aK=0 ; aK <2 ; aK++)
   {
      std::string aNameTri = mNM->NameOriInitTriplet((aK==0),&(aA1.Im()),&(aA2.Im()),&(aA3.Im()));
      MakeFileXML(aRT.mXml,aNameTri);
   }

   mMapTriplets[aTr] =  aRT;

   cXml_OneTriplet aTri;
   aTri.Name1() = aA1.Name();
   aTri.Name2() = aA2.Name();
   aTri.Name3() = aA3.Name();
   if (aNewTriplet)
   {
      mTopoTriplets.Triplets().push_back(aTri);
      cCpleString aCple(aTri.Name1(),aTri.Name2());
      mTriOfCple[aCple].Name().push_back(aTri.Name3());
   }

   return true;
}


void  AddPackToMerge(CamStenope * aCS1,CamStenope * aCS2,const ElPackHomologue & aPack,cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> >&   aMap,int aInd0)
{
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
        Pt2dr aP1 = aCS1->F2toPtDirRayonL3(itP->P1());
        Pt2dr aP2 = aCS2->F2toPtDirRayonL3(itP->P2());
        Pt2df aQ1(aP1.x,aP1.y);
        Pt2df aQ2(aP2.x,aP2.y);
        // if (aSwap) ElSwap(aQ1,aQ2);
        aMap.AddArc(aQ1,aInd0,aQ2,1-aInd0,cCMT_NoVal());
    }
}


cAppli_GenTriplet::cAppli_GenTriplet(int argc,char ** argv) :
    mGrT        (),
    mFlagVois   (mGrT.alloc_flag_som()),
    mCurS1      (0),
    mCurS2      (0),
    mSomTest3   (0),
    // m Show       (true),
    mDebug      (false),
    mTimeMerge  (0.0),
    mTimeSelec  (0.0),
    // mQuick      (true), 
    // mPrefHom    (""),
    // mExtName    (""),
    mRamAllowed (4e9),
    mKS0        (0)
{
   ElTimer aChronoLoad;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(mFullName,"Pattern", eSAM_IsPatFile),
        LArgMain() 
                   << EAM(mShow,"Show",true,"Show intermediary message")
                   << EAM(mNameTest1,"Test1",true,"Name of first test image", eSAM_IsExistFile)
                   << EAM(mNameTest2,"Test2",true,"Name of second test image", eSAM_IsExistFile)
                   << EAM(mNameTest3,"Test3",true,"Name of second test image", eSAM_IsExistFile)
                   << EAM(mDebug,"Debug",true,"Debug .... tuning purpose .... Def=false", eSAM_IsBool)
                   << EAM(mKS0,"KS0",true,"Tuning Def=0", eSAM_IsBool)
                   << ArgCMA()
   );
   
   mSelAll = (ModeNO() == eModeNO_TTK);


   if (MMVisualMode) return;

   mEASF.Init(mFullName);
   StdCorrecNameOrient(mNameOriCalib,mEASF.mDir);

   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,mEASF.mDir,mNameOriCalib,"dat");

   cInterfChantierNameManipulateur::tSet  aVIm = *(mEASF.SetIm());
   std::sort(aVIm.begin(),aVIm.end());
   for (int aKIm=0 ; aKIm<int(aVIm.size());  aKIm++)
   {
        const std::string & aName = (aVIm)[aKIm];
        tSomGT & aS = mGrT.new_som(cGTrip_AttrSom(aKIm,aName,*this));
        if ((aName==mNameTest1) || (aName==mNameTest2))
        {
            aS.attr().SomTest() = true;
        }
        else if (aName==mNameTest3)
             mSomTest3 = & aS;
        mVecAllSom.push_back(&aS);
        mMapS[aName] = &aS;
   }


   const cInterfChantierNameManipulateur::tSet *  aSetCple =  mEASF.mICNM->Get(mNM->KeySetCpleOri());
   std::string aKeyCple2I = mNM->KeyAssocCpleOri();


   for (int aKC=0 ; aKC<int(aSetCple->size());  aKC++)
   {
        std::pair<std::string,std::string> aPair = mEASF.mICNM->Assoc2To1(aKeyCple2I,(*aSetCple)[aKC],false);
        std::string aN1 = aPair.first;
        std::string aN2 = aPair.second;
        ELISE_ASSERT(aN1<aN2,"Order in Hom");

        tSomGT * aS1 = mMapS[aN1];
        tSomGT * aS2 = mMapS[aN2];
        if (aS1 && aS2)
        {
           if (!  mGrT.arc_s1s2(*aS1,*aS2))
           {
               cXml_Ori2Im aXmlO = StdGetFromSI(mEASF.mDir+(*aSetCple)[aKC],Xml_Ori2Im);
               if (aXmlO.Geom().IsInit())
               {
                  const cXml_O2IRotation & aXO = aXmlO.Geom().Val().OrientAff();
                  ElRotation3D aR(aXO.Centre(),ImportMat(aXO.Ori()),true);
                  cGTrip_AttrASym * anASym  =  new  cGTrip_AttrASym(aXmlO);
                  anASym->ArcTest() = (aS1->attr().SomTest() && aS2->attr().SomTest());

                  tArcGT  * anArc = &( mGrT.add_arc(*aS1,*aS2,cGTrip_AttrArc(aR,anASym,true),cGTrip_AttrArc(aR.inv(),anASym,false)));

                  if (aKC%2) anArc = &(anArc->arc_rec());
               }

           }
        }
        if (((aKC%50)==0) && mShow)
        {
           std::cout << "AAAAAAAAAAAA " << aSetCple->size() - aKC << "\n";
        }
   }
   std::cout << "TIME LOAD " << aChronoLoad.uval() << "\n";
}


int GenTriplet_main(int argc,char ** argv)
{
   cAppli_GenTriplet anAppli(argc,argv);
   anAppli.GenTriplet();
   return EXIT_SUCCESS;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant a  la mise en
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
