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

static const int  TMaxNbCase = TNbCaseP1 * TNbCaseP1;
static const int  TMaxGain = 2e9;



class cNewSom_GenTripletOfCple;
class cNewAppli_GenTripletOfCple;


struct cNSGT_Link
{
    public :
          cNSGT_Link(cNewAppli_GenTripletOfCple &,const std::string & aN3);
};


class cNewSom_GenTripletOfCple
{
    public :
         cNewSom_GenTripletOfCple(cNewAppli_GenTripletOfCple &,const std::string & aN3);
    private :
          cNewAppli_GenTripletOfCple * mAppli;
          std::string                  mN3;
          cNewO_OneIm *       mI3;
          int mNb[TMaxNbCase];
          int mDens[TMaxNbCase];
          int mGain[TMaxNbCase];
          int mGainGlob;

};

struct cGTC_OneBaseIm
{
    public :

           cGTC_OneBaseIm(const ElRotation3D &,cNewO_OneIm * anIm,const std::vector<Pt2df> & aVP);

           ElRotation3D                mOri;
           cNewO_OneIm *               mIm;
           const std::vector<Pt2df>  * mVPts;

           Pt2df               mPInf;
           Pt2df               mPSup;
           double              mStepCases;
           Pt2di               mSzCases;
           int                 mNbCases;
           double              mMulQuant;
           int                 mPdsCase[TMaxNbCase];
           int                 mNb[TMaxNbCase];

           int ToIndexVP(const Pt2df &  aP0) const;
           void Init(const std::vector<Pt2df> &  aVP);
           void InitNb(int  * aNb,const std::vector<Pt2df> & );
};


class cNewAppli_GenTripletOfCple
{
      public :
           cNewAppli_GenTripletOfCple(int argc,char ** argv);
           cNewO_NameManager * NM() {return mNM;}
           // int ToIndexVP1(const Pt2df &  aP0) const;

      public :
           std::string mN1;
           std::string mN2;
           std::string mDir;
           bool        mQuick;
           std::string mPrefHom;
           std::string mExtName;
           std::string mNameOriCalib;


           cNewO_NameManager * mNM;
           cNewO_OneIm *       mI1;
           cNewO_OneIm *       mI2;
           std::vector<Pt2df>  mVP1;
           std::vector<Pt2df>  mVP2;
           cXml_Ori2Im         mXlm_Ori2Im;
           cXml_O2IComputed *  mXmlGeom;

           cGTC_OneBaseIm *         mBase1;
           cGTC_OneBaseIm *         mBase2;

           std::vector<cNewSom_GenTripletOfCple *>  mVS3;

 // Va permettre de creer des indexe pour avoir une fonction de densite tenant compte de la densite 
 // de VP1
/*
           Pt2df               mPInf;
           Pt2df               mPSup;
           double              mStepCases;
           Pt2di               mSzCases;
           int                 mNbCases;
           double              mMulQuant;
           int                 mPdsCase1[TMaxNbCase];
           int                 mPdsCase2[TMaxNbCase];
*/

};

/***************************************************************/
/*                                                             */
/*               cNewSom_GenTripletOfCple                      */
/*                                                             */
/***************************************************************/

cNewSom_GenTripletOfCple::cNewSom_GenTripletOfCple(cNewAppli_GenTripletOfCple & anAppli,const std::string & aN3) :
    mAppli  (&anAppli),
    mN3     (aN3),
    mI3     (new cNewO_OneIm(*(mAppli->NM()),mN3))
{
        std::cout << "NAME " << mN3 << "\n"; 
}

/***************************************************************/
/*                                                             */
/*               cGTC_OneBaseIm                                */
/*                                                             */
/***************************************************************/

int cGTC_OneBaseIm::ToIndexVP(const Pt2df &  aP0) const
{
    Pt2di aP =  round_down((aP0-mPInf)/mStepCases);
    aP.x  = ElMax(0,ElMin(mSzCases.x-1,aP.x));
    aP.y  = ElMax(0,ElMin(mSzCases.y-1,aP.y));

    int aRes = aP.x + aP.y * mSzCases.x;
    ELISE_ASSERT(aRes>=0 && aRes<mNbCases,"cAppli_GenTriplet::ToIndex");
    return aRes;
}

cGTC_OneBaseIm::cGTC_OneBaseIm(const ElRotation3D & anOri,cNewO_OneIm * anIm,const std::vector<Pt2df> & aVP):
   mOri  (anOri),
   mIm   (anIm),
   mVPts (&aVP)
{

   // Calcul des valeur permettant d'indexer les points
   mPInf = Pt2df(1e20,1e20);
   mPSup = Pt2df(-1e20,-1e20);

   for (int aKP=0 ; aKP<int(aVP.size()) ; aKP++)
   {
       const Pt2df & aPt = aVP[aKP];
       mPInf = Inf(mPInf,aPt);
       mPSup = Sup(mPSup,aPt);
   }

   Pt2df aPLarg = mPSup-mPInf;
   mStepCases = ElMax(aPLarg.x,aPLarg.y) / TNbCaseP1;
   mSzCases = Pt2di(round_up(aPLarg.x/mStepCases),round_up(aPLarg.y/mStepCases));
   mSzCases = Inf(mSzCases,Pt2di(TNbCaseP1,TNbCaseP1));
   mNbCases = mSzCases.x * mSzCases.y;
   ELISE_ASSERT(mNbCases<=TMaxNbCase,"cNewAppli_GenTripletOfCple::cNewAppli_GenTripletOfCple  : mNbCases");

   // Le MulQuant est un majoration/estimation du maximum du gain quantifie, il est egal a
   //    NbCase  => car somme sur les case
   //  * TQuant  => mPdsCase
   //  * TQuant  => Densite
   //  * TQuant  => Gain
   //  * TQuantBsH  => Dans Gain B/H
   mMulQuant  = mNbCases *pow((float)TQuant,3) * TQuantBsH;
   ELISE_ASSERT(mMulQuant<TMaxGain,"cNewAppli_GenTripletOfCple::cNewAppli_GenTripletOfCple mMulQuant");

   InitNb(mNb,aVP);

   // Creation d'un ponderation quantifie prop a la racine carree
   // de la population

   int aNbMax = 0;
   for (int aK=0 ; aK<mNbCases ; aK++)
   {
        ElSetMax(aNbMax,mNb[aK]);
   }
   for (int aK=0 ; aK<mNbCases ; aK++)
   {
        double aPds = mNb[aK] / double(aNbMax);
        mPdsCase[aK] = round_ni(TQuant*sqrt(aPds));
   }

}


void cGTC_OneBaseIm::InitNb(int  * aNb,const std::vector<Pt2df> & aVP)
{
    for (int aK=0 ;  aK< TMaxNbCase ; aK++)
        aNb[aK] = 0;

    for (int aK=0 ; aK<int(aVP.size()) ; aK++)
    {
        aNb[ToIndexVP(aVP[aK])] ++;
    }
}



/***************************************************************/
/*                                                             */
/*               cNewAppli_GenTripletOfCple                    */
/*                                                             */
/***************************************************************/

/*
int cNewAppli_GenTripletOfCple::ToIndexVP1(const Pt2df &  aP0) const
{
    Pt2di aP =  round_down((aP0-mPInf)/mStepCases);
    aP.x  = ElMax(0,ElMin(mSzCases.x-1,aP.x));
    aP.y  = ElMax(0,ElMin(mSzCases.y-1,aP.y));

    int aRes = aP.x + aP.y * mSzCases.x;
    ELISE_ASSERT(aRes>=0 && aRes<mNbCases,"cAppli_GenTriplet::ToIndex");
    return aRes;
}
*/


cNewAppli_GenTripletOfCple::cNewAppli_GenTripletOfCple(int argc,char ** argv) :
    mDir        ("./"),
    mQuick      (true),
    mPrefHom    (""),
    mExtName    ("")
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mN1,"Image one", eSAM_IsExistFile)
                   << EAMC(mN2,"Image two", eSAM_IsExistFile),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration ", eSAM_IsExistDirOri)
                   << EAM(mDir,"Dir",true,"Directory, Def=./ ",eSAM_IsDir)
                   << EAM(mQuick,"Quick",true,"Quick version")
                   << EAM(mPrefHom,"PrefHom",true,"Prefix Homologous points, def=\"\"")
                   << EAM(mExtName,"ExtName",true,"User's added Prefix , def=\"\"")

   );

   if (mN1 > mN2)
   {
       ElSwap(mN1,mN2);
   }


   mNM = new cNewO_NameManager(mExtName,mPrefHom,mQuick,mDir,mNameOriCalib,"dat");

   mI1 = new cNewO_OneIm(*mNM,mN1);
   mI2 = new cNewO_OneIm(*mNM,mN2);
   mNM->LoadHomFloats(mI1,mI2,&mVP1,&mVP2);
   mXlm_Ori2Im = StdGetFromSI(mNM->NameXmlOri2Im(mI1,mI2,true),Xml_Ori2Im);
   mXmlGeom = mXlm_Ori2Im.Geom().PtrVal();

   ELISE_ASSERT(mXmlGeom!=0,"Unoriented Init in cNewAppli_GenTripletOfCple::cNewAppli_GenTripletOfCple");
   // mI1.mOri = ElRotation::Id;
   // mR2 = Xml2ElRot(mXmlGeom->OrientAff());


   mBase1 = new cGTC_OneBaseIm(ElRotation3D::Id,mI1,mVP1);
   mBase2 = new cGTC_OneBaseIm(Xml2ElRot(mXmlGeom->OrientAff()),mI2,mVP2);
   


   std::list<std::string >  aL3 = mNM->ListeCompleteTripletTousOri(mN1,mN2);
   for (std::list<std::string>::const_iterator it3=aL3.begin() ; it3!=aL3.end() ; it3++)
   {
       mVS3.push_back( new cNewSom_GenTripletOfCple(*this,*it3));
   }
}




int CPP_NewGenTriOfCple(int argc,char ** argv)
{
    cNewAppli_GenTripletOfCple anAppli(argc,argv);

    return EXIT_SUCCESS;
}


/*
cNewAppli_GenTriplet::cNewAppli_GenTriplet(int argc,char ** argv) 
    mQuick      (true), 
    mRamAllowed (4e9),
    mAllocSwap  (mRamAllowed),
*/


/*********************************************************/
/*                                                       */
/*               cGTrip_AttrSom                          */
/*                                                       */
/*********************************************************/


/*
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

      Pt3dr aC31 = aC3 / MedianeSup(aVL13);
      Pt3dr aC32 = aC2 + aV3Bis * MedianeSup(aVL23);
      mC3 = (aC31 + aC32) / 2.0;
      Pt3dr aC3I = aC3 *  MedianeSup(aVLByInt);

      // std::cout << "DDDDD " << euclid(mC3-aC3I)  << " " << aC3I << "\n";

     // Finalement on prefere celui par point multiple, qui n'est pas degenere en cas de sommets alignes
      mC3 = aC3I;
      mM3 =  NearestRotation((aR31.Mat() + aR31Bis.Mat())*0.5);



     // Calcul gain et densite

      int aGain1 = mAppli->GainBSurH( mAppli->CurS1(),aSom);
      int aGain2 = mAppli->GainBSurH( mAppli->CurS2(),aSom);
      int aGain = ElMin(aGain1,aGain2);


      InitNb(aVP1);
      int aNbC = mAppli->NbCases();
      int * aNbGlob =  mAppli->CurS1()->attr().mNb;
      int * aPdsGlob = mAppli->Pds();
      mGainGlob = 0;
      for (int aK=0 ; aK< aNbC ; aK++)
      {
           double aDens = double(mNb[aK]) / double(ElMax(1,aNbGlob[aK]));
           aDens = ElMin(1.0,aDens);
           aDens = (aDens * TAttenDens) / (aDens * TAttenDens +1) ;
           mDens[aK] = round_ni( (TQuant * aDens * (TAttenDens+1)) / TAttenDens);

           mGain[aK] = mDens[aK] * TQuant * aGain;
           mGainGlob +=  mGain[aK] * aPdsGlob[aK];

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
*/

/*
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
      }
}
*/

/*********************************************************/
/*                                                       */
/*              cAppli_GenTriplet                        */
/*                                                       */
/*********************************************************/


/*

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
   int aNbMaxTriplet = mQuick ? TQuickNbMaxTriplet : TStdNbMaxTriplet;
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

   if (aGainMax < (TGainSeuil * mMulQuant)) return 0;

   mVSomEnCourse.erase(mVSomEnCourse.begin()+aIndexRes);
   mVSomSelected.push_back(aRes);

   for (int aK=0 ; aK<int(mVSomEnCourse.size()) ; aK++)
   {
      tSomGT * aSom = mVSomEnCourse[aK];
      aSom->attr().UpdateCost(aSom,aRes);
   }

   return aRes;
}
*/


/*
void cAppli_GenTriplet::GenTriplet(tArcGT & anArc)
{
    if (!anArc.attr().IsDirASym() ) return;
    mCurArc = & anArc;



    std::vector<Pt2df> aVP1,aVP2;
    ElTimer aChrono;
    mNM->LoadHomFloats(&(anArc.s1().attr().Im()),&(anArc.s2().attr().Im()),&aVP1,&aVP2);
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
*/



/*
cNewAppli_GenTriplet::cNewAppli_GenTriplet(int argc,char ** argv) 
    mQuick      (true), 
    mRamAllowed (4e9),
    mAllocSwap  (mRamAllowed),
    mKS0        (0)
{
   ElTimer aChronoLoad;

   ElInitArgMain
   (
        argc,argv,
        LArgMain() <<  EAMC(mFullName,"Pattern", eSAM_IsPatFile),
        LArgMain() << EAM(mNameOriCalib,"OriCalib",true,"Orientation for calibration", eSAM_IsExistDirOri)
                   << EAM(m Show,"Show",true,"Show intermediary message")
                   << EAM(mNameTest1,"Test1",true,"Name of first test image", eSAM_IsExistFile)
                   << EAM(mNameTest2,"Test2",true,"Name of second test image", eSAM_IsExistFile)
                   << EAM(mNameTest3,"Test3",true,"Name of second test image", eSAM_IsExistFile)
                   << EAM(mQuick,"Quick",true,"Quick version", eSAM_IsBool)
                   << EAM(mDebug,"Debug",true,"Debug .... tuning purpose .... Def=false", eSAM_IsBool)
                   << EAM(mKS0,"KS0",true,"Tuning Def=0", eSAM_IsBool)
   );

   if (MMVisualMode) return;

   mEASF.Init(mFullName);
   StdCorrecNameOrient(mNameOriCalib,mEASF.mDir);

   mNM = new cNewO_NameManager(mQuick,mEASF.mDir,mNameOriCalib,"dat");

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

   // const cInterfChantierNameManipulateur::tSet *  aSetCple =  mEASF.mICNM->Get("NKS-Set-CplIm2OriRel@"+mNameOriCalib+"@dmp");
   // std::string aKeyCple2I = "NKS-Assoc-CplIm2OriRel@"+mNameOriCalib+"@dmp";

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
        if (((aKC%50)==0) && m Show)
        {
           std::cout << "AAAAAAAAAAAA " << aSetCple->size() - aKC << "\n";
        }
   }
   std::cout << "TIME LOAD " << aChronoLoad.uval() << "\n";
}


int GenTriplet#_#main(int argc,char ** argv)
{
   cAppli_GenTriplet anAppli(argc,argv);
   anAppli.GenTriplet();
   return EXIT_SUCCESS;
}
*/





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
