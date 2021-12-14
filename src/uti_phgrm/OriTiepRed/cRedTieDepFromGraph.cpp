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

#include "OriTiepRed.h"
NS_OriTiePRed_BEGIN

void Banniere_Ratafia()
{
    std::cout << " ***************************************************\n";
    std::cout << " *                                                 *\n";
    std::cout << " *       R-eduction                                *\n";
    std::cout << " *       A-utomatic of                             *\n"; 
    std::cout << " *       T-ie points                               *\n"; 
    std::cout << " *       A-iming to get                            *\n"; 
    std::cout << " *       F-aster                                   *\n"; 
    std::cout << " *       I-mage                                    *\n"; 
    std::cout << " *       A-erotriangulation                        *\n"; 
    std::cout << " *                                                 *\n";
    std::cout << " ***************************************************\n";
}

class cRealDecoupageInterv2D
{
     public :
          cRealDecoupageInterv2D(const Box2dr & aBox,const Pt2dr & aSzMax,const Box2dr & aBord);

      // Avec Bord par defaut
          Box2dr  KthIntervIn(int aK) const;
          Box2dr KthIntervOut(int aK) const;
          int NbInterv() const;


     private :
          Pt2di ToInt(const Pt2dr &aP,bool WithP0) const ;
          Box2di  ToInt(const Box2dr &aB,bool WithP0) const; 
          Pt2dr ToReal(const Pt2di &aP) const;
          Box2dr  ToReal(const Box2di &aB) const; 

          int mNbDec;
          Pt2dr mStep;
          Pt2dr mP0;
          cDecoupageInterv2D mIDec;
};


Pt2di cRealDecoupageInterv2D::ToInt(const Pt2dr &aP,bool WithP0) const 
{
   return round_ni( (aP-(WithP0 ? mP0: Pt2dr(0,0))).dcbyc(mStep));
}
Box2di  cRealDecoupageInterv2D::ToInt(const Box2dr &aB,bool WithP0) const 
{
   return Box2di(ToInt(aB._p0,WithP0),ToInt(aB._p1,WithP0));
}

Pt2dr cRealDecoupageInterv2D::ToReal(const Pt2di &aP) const 
{
   return mP0 + mStep.mcbyc(Pt2dr(aP));
}
Box2dr  cRealDecoupageInterv2D::ToReal(const Box2di &aB) const 
{
   return Box2dr(ToReal(aB._p0),ToReal(aB._p1));
}

int cRealDecoupageInterv2D::NbInterv() const
{
   return mIDec.NbInterv();
}
Box2dr  cRealDecoupageInterv2D::KthIntervIn(int aK) const
{
   return ToReal(mIDec.KthIntervIn(aK));
}
Box2dr cRealDecoupageInterv2D::KthIntervOut(int aK) const
{
   return ToReal(mIDec.KthIntervOut(aK));
}






cRealDecoupageInterv2D::cRealDecoupageInterv2D(const Box2dr & aBox,const Pt2dr & aSzMax,const Box2dr & aBord) :
    mNbDec (100000000),
    mStep  (aBox.sz() / mNbDec),
    mP0    (aBox._p0),
    mIDec  (ToInt(aBox,true),ToInt(aSzMax,false),ToInt(aBord,false))
{
}

    //  ===================== Histo =====================

template <class Type> void HistoAdd(std::vector<Type> & aVec,int anInd,const Type & aVal)
{
   for (int aK= int(aVec.size()) ; aK <= anInd ; aK++)
   {
       aVec.push_back(Type(0));
   }
   aVec[anInd] += aVal;
}

/*
class cStatArc
{
    public :
       void Add(int aNbSom,int aNbArc);
       cStatArc();
       void Show();
    private :
       std::vector<int> mVH;
       std::vector<int> mVHA;
       int              mNbP;
};
*/


cStatArc::cStatArc() :
   mNbP (0)
{
}


void cStatArc::Add(int aNbSom,int aNbArc)
{
   mNbP++;
   HistoAdd(mVH,aNbSom,1);
   HistoAdd(mVHA,aNbSom,aNbArc);
}
void cStatArc::Show()
{
    std::cout << "----------------------------- NbP=" << mNbP << " --------------------"<<std::endl;
    for (int aKH=0 ; aKH<int(mVH.size()) ; aKH++)
    {
         if (mVH[aKH])
            std::cout << " For muliplicity " << aKH 
                      << " %=" << ((100.0*mVH[aKH])/mNbP) 
                      << " N=" << mVH[aKH] 
                      << " D=" << (mVHA[aKH]/double(mVH[aKH])) 
                      << std::endl;
    }
}

/**********************************************************************/
/*                                                                    */
/*                         cAttArcSymGrRedTP                          */
/*                                                                    */
/**********************************************************************/

cAttArcSymGrRedTP::cAttArcSymGrRedTP(const cXml_Ori2Im & anOri) :
    mOri (anOri)
{
}

const cXml_Ori2Im & cAttArcSymGrRedTP::Ori() const { return mOri; }
std::vector<Pt2df> & cAttArcSymGrRedTP::VP1() { return mVP1; }
std::vector<Pt2df> & cAttArcSymGrRedTP::VP2() { return mVP2; }


/**********************************************************************/
/*                                                                    */
/*                         cAttArcASymGrRedTP                         */
/*                                                                    */
/**********************************************************************/

cAttArcASymGrRedTP::cAttArcASymGrRedTP(cAttArcSymGrRedTP * aASym,bool direct) :
   mASym   (aASym),
   mDirect (direct)
{
}

const cXml_Ori2Im & cAttArcASymGrRedTP::Ori() const
{
   return mASym->Ori();
}

double & cAttArcASymGrRedTP::Recouv() {return mRecouv;}

const Box2dr & cAttArcASymGrRedTP::Box() const
{
   return mDirect ? mASym->Ori().Box1() : mASym->Ori().Box2();
}

const double & cAttArcASymGrRedTP::Foc() const
{
   return mDirect ? mASym->Ori().Foc1() : mASym->Ori().Foc2();
}

std::vector<Pt2df> & cAttArcASymGrRedTP::VP1() { return mASym->VP1();}
std::vector<Pt2df> & cAttArcASymGrRedTP::VP2() { return mASym->VP2();}

/**********************************************************************/
/*                                                                    */
/*                         cV2ParGRT                                  */
/*                                                                    */
/**********************************************************************/

void cV2ParGRT::AddSom(tSomGRTP * aSom)
{
   mVSom.push_back(aSom);
}

const std::vector<tSomGRTP *> & cV2ParGRT::VSom() const
{
   return mVSom;
}

int cV2ParGRT::NumBox0()
{
  ELISE_ASSERT(mVSom.size(),"cV2ParGRT::NumBox0");
  return mVSom[0]->attr()->NumBox0();
}

int cV2ParGRT::NumBox1()
{
  ELISE_ASSERT(mVSom.size(),"cV2ParGRT::NumBox1");
  return mVSom.back()->attr()->NumBox0();
}

/**********************************************************************/
/*                                                                    */
/*                         cAttSomGrRedTP                             */
/*                                                                    */
/**********************************************************************/

cAttSomGrRedTP::cAttSomGrRedTP(cAppliGrRedTieP & anAppli,const std::string & aName) :
   mAppli     (&anAppli),
   mCamGlob   (0),
   mName      (aName),
   mNbPtsMax  (0),
   mRecSelec  (0.0),
   mRecCur    (0.0),
   mMTD       (cMetaDataPhoto::CreateExiv2(mName)),
   mFocPix    (anAppli.IsGBLike()? anAppli.DefFocPix() : mMTD.FocPix()),
   mFoc35     (anAppli.IsGBLike()? anAppli.DefFoc35() : mMTD.Foc35()),
   mSzDec     (anAppli.SzPixDec()/ mFocPix),
   mNumBox0   (-1),
   mNumBox1   (-1),
   mNumSom    (-1),
   mCalCam    (anAppli.IsGBLike()? nullptr : anAppli.NoNM()->CalibrationCamera(aName))
{
}

// const cMetaDataPhoto & cAttSomGrRedTP::MTD() const {return mMTD;}
double  cAttSomGrRedTP::FocPix() const
{
   return mFocPix;
}

double  cAttSomGrRedTP::Foc35()  const
{
  return mFoc35;
}


int & cAttSomGrRedTP::NbPtsMax() {return mNbPtsMax;}
const std::string & cAttSomGrRedTP::Name() const {return mName;}
double & cAttSomGrRedTP::RecSelec() {return mRecSelec;}
double & cAttSomGrRedTP::RecCur() {return mRecCur;}
Box2dr & cAttSomGrRedTP::BoxIm() {return mBoxIm;}
double  cAttSomGrRedTP::SzDec() const {return mSzDec;}
int & cAttSomGrRedTP::NumBox0() {return mNumBox0;}
int & cAttSomGrRedTP::NumBox1() {return mNumBox1;}
int & cAttSomGrRedTP::NumSom() {return mNumSom;}

Pt2dr cAttSomGrRedTP::Hom2Cam(const Pt2df & aP) const
{
  Pt3dr aQ(aP.x,aP.y,1.0);
  return mCalCam->L3toF2(aQ);
}

/**********************************************************************/
/*                                                                    */
/*                         cAppliGrRedTieP                            */
/*                                                                    */
/**********************************************************************/

void cAppliGrRedTieP::SetSelected(tSomGRTP * aSom)
{

    aSom->flag_set_kth_true(mFlagSel);
    aSom->flag_set_kth_true(mFlagCur);
    mPartParal.back()->AddSom(aSom);
    mFiloCur.pushlast(aSom);
    for (tIterArcGRTP  itA=aSom->begin(mSubAll) ; itA.go_on(); itA++)
    {
        tSomGRTP & aS2 = (*itA).s2();
        double aRec = (*itA).attr()->Recouv();
        ElSetMax(aS2.attr()->RecCur(),aRec);
        ElSetMax(aS2.attr()->RecSelec(),aRec);
    }

}



tSomGRTP * cAppliGrRedTieP::GetNextBestSom()
{
   mPcc.pcc (
                mFiloCur,
                mSubNone,
                mSubAll,   // pds => 1.0 par defaut, donc cela convient
                eModePCC_Somme
            );

   tSomGRTP * aRes = 0;

   for (int aKs=0 ; aKs<int(mVSom.size()) ; aKs++)
   {
       tSomGRTP * aSom = mVSom[aKs];
/*
       std::cout << " NNN=" << aSom->attr()->Name() 
                 << " P=" << mPcc.pds(*aSom)  
                 << " Rc=" << aSom->attr()->RecCur() 
                 << " S=" << aSom->flag_kth(mFlagSel)
                 << "\n";
*/
       if (! aSom->flag_kth(mFlagSel))
       {
           if (aRes==0)
           {
              aRes = aSom;
           }
           else
           {
              if (aSom->attr()->RecCur() < aRes->attr()->RecCur())
              {
                 aRes = aSom;
              }
              else if (    (aSom->attr()->RecCur() == aRes->attr()->RecCur())
                        && ( mPcc.pdsDef(*aSom) <  mPcc.pdsDef(*aRes))
                      )
              {
                 aRes = aSom;
              }
           }
       }
   }

   if ((aRes!=0)  && (aRes->attr()->RecCur() > mRecMax))
   {
      aRes = 0;
   }

   return aRes;
}



bool cAppliGrRedTieP::OneItereSelection()
{

     mFiloCur.clear();
     // Pour S0, on prend le non selec le + proche (recouvert) des deja selec
     tSomGRTP * aS0 = 0;
     for (int aKs=0 ; aKs<mNbSom ; aKs++)
     {
        tSomGRTP * aSom = mVSom[aKs];
        aSom->attr()->RecCur() = 0;
        if (
                (! aSom->flag_kth(mFlagSel))  
             && ((aS0==0)|| (aSom->attr()->RecSelec() > aS0->attr()->RecSelec()))
           )
        {
            aS0 = aSom;
        }
     }


     if (aS0==0) 
       return false;


    cV2ParGRT * aSetPart = new cV2ParGRT;
    mPartParal.push_back(aSetPart);
    SetSelected(aS0);

    // recherche de l'ensemble courant
    const std::vector<tSomGRTP *> & aVS = aSetPart->VSom();

    bool ContAdd = true;
    while (ContAdd)
    {
       if (mNbP <= int (aVS.size()  ))
       {
          ContAdd = false;
       }
       else
       {
          tSomGRTP * aSom =  GetNextBestSom();
          if (aSom)
          {
              SetSelected(aSom);
          }
          else
          {
             ContAdd = false;
          }
       }
    }


    // On efface la selection courante
    for (int aKs=0 ; aKs<int(aVS.size()) ; aKs++)
    {
        aVS[aKs]->flag_set_kth_false(mFlagCur);
    }
    
    //
    return true;
}

void cAppliGrRedTieP::Show()
{
    double aRecGr = 0.0;
    for (int aKP=0; aKP<int(mPartParal.size()) ; aKP++)
    {
        double aRecPart = 0.0;
        const std::vector<tSomGRTP *> & aVS = mPartParal[aKP]->VSom();
        if (aKP!=0) std::cout << "**********************************************************\n";
        
        for (int aKS1=0 ; aKS1<int(aVS.size()) ; aKS1++)
        {
            double aRec = 0.0;
            for (int aKS2=0 ; aKS2<int(aVS.size()) ; aKS2++)
            {
               tArcGRTP * anArc = mGr.arc_s1s2(*(aVS[aKS1]),(*aVS[aKS2]));

               if (anArc)
               {
                  aRec += anArc->attr()->Recouv();
               }
            }
            std::cout << "   " << aVS[aKS1]->attr()->Name() << " R=" << aRec << "\n";
            aRecPart += aRec;
        }
        aRecGr += aRecPart;
    }
    std::cout 
              << "NbCl " << mPartParal.size()
              << " Rec=" << aRecGr
              << std::endl;
   getchar();
}

double  cAppliGrRedTieP::SzPixDec() const
{
   return mSzPixDec;
}


        // ------------------ Box creation ------------

std::string cAppliGrRedTieP::ComOfKBox(int aKBox)
{

   return MM3dBinFile("OriRedTieP") + " " + '"' + mPatImage + '"'   // Giang : add "" around Pattern image to avoid error of expression type "epoque1_[a|b|c]|epoque1_d.*.tif"
             + " OriCalib=" + mCalib 
             + " KBox=" + ToString(aKBox)
             + " DistPMul=" + ToString(mDistPMul) 
             + " MVG=" + ToString(mMulVonGruber)
             + " DCA=" + ToString(mDoCompleteArc)
             + " UseP=" + ToString(mUsePrec)
             + " SH=" + mSH
          ;
}


void cAppliGrRedTieP::CreateBoxOfSom(tSomGRTP *aSom)
{
    cAttSomGrRedTP & anAtr =  *(aSom->attr());
    anAtr.NumBox0() = mNumBox;
    Box2dr aBox = anAtr.BoxIm();
    Pt2dr aNbR = aBox.sz() / anAtr.SzDec();
    Pt2di aNbI = round_up(aNbR);
  

    if (anAtr.Foc35() < 20.0)  // Risque de Fish eye envoyant a l'infini
    {
        aNbI = ElMin(Pt2di(2,2),aNbI);
    }

    // cDecoupageInterv2D
    double aSzD = anAtr.SzDec();
    double aRab = aSzD / 50.0;
    Pt2dr aPRab(aRab,aRab);



    cRealDecoupageInterv2D aRDec(aBox,Pt2dr(aSzD,aSzD),Box2dr(-aPRab,aPRab));


    for (int aKI=0 ; aKI< aRDec.NbInterv() ; aKI++)
    {
         cXml_ParamBoxReducTieP aXPB;
         aXPB.Box() = aRDec.KthIntervOut(aKI);
         aXPB.BoxRab() = aRDec.KthIntervIn(aKI);
         aXPB.MasterIm().SetVal(anAtr.Name());
         aXPB.Ims().push_back(anAtr.Name());

         for (tIterArcGRTP  itA=aSom->begin(mSubAll) ; itA.go_on(); itA++)
         {
              aXPB.Ims().push_back((*itA).s2().attr()->Name());
         }
         // std::cout << "BOXXX=" << mAppliTR->NameParamBox(mNumBox,true) << "\n";

         MakeFileXML(aXPB,mAppliTR->NameParamBox(mNumBox,true));
         MakeFileXML(aXPB,mAppliTR->NameParamBox(mNumBox,false));

         if (mTestExeOri)
         {
             std::cout <<  "GGG " << ComOfKBox(mNumBox) << std::endl;
             System(ComOfKBox(mNumBox));
         }
         mNumBox++;
    }

    anAtr.NumBox1() = mNumBox;
    // getchar();


}
void cAppliGrRedTieP::CreateBoxOfSet(cV2ParGRT * aSet)
{
    const std::vector<tSomGRTP *> & aVS = aSet->VSom();

    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
         CreateBoxOfSom(aVS[aKS]);
    }
}

void cAppliGrRedTieP::CreateBox()
{
    for (int aKP=0; aKP<int(mPartParal.size()) ; aKP++)
    {
        CreateBoxOfSet(mPartParal[aKP]);
    }
}

        // ------------------ Execution of command ------------

void cAppliGrRedTieP::FusionSelec(cAttSomGrRedTP& anAtr1,cAttSomGrRedTP& anAtr2,int aKBox)
{
    const std::string & aN1 = anAtr1.Name();
    const std::string & aN2 = anAtr2.Name();
    if (aN1>=aN2)
       return;
    std::string aNameHomBox =  mAppliTR->NameHomol(aN1,aN2,aKBox);

    bool Exist = ELISE_fp::exist_file(aNameHomBox);
    if (! Exist) return;


    std::string aNameHomGlob = mAppliTR->NameHomolGlob(aN1,aN2);

    std::vector<Pt2df> aVPGlob1,aVPGlob2;
    
    if (ELISE_fp::exist_file(aNameHomGlob))
    {
         mNoNM->GenLoadHomFloats(aNameHomGlob,&aVPGlob1,&aVPGlob2,false);
    }

    std::vector<Pt2df> aVPLoc1,aVPLoc2;
    mNoNM->GenLoadHomFloats(aNameHomBox,&aVPLoc1,&aVPLoc2,false);

    for (int aK=0 ; aK<int(aVPLoc1.size()) ; aK++)
    {
        aVPGlob1.push_back(aVPLoc1[aK]);
        aVPGlob2.push_back(aVPLoc2[aK]);
    }
    tCVUI1 aVNb(aVPGlob1.size(),2);

    mNoNM->WriteCouple(aNameHomGlob,aVPGlob1,aVPGlob2,aVNb);

    ELISE_fp::RmFile(aNameHomBox);
}

void cAppliGrRedTieP::ExeSelecOfSet(cV2ParGRT * aPart)
{
    const std::vector<tSomGRTP *> & aVS = aPart->VSom();


    // Execution des commandes individuelles
    std::list<std::string> aLCom;
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
        cAttSomGrRedTP & anAttr = *(aVS[aKS]->attr());
        for (int aKB=anAttr.NumBox0() ; aKB<anAttr.NumBox1() ; aKB++)
        {
               std::string aCom = ComOfKBox(aKB);

               aLCom.push_back(aCom);
        }
    }
    if (mInParal)
        cEl_GPAO::DoComInParal(aLCom);
    else
        cEl_GPAO::DoComInSerie(aLCom);


    // Fusion des pts homologues
    for (int aKS=0 ; aKS<int(aVS.size()) ; aKS++)
    {
        tSomGRTP * aSom = aVS[aKS];
        cAttSomGrRedTP & anAttr0 = *(aSom->attr());
        for (int aKB=anAttr0.NumBox0() ; aKB<anAttr0.NumBox1() ; aKB++)
        {
            for (tIterArcGRTP  itA1=aSom->begin(mSubAll) ; itA1.go_on(); itA1++)
            {
                cAttSomGrRedTP & anAttr1 = *(itA1->s2().attr());
                FusionSelec(anAttr0,anAttr1,aKB);
                FusionSelec(anAttr1,anAttr0,aKB);
                for (tIterArcGRTP  itA2=aSom->begin(mSubAll) ; itA2.go_on(); itA2++)
                {
                    cAttSomGrRedTP & anAttr2 = *(itA2->s2().attr());
                    FusionSelec(anAttr1,anAttr2,aKB);
                }
            }
        }
    }
}
   



void cAppliGrRedTieP::ExeSelec()
{
    for (int aKP=0; aKP<int(mPartParal.size()) ; aKP++)
    {
        ExeSelecOfSet(mPartParal[aKP]);
        std::cout << "=======    Done " << aKP <<  "  Part on " << mPartParal.size() << " ================" << std::endl;
    }
}


void cAppliGrRedTieP::DoExport()
{
    mMergeStruct = new tMergeStrRat(mNbSom,true);

    // Lit les hom et les met ds une structur de merge
    // On passe par la structure merge pour supprimer d'eventuelles incoherences
    for (int aKS=0 ; aKS<int(mVSom.size()) ; aKS++)
    {
        tSomGRTP & aS1 = *(mVSom[aKS]);
        const std::string & aN1 = aS1.attr()->Name();
        int aI1 = aS1.attr()->NumSom();
        for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
        {
            tSomGRTP & aS2 = (*itA).s2();
            const std::string & aN2 = aS2.attr()->Name();
            if (aN1 < aN2)
            {
                int aI2 = aS2.attr()->NumSom();
                std::string aNameHomGlob = mAppliTR->NameHomolGlob(aN1,aN2);
                if (ELISE_fp::exist_file(aNameHomGlob))
                {
                    std::vector<Pt2df> aVP1,aVP2;
                    mNoNM->GenLoadHomFloats(aNameHomGlob,&aVP1,&aVP2,false);

                    for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
                    {
                        mMergeStruct->AddArc(aVP1[aKP],aI1,aVP2[aKP],aI2,cCMT_NoVal());
                    }
                }
            }
        }
    }

    // Passe de du merge au point homologue flottant
    mMergeStruct->DoExport();
    const std::list<tMergeRat *> & aLMerge =  mMergeStruct->ListMerged();

    cStatArc  aStatA;
    for (std::list<tMergeRat *>::const_iterator itM=aLMerge.begin() ; itM!=aLMerge.end() ; itM++)
    {
        if ((mProbaSel>1.0) ||  (NRrandom3() < mProbaSel))
        {
            std::vector<Pt2di>  aVP = (*itM)->Edges();
            int aNbS = (*itM)->NbSom();
            aStatA.Add(aNbS,int(aVP.size()));
            for (int aKCple=0 ; aKCple<int(aVP.size()) ; aKCple++)
            {
               // Histo
               int aKC1 = aVP[aKCple].x;
               int aKC2 = aVP[aKCple].y;
               if (aKC1 >aKC2)
               {
                   ElSwap(aKC1,aKC2);
               }
               tSomGRTP & aS1 = *(mVSom[aKC1]);
               tArcGRTP * anA12 = 0;
               for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
               {
                    if ((*itA).s2().attr()->NumSom() == aKC2)
                    {
                        anA12 = &(*itA);
                    }
               }
               ELISE_ASSERT(anA12!=0,"Canno get arc in cAppliGrRedTieP::DoExport\n");
               anA12->attr()->VP1().push_back((*itM)->GetVal(aKC1));
               anA12->attr()->VP2().push_back((*itM)->GetVal(aKC2));
            }
        }
    }

    aStatA.Show();

    std::string aKeyH = "NKS-Assoc-CplIm2Hom@"+ mOut + "@dat";
    // On genere l'export 
    for (int aKS=0 ; aKS<int(mVSom.size()) ; aKS++)
    {
        tSomGRTP & aS1 = *(mVSom[aKS]);
        int aI1 = aS1.attr()->NumSom();
        const std::string & aN1 = aS1.attr()->Name();
        for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
        {
            tSomGRTP & aS2 = (*itA).s2();
            int aI2 = aS2.attr()->NumSom();
            const std::string & aN2 = aS2.attr()->Name();
            if (aI1 < aI2)
            {
                std::vector<Pt2df> & aVP1 = (*itA).attr()->VP1();
                std::vector<Pt2df> & aVP2 = (*itA).attr()->VP2();

                ElPackHomologue aPack12;
                ElPackHomologue aPack21;

                for (int aKp=0 ; aKp<int(aVP1.size()) ; aKp++)
                {
                    Pt2dr aP1 = aS1.attr()->Hom2Cam(aVP1[aKp]);
                    Pt2dr aP2 = aS2.attr()->Hom2Cam(aVP2[aKp]);
                    aPack12.Cple_Add(ElCplePtsHomologues(aP1,aP2,1.0));
                    aPack21.Cple_Add(ElCplePtsHomologues(aP2,aP1,1.0));
                }

                aPack12.StdPutInFile(mICNM->Assoc1To2(aKeyH,aN1,aN2,true));
                aPack21.StdPutInFile(mICNM->Assoc1To2(aKeyH,aN2,aN1,true));
            }
        }
    }
}

cVirtInterf_NewO_NameManager * cAppliGrRedTieP::NoNM()
{
   return mNoNM;
}

cAppliGrRedTieP::cAppliGrRedTieP(int argc,char ** argv) :
    mIntOrLevel      (eLevO_ByCple),
    mQuick           (true),
    mSH              (""),
    mGBLike          (false),
    mDefFocPix       (10000),
    mDefFoc35        (100),
    mNbP             (-1),
    mFlagSel         (mGr.alloc_flag_som()),
    mFlagCur         (mGr.alloc_flag_som()),
    mRecMax          (DefRecMaxModeIm),
    mShowPart        (false),
    mAppliTR         (0),
    mSzPixDec        (10000),
    mNumBox          (0),
    mTestExeOri      (false),
    mMergeStruct     (NULL),
    mOut             ("-Ratafia"),
    mDistPMul        (200.0),
    mMulVonGruber    (2),
    mLimFullTieP     (3),
    mInParal         (true),
    mDoCompleteArc   (false), // Pour l'instant comprends pas pourquoi cela genere probleme dans Tapas ?
    mUsePrec         (true),
    mProbaSel        (10.0)
{
   // Read parameters 
   MMD_InitArgcArgv(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile),
         LArgMain()  << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
                     << EAM(mIntOrLevel,"LevelOR",true,"Level Or, 0=None,1=Pair,2=Glob, (Def=1 or 0 (if GB))")
                     << EAM(mGBLike,"GBLike",true,"Generik bundle or like, dont read focs")
                     << EAM(mNbP,"NbP",true,"Nb Process, def use all")
                     << EAM(mRecMax,"RecMax",true,"Max overlap acceptable in two parallely processed images")
                     << EAM(mShowPart,"ShowP",true,"Show Partition (def=false)")
                     << EAM(mSzPixDec,"SzPixDec",true,"Sz of decoupe in pixel")
                     << EAM(mTestExeOri,"TEO",true,"Test Execution OriRedTieP ()")
                     << EAM(mOut,"Out",true,"Folder dest => Def=-Ratafia")
                     << EAM(mDistPMul,"DistPMul",true,"Average distance in pixels between 2 Tie points, def=200")
                     << EAM(mMulVonGruber,"MVG",true,"Multiplier VonGruber, Def=" + ToString(mMulVonGruber))
                     << EAM(mInParal,"Paral",true,"Do it in parallel" )
                     << EAM(mDoCompleteArc,"DCA",true,"Do Complete Arc (Def=false)")
                     << EAM(mUsePrec,"UseP",true,"Use prec to avoid redundancy (Def=true), tuning only")
                     << EAM(mProbaSel,"ProbaSel",true,"tuning only, generate a random selection at the end")
                     << EAM(mSH,"SH",true,"Homol Prefix , Def=\"\"")

   );

   if ((!EAMIsInit(&mIntOrLevel))  && mGBLike)
      mIntOrLevel = 0;
   mOrLevel = (eLevelOr) mIntOrLevel;
   mUseOr = (mOrLevel>=eLevO_ByCple);

   {
      std::string aComATR = /* MM3dBinFile("OriRedTieP")i*/ + " " + mPatImage + " FromRG=true";
      if (EAMIsInit(&mCalib))
      {
         aComATR = aComATR + " OriCalib=" + mCalib ;
      }
      std::vector<char *> * aVA = new std::vector<char *>(ToArgMain(aComATR));
      mAppliTR =  new cAppliTiepRed(aVA->size(),&((*aVA)[0]),true);
   }
  // cAppliTiepRed::cAppliT
 // std::vector<char *> ToArgMain(const std::string & aStr);


    cElemAppliSetFile::Init(mPatImage);
    //mNoNM = cVirtInterf_NewO_NameManager::StdAlloc(mDir,mCalib,mQuick);
    mNoNM = cVirtInterf_NewO_NameManager::StdAlloc(mSH,mDir,mCalib,mQuick);

    if (!EAMIsInit(&mNbP))
    {
       mNbP = MMNbProc();
    }

   // Creation des sommets du graphe
    for (int aK=0 ; aK<int(mSetIm->size()) ; aK++)
    {
        const std::string & aName = (*mSetIm)[aK];
        tSomGRTP & aSom = mGr.new_som(new cAttSomGrRedTP(*this,aName));
        mDicoSom[aName] = &aSom;
        mVSom.push_back(&aSom);
        aSom.attr()->NumSom() = aK;
    }
    mNbSom = mVSom.size();

   // Creation des arcs du graphe
    std::string aNameCple =  mUseOr                                ?
                             mNoNM->NameListeCpleOriented(true)    :
                             mNoNM->NameListeCpleConnected(true)   ;

    cSauvegardeNamedRel  aLCple = StdGetFromPCP(aNameCple,SauvegardeNamedRel);

    ELISE_ASSERT(aLCple.Cple().size() !=0,"No hom from Martini in Ratafia");

    for (std::vector<cCpleString>::const_iterator itCp=aLCple.Cple().begin() ; itCp!=aLCple.Cple().end() ; itCp++)
    {
        tSomGRTP * aS1 = mDicoSom[itCp->N1()];
        tSomGRTP * aS2 = mDicoSom[itCp->N2()];

        if ((aS1!=0) && (aS2!=0))
        {
             cXml_Ori2Im anOri = mNoNM->GetOri2Im(itCp->N1(),itCp->N2());
             cAttArcSymGrRedTP * aAASym = new cAttArcSymGrRedTP(anOri);
             mGr.add_arc(*aS1,*aS2,new cAttArcASymGrRedTP(aAASym,true),new cAttArcASymGrRedTP(aAASym,false));

        }
    }
     
// ttttttt
   // Calcul du nombre de connexion max

    for (tIterSomGRTP itS=mGr.begin(mSubAll);itS.go_on();itS++)
    {
        tSomGRTP & aS1 = (*itS);
        double aResiduOr = 1.0; 
        if (mUseOr)
        {
            std::vector<Pt2df> aVecRes;

            for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
            {
                 const cXml_Ori2Im & anOri = (*itA).attr()->Ori();
                 ElSetMax(aS1.attr()->NbPtsMax(),anOri.NbPts());
                 aVecRes.push_back(Pt2df(anOri.Geom().Val().OrientAff().ResiduOr(),anOri.NbPts()));
            }
std::cout << "MMMP Name= " <<   aS1.attr()->Name()  << "\n";
            aResiduOr = MedianPond(aVecRes);
        }
        const std::string & aName = aS1.attr()->Name();
        cXml_RatafiaSom aXRS;
        aXRS.ResiduOr() = aResiduOr;
        MakeFileXML(aXRS,mNoNM->NameRatafiaSom(aName,true));
        MakeFileXML(aXRS,mNoNM->NameRatafiaSom(aName,false));
    }

   // Calcul du taux de recouvrement de chaque arc
    for (tIterSomGRTP itS=mGr.begin(mSubAll);itS.go_on();itS++)
    {
        tSomGRTP & aS1 = (*itS);
        Pt2dr aPInf( 1e60, 1e60);
        Pt2dr aPSup(-1e60,-1e60);
        for (tIterArcGRTP  itA=aS1.begin(mSubAll) ; itA.go_on(); itA++)
        {
             tSomGRTP & aS2 = (*itA).s2();
             const cXml_Ori2Im & anOri = (*itA).attr()->Ori();
             (*itA).attr()->Recouv() =  anOri.NbPts() / double(aS2.attr()->NbPtsMax());
             Box2dr aBox = (*itA).attr()->Box();
             aPInf = Inf(aPInf,aBox._p0);
             aPSup = Sup(aPSup,aBox._p1);
        }
        double aRab = 5 / aS1.attr()->FocPix();
        Pt2dr  aPRab(aRab,aRab);
        aS1.attr()->BoxIm() = Box2dr(aPInf-aPRab,aPSup+aPRab);

        // std::cout <<  aS1.attr()->Name() << " " <<  aS1.attr()->BoxIm().sz() << " " << aS1.attr()->SzDec() << " RAB=" << aPRab << "\n";
        cXml_ResOneImReducTieP aXRIT;
        aXRIT.BoxIm() = aS1.attr()->BoxIm();
        aXRIT.Resol() = 1.0 / aS1.attr()->FocPix();
        const std::string & aName = aS1.attr()->Name() ;
        MakeFileXML(aXRIT,mAppliTR->NameXmlOneIm(aName,true));
        MakeFileXML(aXRIT,mAppliTR->NameXmlOneIm(aName,false));
    }

    // On calcule la partition
    while(OneItereSelection());


    // Creation des box 


    CreateBox();
    

    // Affichage de la partition
    if (mShowPart)
    {
       Show();
    }

    // Lancement de la selection
    ExeSelec();
    // Genere les homologue
    DoExport();
}

bool  cAppliGrRedTieP::IsGBLike()  const
{
   return mGBLike;
}

double  cAppliGrRedTieP::DefFocPix()  const
{
   ELISE_ASSERT(mGBLike,"No DefFocPix for not mGBLike");
   return mDefFocPix;
}
double  cAppliGrRedTieP::DefFoc35()  const
{
   ELISE_ASSERT(mGBLike,"No DefFocPix for not mGBLike");
   return mDefFoc35;
}
/*
bool  DefFocPix() const;
bool  DefFoc35()  const;
*/


NS_OriTiePRed_END


NS_OriTiePRed_USE

int  Ratafia_Main(int argc,char ** argv)
{
    cAppliGrRedTieP anAppli(argc,argv);

    Banniere_Ratafia();
    return EXIT_SUCCESS;
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
