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
#include "ReducHom.h"


bool AllowUnsortedVarIn_SetMappingCur = false;


NS_RHH_BEGIN


/*************************************************/
/*                                               */
/*                XXXXXXX                        */
/*                                               */
/*************************************************/

void WarnTest()
{
    if (TEST)
    {
        for (int aK=0 ; aK< 10 ; aK++)
            std::cout << "========== !!!!! ATTENTION MODE TEST !!!!!!========\n";
    }
}

cAppliReduc::cAppliReduc(int argc,char ** argv) :
   mHomByParal   (true),
   mImportTxt    (false),
   mExportTxt    (false),
   mExtHomol     (""),
   mMinNbPtH     (20),
   mSeuilQual    (20),
   mRatioQualMoy (4.0),
   mSeuilDistNorm (0.2),
   mKernConnec   (3),
   mKernSize     (6),
   mSetEq        (cNameSpaceEqF::eSysL2BlocSym),
  //   mSetEq        (cNameSpaceEqF::eSysPlein),
   mH1On2        (true),
   mHFD          (true),
   mKeyHomogr    (std::string("NKS-RHH-Assoc-CplIm2Data@@Homogr@") + (mHFD ?  "dmp" : "xml")),
   mKeyHomolH    (std::string("NKS-RHH-Assoc-CplIm2Data@@HomolH@dat") ),
   mSkipHomDone  (true),
   mSkipPlanDone (true),
   mSkipAllDone  (true),
   mAltiCible    (1000),
   mHasImFocusPlan (false),
   mImFocusPlan    (""),
   mHasImCAmel     (false),
   mNameICA        (""),
   mImCAmel        (0),
   mDoCompensLoc   (true)
   // mQT        (PtOfPhi,Box2dr(Pt2dr(-100,-100),Pt2dr(30000,30000)),10,500)
{

    int aIntNivShow = eShowGlob;
    CreateIndex();
   // Lecture bas niveau des parametres
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mFullName,"Full Directory (Dir+Pattern)",eSAM_IsPatFile)
                    << EAMC(mOri,"Orientation", eSAM_IsExistDirOri),
        LArgMain()  << EAM(mImportTxt,"ImpTxt",true,"Import in text format(def=false)")
                    << EAM(mExportTxt,"ExpTxt",true,"Export in text format(def=false)")
                    << EAM(mNameICA,"ICA",true,"Central image for local optim of hom", eSAM_IsExistFile)
                    << EAM(mExtHomol,"ExtH",true,"Extension for homol, like SrRes, def=\"\")")
                    << EAM(mMinNbPtH,"NbMinHom",true,"Nb Min Pts For Homography Computation def=20")
                    << EAM(mSeuilQual,"SeuilQual",true,"Quality Theshold for homography (Def=20.0)")
                    << EAM(mRatioQualMoy,"RatioQualMoy",true,"Ratio to validate / average qual (def=4.0)")
                    << EAM(mSeuilDistNorm,"SeuilDistNorm",true,"threshold to validate firt normal / second (def=0.2)")
                    << EAM(aIntNivShow,"Show",true,"Level of Show (0=None, Def= 1)")
                    << EAM(mHomByParal,"HbP",true,"Compute Homography in // (Def=true)")
                    << EAM(mOriVerif,"Verif",true,"To generate perfect homographic tie (tuning purpose)", eSAM_InternalUse)
                    << EAM(mH1On2,"H1on2",true,"Fix arbitrary order of hom , tuning", eSAM_InternalUse)
                    << EAM(mHFD,"HFD",true,"Homogr in dump format, tuning (Def true)", eSAM_InternalUse)
                    << EAM(mSkipHomDone,"SHD",true,"Skip Hom calc when files already Done (accelerate tuning))", eSAM_InternalUse)
                    << EAM(mSkipPlanDone,"SPD",true,"Skip Plan calc when files already Done (accelerate tuning))", eSAM_InternalUse)
                    << EAM(mSkipAllDone,"SAD",true,"Skip All calc when files already Done (accelerate tuning))", eSAM_InternalUse)
                    << EAM(mAltiCible,"Alti",true,"Fix arbitrary altitude (def = 1000)")
                    << EAM(mImFocusPlan,"IFP",true,"Image Focus on Plane, tuning", eSAM_InternalUse)
                    << EAM(mDoCompensLoc,"DCL",true,"DoCompens loc (tuning/testing)", eSAM_InternalUse)
    );

    if (!MMVisualMode)
    {
    if (EAMIsInit(&mSkipAllDone))
    {
         if  (!EAMIsInit(&mSkipHomDone))  mSkipHomDone = mSkipAllDone;
         if  (!EAMIsInit(&mSkipPlanDone)) mSkipPlanDone = mSkipAllDone;
    }

    if (EAMIsInit(&mImFocusPlan))
    {
        mHasImFocusPlan = true;
    }

    if (EAMIsInit(&mNameICA))
    {
        mHasImCAmel = true;
    }


   SplitDirAndFile(mDir,mName,mFullName);
   StdCorrecNameOrient(mOri,mDir);
   if (EAMIsInit(&mOriVerif))
   {
      mHomByParal = false;
      StdCorrecNameOrient(mOriVerif,mDir);
   }

    mKeyOri = "NKS-Assoc-FromFocMm@Ori-" + mOri +"/AutoCal@" + ".xml";
    if (EAMIsInit(&mOriVerif))
       mKeyVerif = "NKS-Assoc-Im2Orient@-" + mOriVerif;

    mNivShow = (eNivShow) aIntNivShow;
    if (Show(eShowGlob))
        std::cout << "RHH begin \n";


   // Creation noms et associations
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   mSetNameIm = mICNM->Get(mName);
   // mKeyHomol = "NKS-Set-Homol@"+mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
   // mKeyH2I = "NKS-Assoc-CplIm2Hom@"+mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
   // mKeySetHomol = KeyHIn("NKS-Set-Homol");
   mKeySetHomol = "NKS-Set-Homol-Filtered@" +mExtHomol+(mImportTxt ?"@txt@" : "@dat@") + mName;
   mKeyInitIm2Homol =  KeyHIn("NKS-Assoc-CplIm2Hom");
   mSetNameHom =  mICNM->Get(mKeySetHomol);

   // If there is a cental image, compute the new set of name

   if (mHasImCAmel)
   {
      std::vector<std::string> * aNewSet = new std::vector<std::string>;
      bool GotImCA = false;
      for (int aK=0 ; aK<int(mSetNameIm->size()); aK++)
      {
          const std::string & aName = (*mSetNameIm)[aK];
          if (aName== mNameICA)
          {
             GotImCA = true;
             aNewSet->push_back(aName);
          }
          else
          {
               std::string aFileH = mDir + mICNM->Assoc1To2(mKeyInitIm2Homol,mNameICA,aName,true);
               if (ELISE_fp::exist_file(aFileH))
               {
                  aNewSet->push_back(aName);
               }
               else
               {
               }
          }
      }
      ELISE_ASSERT(GotImCA,"Central Image for Homography amelioration is not an existing image");
      mSetNameIm = aNewSet;
   }

   if (Show(eShowGlob))
      std::cout << "NbIm " << mSetNameIm->size() << " NbH " << mSetNameHom->size() << "\n";


   // Creation des images
   for (int aKIm=0 ; aKIm<int(mSetNameIm->size()) ; aKIm++)
   {
        const std::string & aName = (*mSetNameIm)[aKIm];
        cImagH * anIm=new cImagH(aName,*this,aKIm);
        mIms.push_back(anIm);
        mDicoIm[aName] = anIm;

        if (aName==mNameICA)
        {
           mImCAmel = anIm;
        }
   }


   //  Create the "topologic" structure of graphe of images connected
   //  this structure is initially empty, points are not loaded
   //
   for (int aKH=0 ; aKH<int(mSetNameHom->size()) ; aKH++)
   {
        const std::string & aName = (*mSetNameHom)[aKH];
        std::pair<std::string,std::string> aN1N2 = mICNM->Assoc2To1(mKeyInitIm2Homol,aName,false);
        cImagH * aI1 = mDicoIm[aN1N2.first];
        cImagH * aI2 = mDicoIm[aN1N2.second];
        if (aI1 && aI2)
        {
            aI1->AddLink(aI2,aName);
        }
   }

    }
}

std::string cAppliReduc::NameCalib(const std::string & aNameIm) const
{
   return mDir+ mICNM->Assoc1To1(mKeyOri,aNameIm,true);
}


std::string cAppliReduc::NameVerif(const std::string & aNameIm) const
{
   return  (mKeyVerif== "") ? ""  : mDir+ mICNM->Assoc1To1(mKeyVerif,aNameIm,true);
}



bool cAppliReduc::Show(eNivShow aLevel) const
{
   return  mNivShow >= aLevel;
}

std::string cAppliReduc::KeyHIn(const std::string & aKeyGen) const
{
    return aKeyGen+"@" + mExtHomol+"@"+(mImportTxt ?"txt" : "dat");
}



void cAppliReduc::ComputeHom()
{
   if (mHomByParal)
   {
       if (Show(eShowGlob))
          std::cout << "HomByParal BEGIN\n";

       std::list<std::string> aLCom;
       for (int aK=0 ; aK<int(mIms.size()) ; aK++)
           mIms[aK]->AddComCompHomogr(aLCom);

       if (0)
       {
          cEl_GPAO::DoComInSerie(aLCom);
          for (int aK=0 ; aK< 10 ; aK++) std::cout << "SERIIIIIIIIIIIIIIIii\n";
       }
       else
       {
          cEl_GPAO::DoComInParal(aLCom);
       }

       if (Show(eShowGlob))
       {
          std::cout << "HomByParal : Command run\n";
          // getchar();
       }

       for (int aK=0 ; aK<int(mIms.size()) ; aK++)
           mIms[aK]->LoadComHomogr();

       if (Show(eShowGlob))
       {
          std::cout << "HomByParal END\n";
          // getchar();
       }
   }


   for (int aK=0 ; aK<int(mIms.size()) ; aK++)
   {
       mIms[aK]->ComputeLnkHom();
   }


/*
   A priori sera remis plus tard  apres la phase d'image init
   if (Show(eShowGlob))
      std::cout << "Lnk && Plan BEGIN\n";
    {
        std::list<std::string> aLComPl;
        for (int aK=0 ; aK<int(mIms.size()) ; aK++)
        {
             std::string aCom = mIms[aK]->EstimatePlan();
             if (aCom!="")
                aLComPl.push_back(aCom);
        }
        cEl_GPAO::DoComInParal(aLComPl);
   }
   if (Show(eShowGlob))
      std::cout << "Lnk && Plan END\n";
*/



   // Space Init comme cela c'est H2-H1 (P1) = P2 qui est ajustee, cela evite
   // d'etre "attire" par la solution nulle.
    bool SpaceInit = true;

    // Init systeme equation
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
          cImagH * anI =  mIms[aK];
          anI->HF() = mSetEq.NewHomF(anI->Hi2t(),cNameSpaceEqF::eHomLibre);
          anI->EqOneHF() = mSetEq.NewOneEqHomog(*anI->HF(),false);
    }

    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
        cImagH * anI1 =  mIms[aK];
        const tMapName2Link & aLL = anI1->Lnks();
        for (tMapName2Link::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
        {
            cImagH * anI2 = itL->second->Dest();
            itL->second->EqHF() = mSetEq.NewEqHomog(SpaceInit,*(anI1->HF()),*(anI2->HF()),0,false);
        }
    }

    int aNbBl = mSetEq.NbBloc();
    cAMD_Interf * mAMD = new cAMD_Interf (aNbBl);
    for (int aK=0 ; aK<aNbBl ; aK++)
    {
        mAMD->AddArc(aK,aK,true);
    }
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
        cImagH * anI1 =  mIms[aK];
        cHomogFormelle *  aHF1 = anI1->HF();

        const tMapName2Link & aLL = anI1->Lnks();
        for (tMapName2Link::const_iterator itL = aLL.begin(); itL != aLL.end(); itL++)
        {
            cImagH * anI2 = itL->second->Dest();
            cHomogFormelle *  aHF2 = anI2->HF();
            int aKbl1 = aHF1->IncInterv().NumBlocAlloc();
            int aKbl2 = aHF2->IncInterv().NumBlocAlloc();
            mAMD->AddArc(aKbl1,aKbl2,true);
        }
    }
   std::vector<int>  anOrder = mAMD->DoRank();
   const std::vector<cIncIntervale *>  & aVInt = mSetEq.BlocsIncAlloc();
   for (int aK=0 ; aK<int(aVInt.size()) ; aK++)
   {
       aVInt[aK]->SetOrder(anOrder[aK]);
   }

   mSetEq.SetClosed();

   if (mImCAmel)
   {
        AmelioHomLocal(*mImCAmel);
        mImCAmel->EstimatePlan();
        exit(0);
   }

   // Cree l'arbre  de fusion hierarchique
    //  TestMerge_CalcHcImage();
}


void cAppliReduc::ComputePts()
{

    // Create the multiple tie points structure
    for (int aK=0 ; aK<int(mIms.size()) ; aK++)
    {
         ClearIndex();
         mIms[aK]->ComputePts();
    }
}

cSetEqFormelles &  cAppliReduc::SetEq()
{
   return mSetEq;
}

const std::string & cAppliReduc::Dir() const
{
   return mDir;
}

cInterfChantierNameManipulateur * cAppliReduc::ICNM() const
{
    return mICNM;
}



void cAppliReduc::DoAll()
{
    ComputeHom();
    ComputePts();
    cPtHom::ShowAll();
}

int  cAppliReduc::MinNbPtH() const
{
   return mMinNbPtH;
}


double cAppliReduc::SeuilQual () const
{
   return mSeuilQual;
}

double cAppliReduc::RatioQualMoy () const
{
   return mRatioQualMoy;
}

double cAppliReduc::SeuilDistNorm () const
{
   return mSeuilDistNorm;
}

int    cAppliReduc::KernConnec() const
{
    return mKernConnec;
}
int    cAppliReduc::KernSize() const
{
    return mKernSize;
}

bool  cAppliReduc::H1On2() const
{
   return mH1On2;
}


std::string cAppliReduc::NameFileHomogr(const cLink2Img & aLnK) const
{
   return mDir + mICNM->Assoc1To2(mKeyHomogr,aLnK.Srce()->Name(),aLnK.Dest()->Name(),true);
}
std::string cAppliReduc::NameFileHomolH(const cLink2Img & aLnK) const
{
   return mDir + mICNM->Assoc1To2(mKeyHomolH,aLnK.Srce()->Name(),aLnK.Dest()->Name(),true);
}

bool cAppliReduc::SkipHomDone() const
{
   return mSkipHomDone;
}
bool cAppliReduc::SkipPlanDone() const
{
   return mSkipPlanDone;
}
double cAppliReduc::AltiCible() const
{
   return mAltiCible;
}

bool   cAppliReduc::HasImFocusPlan () const
{
    return mHasImFocusPlan;
}

std::string      cAppliReduc::ImFocusPlan () const
{
    return mImFocusPlan;
}



NS_RHH_END


NS_RHH_USE

int RHH_main(int argc,char **argv)
{
   AllowUnsortedVarIn_SetMappingCur = true;

   cAppliReduc anAppli(argc,argv);

   anAppli.ComputeHom();

   if (anAppli.Show(eShowGlob))
      std::cout << "RHH end \n";

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
