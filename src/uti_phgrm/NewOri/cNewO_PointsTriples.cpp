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

/*******************************************************************/
/*                                                                 */
/*             cNewO_NameManager : nom des cples                   */
/*                                                                 */
/*******************************************************************/

const std::string  cNewO_NameManager::NameDirPtsTriple = "P3Homol/";

std::string  cNewO_NameManager::Dir3P(bool WithMakeDir)
{
    std::string aRes = mDir + NameDirPtsTriple ;
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}

std::string  cNewO_NameManager::Dir3POneImage(cNewO_OneIm * anIm,bool WithMakeDir)
{
    std::string aRes = Dir3P(WithMakeDir) + anIm->Name() + "/";
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}


std::string  cNewO_NameManager::Dir3PDeuxImage(cNewO_OneIm * anI1,cNewO_OneIm * anI2,bool WithMakeDir)
{
    std::string aRes = Dir3POneImage(anI1,WithMakeDir) + anI2->Name() + "/";
    if (WithMakeDir)  ELISE_fp::MkDir(aRes);
    return aRes;
}

std::string cNewO_NameManager::NameHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2)
{
   return Dir3PDeuxImage(anI1,anI2,false) + "HomFloatSym" + mOriCal + ".dat";
}


/*******************************************************************/
/*                                                                 */
/*             cNewO_NameManager : nom des triplets                */
/*                                                                 */
/*******************************************************************/



typedef const std::string * tCPString;
typedef std::pair<std::string,std::string>  tPairStr;

class cNO_P3_NameM
{
       public :
          cNO_P3_NameM
          (
               const std::string & aN0,
               const std::string & aN1,
               const std::string & aN2
          );

      // private :

          int mRank[3];
          
};

cNO_P3_NameM::cNO_P3_NameM
(
               const std::string & aN0,
               const std::string & aN1,
               const std::string & aN2
) 
{
     mRank[0] = (aN0>aN1)  +  (aN0>aN2);
     mRank[1] = (aN0<=aN1) +  (aN1>aN2);
     mRank[2] = (aN0<=aN2)  +  (aN1<=aN2);
}

std::string cNewO_NameManager::NameTriplet(cNewO_OneIm * aI1,cNewO_OneIm * aI2,cNewO_OneIm * aI3,bool WithMakeDir)
{
    ELISE_ASSERT(aI1->Name()<aI2->Name(),"cNO_P3_NameM::NameTriplet");
    ELISE_ASSERT(aI2->Name()<aI3->Name(),"cNO_P3_NameM::NameTriplet");

    std::string aDir = Dir3PDeuxImage(aI1,aI2,WithMakeDir);

    return aDir + "Triplet-" + aI3->Name() + mOriCal + ".dat";
}

typedef std::vector<Pt2df> * tPtrVPt2df;
typedef cNewO_OneIm *        tPtrNIm;

bool cNewO_NameManager::LoadTriplet(cNewO_OneIm * anI1 ,cNewO_OneIm * anI2,cNewO_OneIm * anI3,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3)
{
   cNO_P3_NameM aP3N(anI1->Name(),anI2->Name(),anI3->Name());
   int * aRnk = aP3N.mRank;

   tPtrNIm aVIm[3];
   aVIm[aRnk[0]] =  anI1;
   aVIm[aRnk[1]] =  anI2;
   aVIm[aRnk[2]] =  anI3;

   std::string aName3 = NameTriplet(aVIm[0],aVIm[1],aVIm[2]);
   if (! ELISE_fp::exist_file(aName3)) return false;
   //  std::string aNameT = NameTriplet(*(aP3N.mNames[0]),*(aP3N.mNames[1]),(aP3N.mNames[2]));

   tPtrVPt2df aVPt[3];
   aVPt[aRnk[0]] =  aVP1;
   aVPt[aRnk[1]] =  aVP2;
   aVPt[aRnk[2]] =  aVP3;


   
   ELISE_fp aFile(aName3.c_str(),ELISE_fp::READ,false);
   int aRev = aFile.read_INT4();
   if (aRev>NumHgRev())
   {
   }
   int aNb = aFile.read_INT4();
   for (int aK=0 ; aK<3 ; aK++)
   {
      aVPt[aK]->reserve(aNb);
      aVPt[aK]->resize(aNb);
      aFile.read(VData(*(aVPt[aK])),sizeof(Pt2df),aNb);
   }
   aFile.close();
   return true;
}




/*******************************************************************/
/*                                                                 */
/*                cAppli_GenPTripleOneImage                        */
/*                                                                 */
/*******************************************************************/


class cAppli_GenPTripleOneImage
{
      public :
           cAppli_GenPTripleOneImage(int argc,char ** argv);
           // Genere des homologues, filtre A/R et flottants
           void GenerateHomFloat();
           void GenerateTriplets();
      private :
           void  GenerateTriplet(int aKC1,int aKC2);

           void GenerateHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2);
           void AddOnePackOneSens(cFixedMergeStruct<2,Pt2df> &,cNewO_OneIm * anI1,int anIndI1,cNewO_OneIm * anI2);

           cNewO_NameManager *        mNM;
           std::string                mFullName;
           std::string                mDir;
           std::string                mDir3P;
           std::string                mDirSom;
           std::string                mName;
           std::string                mOriCalib;
           std::vector<cNewO_OneIm*>  mVCams;
           cNewO_OneIm*               mCam;
           int                        mSeuilNbArc;
           // std::map<tPairStr>
           std::vector<std::vector<Pt2df> >  mVP1;
           std::vector<std::vector<Pt2df> >  mVP2;
};

class cCmpPtrIOnName
{
    public :
       bool operator() (cNewO_OneIm * aI1,cNewO_OneIm* aI2) {return aI1->Name() < aI2->Name();}
};
static cCmpPtrIOnName TheCmpPtrIOnName;


cAppli_GenPTripleOneImage::cAppli_GenPTripleOneImage(int argc,char ** argv) :
    mSeuilNbArc (2)
{
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mFullName,"Name of Image"),
        LArgMain() << EAM(mOriCalib,"OriCalib",true,"Calibration directory ")
   );

   SplitDirAndFile(mDir,mName,mFullName);
   mNM  = new cNewO_NameManager(mDir,mOriCalib,"dat");
   mDir3P = mNM->Dir3P(false);
   ELISE_ASSERT(ELISE_fp::IsDirectory(mDir3P),"Dir point triple");

   mCam = new cNewO_OneIm(*mNM,mName);
   mDirSom =  mNM->Dir3POneImage(mCam,true);


   std::string aKeySetHom = "NKS-Set-HomolOfOneImage@@dat@" +mName;
   std::string aKeyAsocHom = "NKS-Assoc-CplIm2Hom@@dat";

   const std::vector<std::string>* aVH = mNM->ICNM()->Get(aKeySetHom);

   // std::cout << aKeySetHom << " " << aVH->size() << "\n";
   mVCams.push_back(mCam);

   for (int aKH = 0 ; aKH <int(aVH->size()) ; aKH++)
   {
        //std::pair<std::string,std::string> aPair = 
        //std::string  aName2 = (*aVH)[aKH];
        std::string aName2 =  mNM->ICNM()->Assoc2To1(aKeyAsocHom,(*aVH)[aKH],false).second;
        if (mName < aName2)
        {
             std::string aName21 = mNM->ICNM()->Assoc1To2(aKeyAsocHom,aName2,mName,true);
             if (ELISE_fp::exist_file(aName21))
             {
                 cNewO_OneIm * aCam2 = new cNewO_OneIm(*mNM,aName2);
                 mVCams.push_back(aCam2);
                 mNM->Dir3PDeuxImage(mCam,aCam2,true);
             }
        }
   }
   std::sort(mVCams.begin(),mVCams.end(),TheCmpPtrIOnName);


/*
   for (int aKC1=0 ; aKC1<int(mVCams.size()) ; aKC1++)
   {
      for (int aKC2=aKC1+1 ; aKC2<int(mVCams.size()) ; aKC2++)
      {
          std::string aName1 = mVCams[aKC1]->Name();
          std::string aName2 = mVCams[aKC2]->Name();
          std::string aNH1H2 = mNM->ICNM()->Assoc1To2(aKeyAsocHom,aName1,aName2,true);
          std::string aNH2H1 = mNM->ICNM()->Assoc1To2(aKeyAsocHom,aName2,aName1,true);
      }
   }
*/
}

/*******************************************************************/
/*                                                                 */
/*             Generation des Triplets                             */
/*                                                                 */
/*******************************************************************/

void cAppli_GenPTripleOneImage::GenerateTriplets()
{
   std::cout << "GeneratePointTriple " << mCam->Name() << "\n";

   mVP1.resize(mVCams.size());
   mVP2.resize(mVCams.size());
   for (int aKC=1 ; aKC<int(mVCams.size()) ; aKC++)
   {
       mNM->LoadHomFloats(mCam,mVCams[aKC],&(mVP1[aKC]),&(mVP2[aKC]));
   }

   for (int aKC1=1 ; aKC1<int(mVCams.size()) ; aKC1++)
   {
      for (int aKC2=aKC1+1 ; aKC2<int(mVCams.size()) ; aKC2++)
      {
            GenerateTriplet(aKC1,aKC2);
      }
   }
}


void AddVPts2Map(cFixedMergeStruct<3,Pt2df> & aMap,const std::vector<Pt2df> & aVP1,int anInd1,const std::vector<Pt2df> & aVP2,int anInd2)
{
    for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
        aMap.AddArc(aVP1[aKP],anInd1,aVP2[aKP],anInd2);
}


typedef  cFixedMergeTieP<3,Pt2df>    tElM;
typedef  cFixedMergeStruct<3,Pt2df>  tMapM;
typedef  std::list<tElM *>           tListM;


void  cAppli_GenPTripleOneImage::GenerateTriplet(int aKC1,int aKC2)
{
   std::string aNameH12 = mNM->NameHomFloat(mVCams[aKC1],mVCams[aKC2]);
   if (!ELISE_fp::exist_file(aNameH12)) return;


   std::string aName3 = mNM->NameTriplet(mCam,mVCams[aKC1],mVCams[aKC2]);
   if (ELISE_fp::exist_file(aName3)) return;

   tMapM aMap;

    {
       std::vector<Pt2df> aVP1;
       std::vector<Pt2df> aVP2;
       mNM->LoadHomFloats(mVCams[aKC1],mVCams[aKC2],&aVP1,&aVP2);

       AddVPts2Map(aMap,aVP1,1,aVP2,2);
    }
    AddVPts2Map(aMap,mVP1[aKC1],0,mVP2[aKC1],1);
    AddVPts2Map(aMap,mVP1[aKC2],0,mVP2[aKC2],2);

    aMap.DoExport();
    const tListM aLM =  aMap.ListMerged();

    std::vector<Pt2df> aVP1,aVP2,aVP3;
    std::vector<U_INT1> aVNb;
    for (tListM::const_iterator itM=aLM.begin() ; itM!=aLM.end() ; itM++)
    {
          if ((*itM)->NbSom()==3 )
          {
              aVP1.push_back((*itM)->GetVal(0));
              aVP2.push_back((*itM)->GetVal(1));
              aVP3.push_back((*itM)->GetVal(2));
              aVNb.push_back((*itM)->NbArc());
          }
    }

    int aNb = aVP1.size();
    if (aNb<TNbMinTriplet)
    {
       aMap.Delete();
       return;
    }

    ELISE_fp aFile(aName3.c_str(),ELISE_fp::WRITE,false);
    aFile.write_INT4(NumHgRev());
    aFile.write_INT4(aNb);
    aFile.write(&(aVP1[0]),sizeof(aVP1[0]),aNb);
    aFile.write(&(aVP2[0]),sizeof(aVP2[0]),aNb);
    aFile.write(&(aVP3[0]),sizeof(aVP3[0]),aNb);
    aFile.write(&(aVNb[0]),sizeof(aVNb[0]),aNb);
    aFile.close();



    aMap.Delete();
}

/*******************************************************************/
/*                                                                 */
/*             Generation des Hom / Flot / Sym                     */
/*                                                                 */
/*******************************************************************/

void  cAppli_GenPTripleOneImage::AddOnePackOneSens(cFixedMergeStruct<2,Pt2df> & aMap,cNewO_OneIm * anI1,int anIndI1,cNewO_OneIm * anI2)
{
    ElPackHomologue aPack = mNM->PackOfName(anI1->Name(),anI2->Name());

    CamStenope * aCS1 = anI1->CS();
    CamStenope * aCS2 = anI2->CS();


    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
        Pt2dr aP1 = aCS1->F2toPtDirRayonL3(itP->P1());
        Pt2dr aP2 = aCS2->F2toPtDirRayonL3(itP->P2());
        Pt2df aQ1(aP1.x,aP1.y);
        Pt2df aQ2(aP2.x,aP2.y);
        // if (aSwap) ElSwap(aQ1,aQ2);
        aMap.AddArc(aQ1,anIndI1,aQ2,1-anIndI1);
    }

}


void  cAppli_GenPTripleOneImage::GenerateHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2)
{
     std::string aNameH = mNM->NameHomFloat(anI1,anI2);
     if (ELISE_fp::exist_file(aNameH)) return;
     // std::cout << "NHH " << aNameH << "\n";


     cFixedMergeStruct<2,Pt2df>   aMap;
     AddOnePackOneSens(aMap,anI1,0,anI2);
     AddOnePackOneSens(aMap,anI2,1,anI1);

     std::vector<Pt2df> aVP1;
     std::vector<Pt2df> aVP2;
     std::vector<U_INT1> aVNb;

     aMap.DoExport();
     const  std::list<cFixedMergeTieP<2,Pt2df> *> &  aLM = aMap.ListMerged();
     for (std::list<cFixedMergeTieP<2,Pt2df> *>::const_iterator itM = aLM.begin() ; itM!=aLM.end() ; itM++)
     {
          const cFixedMergeTieP<2,Pt2df> & aM = **itM;
          if (aM.NbArc() >= mSeuilNbArc)
          {
              aVP1.push_back(aM.GetVal(0));
              aVP2.push_back(aM.GetVal(1));
              aVNb.push_back(aM.NbArc());
          }
     }


     ELISE_fp aFile(aNameH.c_str(),ELISE_fp::WRITE,false);

     int aNb = aVP1.size();
     aFile.write_INT4(NumHgRev());
     aFile.write_INT4(aNb);
     aFile.write(&(aVP1[0]),sizeof(aVP1[0]),aNb);
     aFile.write(&(aVP2[0]),sizeof(aVP2[0]),aNb);
     aFile.write(&(aVNb[0]),sizeof(aVNb[0]),aNb);
     aFile.close();

     aMap.Delete();
}

void cNewO_NameManager::LoadHomFloats(cNewO_OneIm * anI1,cNewO_OneIm * anI2,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2)
{
   if (anI1->Name() > anI2->Name())
   {
       ElSwap(anI1,anI2);
       ElSwap(aVP1,aVP2);
   }
   std::string aNameH = NameHomFloat(anI1,anI2);

   ELISE_fp aFile(aNameH.c_str(),ELISE_fp::READ,false);
   // FILE *  aFP = aFile.FP() ;
   int aRev = aFile.read_INT4();
   if (aRev>NumHgRev())
   {
   }
   int aNb = aFile.read_INT4();


   aVP1->reserve(aNb);
   aVP2->reserve(aNb);
   aVP1->resize(aNb);
   aVP2->resize(aNb);
   aFile.read(VData(*aVP1),sizeof((*aVP1)[0]),aNb);
   aFile.read(VData(*aVP2),sizeof((*aVP2)[0]),aNb);

   aFile.close();
}

void cAppli_GenPTripleOneImage::GenerateHomFloat()
{
    for (int aK=1 ; aK<int(mVCams.size()) ; aK++)
    {
        GenerateHomFloat(mVCams[0],mVCams[aK]);
    }
}


/*******************************************************************/
/*                                                                 */
/*                  ::                                             */
/*                                                                 */
/*******************************************************************/

int CPP_GenOneImP3(int argc,char ** argv)
{
    cAppli_GenPTripleOneImage anAppli(argc,argv);

    anAppli.GenerateTriplets();
    return EXIT_SUCCESS;
}

int CPP_GenOneHomFloat(int argc,char ** argv)
{
    cAppli_GenPTripleOneImage anAppli(argc,argv);

    anAppli.GenerateHomFloat();
    return EXIT_SUCCESS;
}


int PreGenerateDuTriplet(int argc,char ** argv,const std::string & aComIm)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aFullName,anOriCalib;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aFullName,"Name of Image"),
        LArgMain() << EAM(anOriCalib,"OriCalib",true,"Calibration directory ")
   );

   cElemAppliSetFile anEASF(aFullName);
   if (!EAMIsInit(&anOriCalib))
   {
      MakeXmlXifInfo(aFullName,anEASF.mICNM);
   }

   cNewO_NameManager aNM(anEASF.mDir,anOriCalib,"dat");
   aNM.Dir3P(true);
   const cInterfChantierNameManipulateur::tSet * aSetIm = anEASF.SetIm();

   std::list<std::string> aLCom;
   for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
   {
        std::string aCom =   MM3dBinFile_quotes( "TestLib ") + aComIm + " "  + anEASF.mDir+(*aSetIm)[aKIm] ;

        if (EAMIsInit(&anOriCalib))  aCom = aCom + " OriCalib=" + anOriCalib;
        aLCom.push_back(aCom);

        //std::cout << aCom << "\n";

   }

   cEl_GPAO::DoComInParal(aLCom);

    return EXIT_SUCCESS;
}

int CPP_GenAllHomFloat(int argc,char ** argv)
{
     return  PreGenerateDuTriplet(argc,argv,"NO_OneHomFloat");
}

int CPP_GenAllImP3(int argc,char ** argv)
{
     return  PreGenerateDuTriplet(argc,argv,"NO_OneImTriplet");
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
