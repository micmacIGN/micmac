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
/*             cNewO_NameManager : nom des triplets                */
/*                                                                 */
/*******************************************************************/



typedef const std::string * tCPString;
typedef std::pair<std::string,std::string>  tPairStr;





void F()
{
    cTplTriplet<int> aTP(1,2,3);
    cTplTripletByRef<int> aTP2(1,2,3);
}




typedef std::vector<Pt2df> * tPtrVPt2df;
typedef cNewO_OneIm *        tPtrNIm;

bool cNewO_NameManager::LoadTriplet(const std::string & anI1,const std::string & anI2,const std::string & anI3,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3)
{
   int aRnk[3] ;
   Rank3(aRnk,anI1,anI2,anI3);

   std::string aVIm[3];
   aVIm[aRnk[0]] =  anI1;
   aVIm[aRnk[1]] =  anI2;
   aVIm[aRnk[2]] =  anI3;

   std::string aName3 = NameHomTriplet(aVIm[0],aVIm[1],aVIm[2]);
   if (! ELISE_fp::exist_file(aName3)) return false;

if (0 && MPD_MM())
{
   std::cout << "NN33333 " << aName3 << "\n";
}

   tPtrVPt2df aVPt[3];
   aVPt[aRnk[0]] =  aVP1;
   aVPt[aRnk[1]] =  aVP2;
   aVPt[aRnk[2]] =  aVP3;



   ELISE_fp aFile(aName3.c_str(),ELISE_fp::READ,false);
   /*int aRev = */aFile.read_INT4();
   // NumHgRev doesn't work
   //if (aRev>NumHgRev())
   //{
   //}
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
/*
*/


bool cNewO_NameManager::LoadTriplet(cNewO_OneIm * anI1 ,cNewO_OneIm * anI2,cNewO_OneIm * anI3,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,std::vector<Pt2df> * aVP3)
{
    return LoadTriplet(anI1->Name(),anI2->Name(),anI3->Name(),aVP1,aVP2,aVP3);
}



void cNewO_NameManager::LoadHomFloats(std::string  aName1,std::string  aName2,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP)
{
   if (aName1 > aName2)
   {
       ElSwap(aName1,aName2);
       ElSwap(aVP1,aVP2);
   }
   std::string aNameH = NameHomFloat(aName1,aName2);
if (0 &&MPD_MM())
{
   std::cout << "N22222 " << aNameH << "\n";
}
   GenLoadHomFloats(aNameH,aVP1,aVP2,SVP);
}


void cNewO_NameManager::GenLoadHomFloats(const std::string &  aNameH,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2,bool SVP)
{
   if (SVP)
   {
        if (! ELISE_fp::exist_file(aNameH))
        {
            aVP1->clear();
            aVP2->clear();
            return;
        }
   }

   ELISE_fp aFile(aNameH.c_str(),ELISE_fp::READ,false);
   // FILE *  aFP = aFile.FP() ;
   /*int aRev = */aFile.read_INT4();
   // NumHgRev doesn't work
   //if (aRev>NumHgRev())
   //{
   //}
   int aNb = aFile.read_INT4();
   aVP1->reserve(aNb);
   aVP2->reserve(aNb);
   aVP1->resize(aNb);
   aVP2->resize(aNb);
   aFile.read(VData(*aVP1),sizeof((*aVP1)[0]),aNb);
   aFile.read(VData(*aVP2),sizeof((*aVP2)[0]),aNb);

   aFile.close();
}
void cNewO_NameManager::LoadHomFloats(cNewO_OneIm * anI1,cNewO_OneIm * anI2,std::vector<Pt2df> * aVP1,std::vector<Pt2df> * aVP2)
{
    LoadHomFloats(anI1->Name(),anI2->Name(),aVP1,aVP2);
}





void cNewO_NameManager::WriteTriplet
     (
          const std::string         & aNameFile,
          const std::vector<Pt2df>  & aVP1,
          const std::vector<Pt2df>  & aVP2,
          const std::vector<Pt2df>  * aVP3,
          const std::vector<U_INT1> & aVNb
     )
{
    int aNb = (int)aVP1.size();
    ELISE_fp aFile(aNameFile.c_str(),ELISE_fp::WRITE,false);
    // NumHgRev doesn't work
    //aFile.write_INT4(NumHgRev());
    aFile.write_INT4(0);
    aFile.write_INT4(aNb);
    aFile.write(&(aVP1[0]),sizeof(aVP1[0]),aNb);
    aFile.write(&(aVP2[0]),sizeof(aVP2[0]),aNb);
    if (aVP3)
    {
        aFile.write(&((*aVP3)[0]),sizeof((*aVP3)[0]),aNb);
    }
    aFile.write(&(aVNb[0]),sizeof(aVNb[0]),aNb);
    aFile.close();
}


void cNewO_NameManager::WriteTriplet
     (
          const std::string         & aNameFile,
          const std::vector<Pt2df>  & aVP1,
          const std::vector<Pt2df>  & aVP2,
          const std::vector<Pt2df>  & aVP3,
          const std::vector<U_INT1> & aVNb
     )
{
    WriteTriplet(aNameFile,aVP1,aVP2,&aVP3,aVNb);
}

void cNewO_NameManager::WriteCouple
     (
          const std::string         & aNameFile,
          const std::vector<Pt2df>  & aVP1,
          const std::vector<Pt2df>  & aVP2,
          const std::vector<U_INT1> & aVNb
     )
{
    WriteTriplet(aNameFile,aVP1,aVP2,0,aVNb);
}



/*******************************************************************/
/*                                                                 */
/*                cAppli_GenPTripleOneImage                        */
/*                                                                 */
/*******************************************************************/


class cAppli_GenPTripleOneImage : public cCommonMartiniAppli
{
      public :
           cAppli_GenPTripleOneImage(int argc,char ** argv);
           // Genere des homologues, filtre A/R et flottants
           void GenerateHomFloat();
           void GenerateTriplets();
      private :
           void  GenerateTriplet(int aKC1,int aKC2);

           void GenerateHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2);
           void AddOnePackOneSens(cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> >  &,cNewO_OneIm * anI1,int anIndI1,cNewO_OneIm * anI2);

           cNewO_NameManager *        mNM;
           std::string                mFullName;
           std::string                mDir;
           std::string                mDir3P;
           std::string                mDirSom;
           std::string                mName;
           std::vector<cNewO_OneIm*>  mVCams;
           cNewO_OneIm*               mCam;
           int                        mSeuilNbArc;
           // std::map<tPairStr>
           std::vector<std::vector<Pt2df> >  mVP1;
           std::vector<std::vector<Pt2df> >  mVP2;
           bool                              mSkWhenExist;
};

class cCmpPtrIOnName
{
    public :
       bool operator() (cNewO_OneIm * aI1,cNewO_OneIm* aI2) {return aI1->Name() < aI2->Name();}
};
static cCmpPtrIOnName TheCmpPtrIOnName;


cAppli_GenPTripleOneImage::cAppli_GenPTripleOneImage(int argc,char ** argv) :
    mSeuilNbArc  (1),  // Change car : existe des cas a forte assymetrie + 2 genere des plantage lorsque + de triplets que de couples
    mSkWhenExist (true)
{
   
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(mFullName,"Name of Image", eSAM_IsExistFile),
        LArgMain() 
                   << EAM(mSkWhenExist,"SWE",true,"Skip when file alreay exist (Def=true, tuning purpose)", eSAM_IsBool)
                   << ArgCMA()
   );


   if (MMVisualMode) return;

   SplitDirAndFile(mDir,mName,mFullName);
   mNM  = new cNewO_NameManager(mExtName,mPrefHom,mQuick,mDir,mNameOriCalib,"dat");
   mDir3P = mNM->Dir3P(false);
   ELISE_ASSERT(ELISE_fp::IsDirectory(mDir3P),"Dir point triple");

   mCam = new cNewO_OneIm(*mNM,mName);
   mDirSom =  mNM->Dir3POneImage(mCam,true);

   mVCams.push_back(mCam);

   std::list<std::string> aLN = mNM->ListeImOrientedWith(mName);
   for (std::list<std::string>::const_iterator itN=aLN.begin() ; itN!=aLN.end() ; itN++)
   {
        if (mName < *itN)
        {
            /// std::cout << "PPPP " << mName << " " << *itN << "\n";
            cNewO_OneIm * aCam2 = new cNewO_OneIm(*mNM,*itN);
            mVCams.push_back(aCam2);
        }
   }
/*
   std::string aKeySetHom = "NKS-Set-HomolOfOneImage@@dat@" +mName;
   std::string aKeyAsocHom = "NKS-Assoc-CplIm2Hom@@dat";

   const std::vector<std::string>* aVH = mNM->ICNM()->Get(aKeySetHom);

   // std::cout << aKeySetHom << " " << aVH->size() << "\n";

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
*/
   std::sort(mVCams.begin(),mVCams.end(),TheCmpPtrIOnName);
}

/*******************************************************************/
/*                                                                 */
/*             Generation des Triplets                             */
/*                                                                 */
/*******************************************************************/


void cAppli_GenPTripleOneImage::GenerateTriplets()
{
   ElTimer aChrono;
   if (!mSkWhenExist)  std::cout << "GeneratePointTriple " << mCam->Name() << "\n";  // !mSkWhenExist ~ mise au point


   /* Charge les points correspond a tous les homologues avec le "master" */

   mVP1.resize(mVCams.size());
   mVP2.resize(mVCams.size());
   for (int aKC=1 ; aKC<int(mVCams.size()) ; aKC++)
   {
       mNM->LoadHomFloats(mCam,mVCams[aKC],&(mVP1[aKC]),&(mVP2[aKC]));
   }

   /* parcour tout les couple d'images ne contenant pas le master, pour generer
      les triplets avec le master */
   for (int aKC1=1 ; aKC1<int(mVCams.size()) ; aKC1++)
   {
      for (int aKC2=aKC1+1 ; aKC2<int(mVCams.size()) ; aKC2++)
      {
            GenerateTriplet(aKC1,aKC2);
      }
   }
    if (!mSkWhenExist)  std::cout << "   ==> END GeneratePointTriple " << mCam->Name()  << " " << aChrono.uval() << "\n";
}

typedef  cFixedSizeMergeTieP<3,Pt2df,cCMT_NoVal>    tElM;
typedef  cStructMergeTieP<cFixedSizeMergeTieP<3,Pt2df,cCMT_NoVal> >  tMapM;
typedef  std::list<tElM *>           tListM;

void AddVPts2Map(tMapM & aMap,const std::vector<Pt2df> & aVP1,int anInd1,const std::vector<Pt2df> & aVP2,int anInd2)
{
    for (int aKP=0 ; aKP<int(aVP1.size()) ; aKP++)
        aMap.AddArc(aVP1[aKP],anInd1,aVP2[aKP],anInd2,cCMT_NoVal());
}


/*
   Ecrit eventuellement le fichier de points tripe [0,KC1,KC2]
*/


void  cAppli_GenPTripleOneImage::GenerateTriplet(int aKC1,int aKC2)
{
    std::string aNameH12 = mNM->NameHomFloat(mVCams[aKC1],mVCams[aKC2]);
    if (!ELISE_fp::exist_file(aNameH12)) return;

    std::string aName3 = mNM->NameHomTriplet(mCam,mVCams[aKC1],mVCams[aKC2]);
    if (mSkWhenExist && ELISE_fp::exist_file(aName3)) return;

    // Lit le dernier vecteur de point homologue qui manquait
    std::vector<Pt2df> aVP1In;
    std::vector<Pt2df> aVP2In;
    mNM->LoadHomFloats(mVCams[aKC1],mVCams[aKC2],&aVP1In,&aVP2In);


    // Cree la structure de points multiples
    tMapM aMap(3,false);
    AddVPts2Map(aMap,aVP1In,1,aVP2In,2);
    AddVPts2Map(aMap,mVP1[aKC1],0,mVP2[aKC1],1);
    AddVPts2Map(aMap,mVP1[aKC2],0,mVP2[aKC2],2);
    aMap.DoExport();
    const tListM aLM =  aMap.ListMerged();

    // Enregistre les points triples
    std::vector<Pt2df> aVP1Exp,aVP2Exp,aVP3Exp;
    std::vector<U_INT1> aVNb;
    for (tListM::const_iterator itM=aLM.begin() ; itM!=aLM.end() ; itM++)
    {
          if ((*itM)->NbSom()==3 )
          {
              aVP1Exp.push_back((*itM)->GetVal(0));
              aVP2Exp.push_back((*itM)->GetVal(1));
              aVP3Exp.push_back((*itM)->GetVal(2));
              aVNb.push_back((*itM)->NbArc());
          }
    }

    int aNb = (int)aVP1Exp.size();
    if (aNb<mTNbMinTriplet)
    {
       aMap.Delete();
       return;
    }
    // Sauvegarde les triplet si assez

    mNM->WriteTriplet(aName3,aVP1Exp,aVP2Exp,aVP3Exp,aVNb);

if (0 && MPD_MM())
{
    std::cout << "GGTTt 3 " <<  aNb  
             << " :: " << aVP1In.size() << " " << aVP2In.size()
             << " :: " << mVP1[aKC1].size() << " " <<  mVP2[aKC1].size()
             << " :: " << mVP1[aKC2].size() << " " <<  mVP2[aKC2].size()
             << "\n";
}

    aMap.Delete();
}

/*******************************************************************/
/*                                                                 */
/*             Generation des Hom / Flot / Sym                     */
/*                                                                 */
/*******************************************************************/

void  cAppli_GenPTripleOneImage::AddOnePackOneSens(cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> > & aMap,cNewO_OneIm * anI1,int anIndI1,cNewO_OneIm * anI2)
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
        aMap.AddArc(aQ1,anIndI1,aQ2,1-anIndI1,cCMT_NoVal());
    }

}


void  cAppli_GenPTripleOneImage::GenerateHomFloat(cNewO_OneIm * anI1,cNewO_OneIm * anI2)
{
     std::string aNameH = mNM->NameHomFloat(anI1,anI2);
     if (mSkWhenExist && ELISE_fp::exist_file(aNameH)) return;
     // std::cout << "NHH " << aNameH << "\n";


     cStructMergeTieP< cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> >    aMap(2,false);
     AddOnePackOneSens(aMap,anI1,0,anI2);
     AddOnePackOneSens(aMap,anI2,1,anI1);

     std::vector<Pt2df> aVP1;
     std::vector<Pt2df> aVP2;
     std::vector<U_INT1> aVNb;

     aMap.DoExport();
     const  std::list<cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> *> &  aLM = aMap.ListMerged();
     for (std::list<cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> *>::const_iterator itM = aLM.begin() ; itM!=aLM.end() ; itM++)
     {
          const cFixedSizeMergeTieP<2,Pt2df,cCMT_NoVal> & aM = **itM;
          if (aM.NbArc() >= mSeuilNbArc)
          {
              aVP1.push_back(aM.GetVal(0));
              aVP2.push_back(aM.GetVal(1));
              aVNb.push_back(aM.NbArc());
          }
     }
     mNM->WriteCouple(aNameH,aVP1,aVP2,aVNb);
     aMap.Delete();
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

cExeParalByPaquets::cExeParalByPaquets(const std::string & aMes,int anEstimNbCom) :
     mMes             (aMes),
     mEstimNbCom      (anEstimNbCom),
     mCpt             (0),
     mNbInOnePaquet   (ElMax(2*MMNbProc(),anEstimNbCom/20))
{
}

void cExeParalByPaquets::AddCom(const std::string & aCom)
{
    mLCom.push_back(aCom);
    mCpt++;
    if ((mCpt%mNbInOnePaquet)==0) 
    {
       ExeCom();
    }
}

void cExeParalByPaquets::ExeCom()
{
   if (! mLCom.empty())
   {
      cEl_GPAO::DoComInParal(mLCom);
      mLCom.clear();
      std::cout << "   " << mMes << " Done " << mCpt << " out of " << mEstimNbCom << " in time=" << mChrono.uval() << "\n";
   }
}

cExeParalByPaquets::~cExeParalByPaquets()
{
   ExeCom();
}



int PreGenerateDuTriplet(int argc,char ** argv,const std::string & aComIm)
{
   MMD_InitArgcArgv(argc,argv);

   std::string aFullName;
   bool aSkWhenExist;
   cCommonMartiniAppli aCMA;
   ElInitArgMain
   (
        argc,argv,
        LArgMain() << EAMC(aFullName,"Name of Image"),
        LArgMain()
                   << EAM(aSkWhenExist,"SWE",true,"Skip when file alreay exist (Def=true, tuning purpose)", eSAM_IsBool)
                   << aCMA.ArgCMA()
   );

   cElemAppliSetFile anEASF(aFullName);
   if (!EAMIsInit(&aCMA.mNameOriCalib))
   {
      MakeXmlXifInfo(aFullName,anEASF.mICNM);
   }

   cNewO_NameManager aNM(aCMA.mExtName,aCMA.mPrefHom,aCMA.mQuick,anEASF.mDir,aCMA.mNameOriCalib,"dat");
   aNM.Dir3P(true);
   const cInterfChantierNameManipulateur::tSet * aSetIm = anEASF.SetIm();

   {
       cExeParalByPaquets anEPbP(aComIm, (int)aSetIm->size());

       for (int aKIm=0 ; aKIm<int(aSetIm->size()) ; aKIm++)
       {
            std::string aCom =   MM3dBinFile_quotes( "TestLib ") + aComIm + " "  + anEASF.mDir+(*aSetIm)[aKIm] ;

            aCom += " SWE=" +ToString(aSkWhenExist);
            aCom +=  aCMA.ComParam();

            anEPbP.AddCom(aCom);
       }
   }

    return EXIT_SUCCESS;
}

/* Apparemment cette fonction n'est pas appelle dans la chaine */

int CPP_GenAllHomFloat(int argc,char ** argv)
{
     return  PreGenerateDuTriplet(argc,argv,"NO_OneHomFloat");
}

int CPP_GenAllImP3(int argc,char ** argv)
{
     return  PreGenerateDuTriplet(argc,argv,"NO_OneImTriplet");
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
